"""Thin command entry points; scientific work lives in reusable modules."""
import argparse
import json
from pathlib import Path
import shutil

from .config import execution_environment, load_config, project_path
from .runs import Run

STAGES=('generate','audit','claims','retrieval','nli','taxonomy')
REFERENCE=('prepare-reference','sample-reference','review-html','export-reference','suggest-reference','audit-qwen','translate-reference')


def parser(command):
    p=argparse.ArgumentParser(description=f'Phase I: {command}. Every command creates an immutable run directory.')
    p.add_argument('--config',help='YAML override, relative to project root or absolute')
    p.add_argument('--input','--entrada','--dados',dest='input',help='Input dataset or intermediate artifact')
    p.add_argument('--output-root',default='outputs/runs',help='Parent directory for new runs')
    p.add_argument('--run-id',help='Optional unique directory name; existing names are refused')
    p.add_argument('--reuse-from',help='Previous run; reuse requires matching config, input, code, environment and hashes')
    p.add_argument('--device',choices=('auto','cuda','cuda:0','cpu'))
    p.add_argument('--start',type=int,help='Initial document position, inclusive')
    selection=p.add_mutually_exclusive_group()
    selection.add_argument('--num-docs',type=int,help='Number of documents (not document/variant rows)')
    selection.add_argument('--all-docs',action='store_true',help='Process every document after --start')
    if command=='pipeline':
        p.add_argument('--from-stage',choices=('generate','claims','retrieval','nli','taxonomy'),default='claims')
        p.add_argument('--until-stage',choices=('generate','claims','retrieval','nli','taxonomy'),default='taxonomy')
    if command=='export-reference':
        p.add_argument('--annotations','--anotacoes',required=True)
        p.add_argument('--include-outside-sample',action='store_true')
    if command=='prepare-reference':
        p.add_argument('--backend',choices=('hf','tfidf'))
        p.add_argument('--claim-field',choices=('claim_original','claim_resolvida','texto'))
        p.add_argument('--max-claims',type=int)
    if command=='sample-reference':
        p.add_argument('--sample-size','--amostra',type=int)
        p.add_argument('--seed',type=int)
    if command=='translate-reference':
        p.add_argument('--model',default='qwen3.5:9b')
        p.add_argument('--ollama-host',default='http://127.0.0.1:11434')
        p.add_argument('--timeout',type=float,default=180)
        p.add_argument('--cache',default='outputs/translation-cache.sqlite3')
        p.add_argument('--glossary',help='Optional JSON glossary; included in cache fingerprint')
        p.add_argument('--scope',choices=('sample','all'),default='sample')
        p.add_argument('--max-calls',type=int,default=0,help='New segment calls; 0 means unlimited')
        p.add_argument('--batch-size',type=int,default=20,help='Progress reporting interval (units); calls are sequential')
    if command=='review-html':
        p.add_argument('--translations',help='Translation artifact directory or translations.jsonl')
    if command in ('suggest-reference','audit-qwen'):
        p.add_argument('--model')
        p.add_argument('--max-claims',type=int,default=0)
    if command=='suggest-reference':
        p.add_argument('--rank',type=int,default=1)
    return p


def selected_parquet(path, folder, start, num_docs):
    if not start and num_docs is None:
        return path
    import pandas as pd
    frame=pd.read_parquet(path)
    if 'doc_idx' not in frame:
        raise ValueError('Document selection requires doc_idx')
    docs=list(dict.fromkeys(frame['doc_idx']))
    chosen=docs[start:None if num_docs is None else start+num_docs]
    if not chosen:
        raise ValueError('Document interval is empty')
    output=folder/'selected_input.parquet'
    frame[frame['doc_idx'].isin(chosen)].to_parquet(output,index=False,engine='pyarrow')
    return output


def counts(path):
    import pandas as pd
    frame=pd.read_parquet(path) if Path(path).suffix=='.parquet' else pd.read_csv(path)
    result={'rows':len(frame)}
    if 'doc_idx' in frame:
        result['documents']=int(frame['doc_idx'].nunique())
    if 'claims' in frame:
        result['claims']=sum(len(x) for x in frame['claims'])
    if 'status_factual_nli' in frame:
        result['status_counts']={str(k):int(v) for k,v in frame['status_factual_nli'].value_counts().items()}
    return result


def execute_stage(stage, input_path, folder, config, start=0, num_docs=None, all_docs=False):
    if stage=='generate':
        from .data.injection import run
        return run(folder,config,start,num_docs,input_path,all_docs=all_docs)
    if stage=='audit':
        from .data.validation import audit_dataframe
        from .schemas import read_injected_csv
        frame=read_injected_csv(input_path).iloc[start:None if num_docs is None else start+num_docs]
        if frame.empty:
            raise ValueError('Document interval is empty')
        result=folder/'technical_audit.json'
        result.write_text(json.dumps(audit_dataframe(frame),ensure_ascii=False,indent=2))
        return result
    output=folder/f'{stage}.parquet'
    if stage=='claims':
        if config['claims']['variant']=='enriched':
            from .claims.enriched import run
        else:
            from .claims.extraction import run
        run(input_path,output,config,start,num_docs)
        from .schemas import extraction_claims,write_jsonl
        labels=[{k:r[k] for k in ('claim_id','doc_id','generator_label','is_gold')} for r in extraction_claims(output)]
        write_jsonl(folder/'generator_labels.jsonl',labels)
    else:
        input_path=selected_parquet(input_path,folder,start,num_docs)
        if stage=='retrieval':
            from .retrieval.pipeline import run
        elif stage=='nli':
            from .nli.evaluation import run
        elif stage=='taxonomy':
            from .taxonomy.classification import run
        else:
            raise ValueError(f'Unknown stage: {stage}')
        run(input_path,output,config)
    report={'input':counts(input_path),'output':counts(output),
            'label_policy':'gabarito_injetado is legacy generator metadata, never independent human gold.'}
    (folder/'stage_counts.json').write_text(json.dumps(report,indent=2))
    return output


def reference_stage(command, input_path, folder, config, args):
    if command=='prepare-reference':
        from argparse import Namespace
        from .reference.preparation import run
        opts=dict(config['preparation'])
        # Revisions are supported by the migrated implementation and pinned in config.
        opts.update(entrada=str(input_path),saida=str(folder))
        run(Namespace(**opts))
        return folder/'manifest.json'
    if command=='sample-reference':
        from .reference.sampling import select
        # Preserve raw IDs and original payloads; sampling lives with its exact frame.
        for p in Path(input_path).iterdir():
            if p.is_file() and p.name in ('manifest.json','documentos.jsonl','sentencas.jsonl','claims.jsonl','candidatos.jsonl'):
                shutil.copyfile(p,folder/p.name)
        s=config['sampling']
        select(folder,s['sample_size'],(s['fraction_a'],s['fraction_b'],s['fraction_c']),s['seed'])
        return folder/'plano_amostragem.json'
    if command=='review-html':
        from .reference.html import run
        return run(input_path,folder/'avaliacao.html',project_path(args.translations) if args.translations else None)
    if command=='translate-reference':
        from .reference.translation import OllamaTranslator, translate_reference
        translator=OllamaTranslator(args.model,args.ollama_host,args.timeout)
        glossary=json.loads(project_path(args.glossary).read_text(encoding='utf-8')) if args.glossary else None
        return translate_reference(input_path,folder,project_path(args.cache),translator,
                                   scope=args.scope,max_calls=args.max_calls,batch_size=args.batch_size,glossary=glossary)
    if command=='export-reference':
        from .reference.export import validate
        validate(input_path,project_path(args.annotations),folder,args.include_outside_sample)
        return folder/'resumo_validacao.json'
    if command=='suggest-reference':
        from .reference.model_suggestions import run
        output=folder/'predictions.jsonl'
        run(input_path,output,args.model or 'MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli',
            rank=args.rank,max_claims=args.max_claims,device=args.device or 'auto')
        return output
    if command=='audit-qwen':
        import pandas as pd
        from .reference.qwen import auditar_dataframe,MODEL
        frame=pd.read_parquet(input_path)
        if args.max_claims:
            frame=frame.iloc[:args.max_claims]
        result=auditar_dataframe(frame,modelo=args.model or MODEL)
        result.to_parquet(folder/'predictions.parquet',index=False)
        result.to_csv(folder/'predictions.csv',index=False)
        return folder/'predictions.parquet'
    raise ValueError(command)


def main(command=None, argv=None):
    if command is None:
        import sys
        top=argparse.ArgumentParser(description='Factual evaluation, Phase I')
        top.add_argument('command',choices=STAGES+REFERENCE+('pipeline',))
        arguments=list(sys.argv[1:] if argv is None else argv)
        ns=top.parse_args(arguments[:1])
        return main(ns.command,arguments[1:])
    p=parser(command)
    args=p.parse_args(argv)
    if (args.start is not None and args.start<0) or (args.num_docs is not None and args.num_docs<1):
        p.error('--start must be >=0 and --num-docs must be >=1')
    reference=command in REFERENCE and command!='audit-qwen'
    config=load_config(args.config,reference=reference)
    if args.device:
        if reference:
            config['preparation']['device']=args.device
        else:
            config['runtime']['device']=args.device
    if command=='prepare-reference':
        for arg,key in [('backend','backend'),('claim_field','campo_claim'),('max_claims','max_claims')]:
            if getattr(args,arg) is not None:
                config['preparation'][key]=getattr(args,arg)
    if command=='sample-reference':
        for key in ('sample_size','seed'):
            if getattr(args,key) is not None:
                config['sampling'][key]=getattr(args,key)
    input_path=project_path(args.input) if args.input else None
    if command=='pipeline':
        order=['generate','claims','retrieval','nli','taxonomy']
        first,last=order.index(args.from_stage),order.index(args.until_stage)
        if first>last:
            p.error('--until-stage precedes --from-stage')
        stages=order[first:last+1]
    else:
        stages=[command]
    if args.start is None:
        args.start=0 if reference else config['selection']['start']
    if args.num_docs is None and not args.all_docs:
        if stages[0]=='claims':
            args.num_docs=config['selection']['num_docs']
        elif stages[0]=='generate':
            args.num_docs=config['injection']['num_docs']
    if input_path is None and stages[0]!='generate':
        p.error('--input is required')
    if input_path is not None and not input_path.exists():
        p.error(f'Input does not exist: {input_path}')
    reuse=project_path(args.reuse_from) if args.reuse_from else None
    with Run(project_path(args.output_root),config,run_id=args.run_id,reuse_from=reuse) as run:
        if not reference and command!='audit-qwen':
            execution_environment(config)
        for index,stage in enumerate(stages):
            start=args.start if index==0 else 0
            count=args.num_docs if index==0 else None
            parameters={'start':start,'num_docs':count,'all_docs':args.all_docs if index==0 else False}
            if command=='export-reference':
                from .runs import sha256
                parameters.update(annotations_sha256=sha256(project_path(args.annotations)),include_outside_sample=args.include_outside_sample)
            if command in ('suggest-reference','audit-qwen'):
                parameters.update(model=args.model,max_claims=args.max_claims,rank=getattr(args,'rank',None))
            if command=='review-html' and args.translations:
                from .runs import file_manifest
                parameters['translations']=file_manifest(project_path(args.translations).parent if project_path(args.translations).is_file() else project_path(args.translations))
            if command in REFERENCE:
                if start or count is not None:
                    raise ValueError('Reference document subsets must be prepared explicitly; IDs/frame are never silently resampled')
                operation=lambda folder: reference_stage(command,input_path,folder,config,args)
            else:
                operation=lambda folder: execute_stage(stage,input_path,folder,config,start,count,all_docs=args.all_docs if index==0 else False)
            # Translation cache is mutable and checked per segment, outside run reuse.
            can_reuse=stage!='translate-reference' and not (stage=='generate' and input_path is None)
            input_path=run.stage(stage,input_path,operation,parameters=parameters,allow_reuse=can_reuse)
        print('Run:',run.path)
        print('Output:',input_path)
    return 0
