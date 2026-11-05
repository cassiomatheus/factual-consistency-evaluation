#!/usr/bin/env python3
"""Explicitly protect reviewed translation segments in the mutable cache, never gold."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from factual_eval.config import project_path
from factual_eval.reference.html import jsonl
from factual_eval.reference.translation import Cache


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--translations', required=True, help='translations.jsonl from a finished run')
    parser.add_argument('--cache', default='outputs/translation-cache.sqlite3')
    parser.add_argument('--unit-id', action='append', required=True)
    parser.add_argument('--reviewer', required=True)
    args=parser.parse_args()
    if not args.reviewer.strip():
        parser.error('Revisor obrigatório')
    rows={r['unit_id']:r for r in jsonl(project_path(args.translations))}
    chosen=[rows[uid] for uid in args.unit_id]
    if any(r['status'] not in ('translated','needs_review','human_verified') for r in chosen):
        parser.error('Há unidades incompletas ou com falha')
    cache=Cache(project_path(args.cache))
    try:
        for row in chosen:
            for part in row['segments']:
                cache.verify(part['cache_key'],args.reviewer)
            print('Tradução verificada (não é rótulo gold):',row['unit_id'])
    finally:
        cache.close()


if __name__=='__main__':
    main()
