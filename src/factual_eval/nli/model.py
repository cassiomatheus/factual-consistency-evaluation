from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from factual_eval.utils.batching import chunks


class NLIAligner:
    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
        batch_size: int = 32,
        device: str = "cuda:0",
        max_length: int = 512,
        revision=None,
        **kwargs,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)

        self.device = device
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, revision=revision).to(
            device
        )
        labels = {int(k):str(v).lower() for k,v in self.model.config.id2label.items()}
        if labels != {0:'entailment',1:'neutral',2:'contradiction'}:
            raise ValueError(f'Checkpoint label order is incompatible with the preserved NLIAligner: {labels}')
        self.batch_size = batch_size
        self.max_length = max_length
        from factual_eval.provenance import record_model
        record_model("nli.classifier", model_name, self.model, revision)
        
        # Aquece a GPU gradualmente para evitar o pico transiente de energia
        self.warmup_gpu()

    def score(self, premises, hypothesis, lower=False, disable_prog_bar: bool = False):
        if isinstance(hypothesis, str):
            hypothesis = [hypothesis] * len(premises)
        if lower:
            premises = [p.lower() for p in premises]
            hypothesis = [h.lower() for h in hypothesis]
        prem_hyp_pairs = list(zip(premises, hypothesis))
        return self.process_batch(prem_hyp_pairs, disable_prog_bar=disable_prog_bar)

    def process_batch(self, prem_hyp_pairs, disable_prog_bar=False):
        batches = list(chunks(prem_hyp_pairs, self.batch_size))
        all_entailment_probs = []
        all_contradiction_probs = []
        all_neutral_probs = []
        for batch in tqdm(
            batches,
            desc="Computing alignments...",
            total=len(batches),
            disable=disable_prog_bar,
        ):
            entailment_dists = self.score_sample(batch)
            all_entailment_probs.extend([d[0] for d in entailment_dists])
            all_neutral_probs.extend([d[1] for d in entailment_dists])
            all_contradiction_probs.extend(d[2] for d in entailment_dists)
        return all_entailment_probs, all_contradiction_probs, all_neutral_probs

    def score_sample(self, batch):
        # 1. Desempacota as tuplas nas listas de premissas e hipóteses (Exigência da nova API)
        premises = [pair[0] for pair in batch]
        hypotheses = [pair[1] for pair in batch]
        
        # 2. Chama o tokenizer na sintaxe moderna
        inputs = self.tokenizer(
            text=premises,
            text_pair=hypotheses,
            padding=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        
        # =====================================================================
        # VRAM FIX: inference_mode bloqueia retenção de histórico de gradiente
        # =====================================================================
        with torch.inference_mode():
            # Desempacota dinamicamente os tensores gerados para dentro do modelo
            outputs = self.model(**inputs)

        entailment_dists = torch.softmax(outputs.logits, dim=1)
        
        # =====================================================================
        # VRAM FIX: .cpu() descarrega as probabilidades imediatamente para a RAM
        # =====================================================================
        return entailment_dists.cpu()
    
    def warmup_gpu(self):
        """Executa pequenas inferências progressivas para aquecer a GPU suavemente."""
        dummy_pairs = [("Document context example.", "Hypothesis statement.")] * 32
        with torch.inference_mode():
            for size in [1, 4, 16, 32]:
                batch = dummy_pairs[:size]
                _ = self.score_sample(batch)
                if str(self.device).startswith("cuda"):
                    torch.cuda.synchronize()
