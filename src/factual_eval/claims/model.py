from typing import List

from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


class ClaimExtractor:
    def __init__(
        self,
        model_name: str = "Babelscape/t5-base-summarization-claim-extractor",
        device: str = "cuda:0",
        batch_size: int = 32,
        revision=None,
        max_input_length=256,
        max_output_length=128,
        num_beams=4,
        early_stopping=True,
    ):
        self.device = device
        # load model from HF
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, revision=revision).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, revision=revision)
        self.batch_size = batch_size
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.num_beams = num_beams
        self.early_stopping = early_stopping
        from factual_eval.provenance import record_model
        record_model("claims.extractor", model_name, self.model, revision)

    def process_batch(self, batch):
        # NOVA SINTAXE DO TOKENIZADOR
        tok_input = self.tokenizer(
            text=batch, 
            padding=True, 
            truncation=True, 
            max_length=self.max_input_length, # <--- Definido explicitamente
            return_tensors="pt"
        ).to(self.device)
        
        with torch.inference_mode():
            # Passamos os tensores dinamicamente para o modelo
            outputs = self.model.generate(
                **tok_input, 
                max_length=self.max_output_length, # <--- Limite de geração (um claim atômico é curto)
                num_beams=self.num_beams,
                early_stopping=self.early_stopping
            )
            
        # Decodificação
        claims_batch = []
        for i in range(outputs.shape[0]):
            output_text = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
            
            # O modelo geralmente separa os claims por <n> ou \n. 
            claims_limpos = [c.strip() for c in output_text.split('<n>') if c.strip()]
            
            if not claims_limpos:
                claims_limpos = [c.strip() for c in output_text.split('\n') if c.strip()]
                
            claims_batch.append(claims_limpos)
            
        return claims_batch
