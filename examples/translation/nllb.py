#!/usr/bin/env python3
"""Teste independente de tradução de inglês para português com NLLB."""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_ID = "facebook/nllb-200-distilled-600M"
TEXTOS = [
    """EPA generally followed its established process for most
    of its 22 advisory committees. In fiscal year 2018,
    EPA did not follow a key step for appointing 20 committee
    members to two committees GAO reviewed: the EPA Science
    Advisory Board and Clean Air Scientific Advisory Committee.
    The committees advise the agency on environmental
    regulatory matters.""",

    """Only one of the 23 installations GAO visited or
    contacted had considered some extreme weather and climate
    change effects in their plans as required by DOD guidance.
    8 installations had not considered extreme weather and
    climate change effects in their plans.""",

    """GAO reviewed CMS's policies. GAO analyzed CMS data on
    abuse deficiencies from 2013 through 2017. The most
    recent data was at the time of our review. GAO interviewed
    officials from CMS and state survey agencies in five states.
    GAO interviewed other key stakeholders in those states
    such as ombudsmen and law enforcement officials.""",

    """Such a strategy could help EM in seeking alternatives
    to its selected treatment technology. Such a strategy could
    provide assurance that it will meet its commitments with
    Idaho for removing calcine waste from the state
    by the end of 2035.""",
]


def traduzir(texto, tokenizer, model, device):
    entradas = tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to(device)

    with torch.inference_mode():
        saida = model.generate(
            **entradas,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("por_Latn"),
            max_new_tokens=256,
            num_beams=4,
        )

    return tokenizer.decode(saida[0], skip_special_tokens=True)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Modelo: {MODEL_ID} | Dispositivo: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID).to(device).eval()

    print(traduzir(TEXTO, tokenizer, model, device), flush=True)


if __name__ == "__main__":
    main()
