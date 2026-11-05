#!/usr/bin/env python3
"""Teste independente de tradução dos exemplos de inglês para português."""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


MODEL_ID = "Helsinki-NLP/opus-mt-tc-big-en-pt"

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


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Modelo: {MODEL_ID} | Dispositivo: {device}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
    model = model.to(device).eval()

    for indice, texto in enumerate(TEXTOS, start=1):
        # >>pob<< seleciona português brasileiro.
        entrada = tokenizer(
            ">>pob<< " + " ".join(texto.split()),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        with torch.inference_mode():
            tokens = model.generate(
                **entrada,
                num_beams=4,
                max_new_tokens=384,
            )

        traducao = tokenizer.decode(tokens[0], skip_special_tokens=True)

        print(f"\n{'=' * 60}")
        print(f"TEXTO {indice}")
        print(f"{'=' * 60}")
        print(traducao, flush=True)


if __name__ == "__main__":
    main()
