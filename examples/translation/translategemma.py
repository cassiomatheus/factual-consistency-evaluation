#!/usr/bin/env python3
"""Traduz exemplos em chamadas individuais com TranslateGemma no Ollama."""
import argparse


MODEL = "translategemma:12b"
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


def traduzir(texto):
    from ollama import chat

    prompt = (
        "You are a professional English (en) to "
        "Brazilian Portuguese (pt-BR) translator. "
        "Your goal is to accurately convey the meaning "
        "and nuances of the original English text while "
        "adhering to Brazilian Portuguese grammar, "
        "vocabulary, and cultural sensitivities.\n"
        "Produce only the Brazilian Portuguese translation, "
        "without any additional explanations or commentary. "
        "Please translate the following English text "
        "into Brazilian Portuguese:\n\n\n"
        f"{texto}\n"
    )
    response = chat(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "num_ctx": 4096, "num_predict": 512},
    )
    return response.message.content.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--texto", type=int, choices=range(1, len(TEXTOS) + 1), nargs='+', required=True,
        help="Textos a traduzir (1 a 4); cada texto usa uma chamada independente.",
    )
    args = parser.parse_args(argv)
    print(f"Modelo: {MODEL}", flush=True)
    falhas = 0
    for indice in args.texto:
        texto = " ".join(TEXTOS[indice - 1].split())
        print(f"\n{'=' * 60}\nTEXTO {indice}\n{'=' * 60}", flush=True)
        try:
            print(traduzir(texto), flush=True)
        except Exception as erro:
            print(f"Erro no texto {indice}: {erro}", flush=True)
            falhas += 1
    return 1 if falhas else 0


if __name__ == "__main__":
    raise SystemExit(main())
