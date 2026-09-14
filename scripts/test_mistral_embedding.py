"""Test minimal des embeddings Mistral pour Puls-Events.

Ce script :
- charge un seul chunk ;
- appelle le modèle mistral-embed ;
- vérifie que le résultat est un vecteur numérique valide.

Il ne construit pas encore l'index FAISS.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings


CHUNKS_PATH = Path(
    "data/processed/events_chunks.jsonl"
)

MODEL_NAME = "mistral-embed"


def main() -> None:
    """Teste l'embedding d'un seul chunk."""

    load_dotenv()

    api_key = os.getenv(
        "MISTRAL_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY est absente du fichier .env."
        )

    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {CHUNKS_PATH}"
        )

    with CHUNKS_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        first_line = file.readline()

    if not first_line.strip():
        raise RuntimeError(
            "Le fichier de chunks est vide."
        )

    record = json.loads(
        first_line
    )

    text = str(
        record["content"]
    ).strip()

    print("=" * 75)
    print("TEST EMBEDDING MISTRAL - PULS-EVENTS")
    print("=" * 75)

    print(
        f"\nModèle     : {MODEL_NAME}"
    )

    print(
        f"Chunk ID   : {record['chunk_id']}"
    )

    print(
        f"UID        : {record['uid']}"
    )

    print(
        f"Titre      : {record['title']}"
    )

    print(
        f"Caractères : {len(text)}"
    )

    embeddings = MistralAIEmbeddings(
        model=MODEL_NAME,
        api_key=api_key,
    )

    vectors = embeddings.embed_documents(
        [text]
    )

    if len(vectors) != 1:
        raise RuntimeError(
            "Mistral n'a pas retourné exactement un vecteur."
        )

    vector = vectors[0]

    if not vector:
        raise RuntimeError(
            "Le vecteur retourné est vide."
        )

    if not all(
        math.isfinite(value)
        for value in vector
    ):
        raise RuntimeError(
            "Le vecteur contient des valeurs invalides."
        )

    print("\n--- RESULTAT ---")

    print(
        f"Nombre de vecteurs : {len(vectors)}"
    )

    print(
        f"Dimension          : {len(vector)}"
    )

    print(
        "5 premières valeurs : "
        f"{vector[:5]}"
    )

    print(
        "\nEMBEDDING MISTRAL : OK"
    )


if __name__ == "__main__":
    main()