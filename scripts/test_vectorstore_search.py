"""Smoke test de recherche sémantique dans l'index FAISS Puls-Events.

Ce script recharge le petit index FAISS de test, vectorise une question
avec Mistral et affiche les chunks les plus proches.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings


INDEX_PATH = Path(
    "vectorstore/faiss_smoke_test"
)

MODEL_NAME = "mistral-embed"

QUERY = (
    "Quelle exposition à Nantes parle de la Libération "
    "et de la Seconde Guerre mondiale ?"
)

TOP_K = 5


def main() -> None:
    """Recharge l'index et effectue une recherche sémantique."""

    load_dotenv()

    api_key = os.getenv(
        "MISTRAL_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY est absente du fichier .env."
        )

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Index introuvable : {INDEX_PATH}"
        )

    embeddings = MistralAIEmbeddings(
        model=MODEL_NAME,
        api_key=api_key,
    )

    vectorstore = FAISS.load_local(
        str(INDEX_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    print("=" * 75)
    print("TEST RECHERCHE SEMANTIQUE FAISS - PULS-EVENTS")
    print("=" * 75)

    print(
        f"\nVecteurs dans l'index : "
        f"{vectorstore.index.ntotal}"
    )

    if vectorstore.index.ntotal != 100:
        raise RuntimeError(
            "Le smoke index devrait contenir exactement "
            "100 vecteurs."
        )

    print(
        f"\nQuestion : {QUERY}"
    )

    results = (
        vectorstore.similarity_search_with_score(
            QUERY,
            k=TOP_K,
        )
    )

    if not results:
        raise RuntimeError(
            "La recherche n'a retourné aucun résultat."
        )

    print(
        f"\nTop {len(results)} résultats :"
    )

    for rank, (
        document,
        score,
    ) in enumerate(
        results,
        start=1,
    ):

        metadata = document.metadata

        print(
            "\n"
            + "-" * 75
        )

        print(
            f"Rang      : {rank}"
        )

        print(
            f"Score L2  : {score:.6f}"
        )

        print(
            "Chunk ID  : "
            f"{metadata.get('chunk_id')}"
        )

        print(
            f"UID       : "
            f"{metadata.get('uid')}"
        )

        print(
            "Titre     : "
            f"{metadata.get('title')}"
        )

        print(
            "Lieu      : "
            f"{metadata.get('location_text')}"
        )

        print(
            "Dates     : "
            f"{metadata.get('date_range')}"
        )

        excerpt = (
            document.page_content
            .replace(
                "\n",
                " ",
            )
            .strip()
        )

        print(
            "Extrait   : "
            f"{excerpt[:400]}"
        )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "RECHERCHE SEMANTIQUE FAISS : OK"
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":
    main()