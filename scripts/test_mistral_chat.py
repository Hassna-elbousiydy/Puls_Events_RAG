"""Smoke test du modèle de génération Mistral pour Puls-Events.

Ce test vérifie que le modèle :
- reçoit une question utilisateur ;
- reçoit un contexte documentaire imposé ;
- répond uniquement à partir du contexte ;
- indique clairement lorsqu'une information n'est pas disponible.

Aucune recherche FAISS n'est effectuée ici.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI


MODEL_NAME = "mistral-small-2603"


CONTEXT = """
Titre : Exposition « Libres ! 1944-1947 »

Description :
Une exposition des Archives de Nantes sur la fin de la
Seconde Guerre mondiale dans la métropole nantaise.

Détails :
À l'occasion du 80e anniversaire de la Libération,
les Archives de Nantes présentent une exposition sur
la fin de la guerre, la Libération et les premières
années du retour de la République à Nantes.

Lieu :
Cours Franklin Roosevelt, Nantes,
Loire-Atlantique, Pays de la Loire.

Dates :
26 juin - 1 octobre 2025.
""".strip()


QUESTION = (
    "Quelle exposition à Nantes parle de la Libération "
    "et de la Seconde Guerre mondiale ?"
)


def main() -> None:
    """Teste une génération Mistral à partir d'un contexte contrôlé."""

    load_dotenv()

    api_key = os.getenv(
        "MISTRAL_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY est absente du fichier .env."
        )

    llm = ChatMistralAI(
        model=MODEL_NAME,
        temperature=0,
        max_retries=5,
        api_key=api_key,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
Tu es Puls-Events, un assistant de recommandation
d'événements culturels.

Tu dois répondre uniquement à partir du contexte fourni.

Règles :
- n'invente aucune information ;
- si le contexte ne permet pas de répondre, dis-le clairement ;
- réponds en français ;
- sois concis ;
- utilise le titre, le lieu et les dates lorsqu'ils sont disponibles.
""".strip(),
            ),
            (
                "human",
                """
CONTEXTE :

{context}

QUESTION :

{question}
""".strip(),
            ),
        ]
    )

    chain = (
        prompt
        | llm
    )

    print("=" * 75)
    print("TEST GENERATION MISTRAL - PULS-EVENTS")
    print("=" * 75)

    print(
        f"\nModèle   : {MODEL_NAME}"
    )

    print(
        f"\nQuestion : {QUESTION}"
    )

    response = chain.invoke(
        {
            "context": CONTEXT,
            "question": QUESTION,
        }
    )

    answer = str(
        response.content
    ).strip()

    if not answer:
        raise RuntimeError(
            "Mistral a retourné une réponse vide."
        )

    print(
        "\n--- REPONSE ---\n"
    )

    print(
        answer
    )

    print(
        "\n--- METADONNEES ---"
    )

    print(
        response.response_metadata
    )

    print(
        "\n"
        + "=" * 75
    )

    print(
        "GENERATION MISTRAL : OK"
    )

    print(
        "=" * 75
    )


if __name__ == "__main__":
    main()