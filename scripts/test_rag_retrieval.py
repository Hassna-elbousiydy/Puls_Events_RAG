"""Test amélioré du retrieval RAG pour Puls-Events.

Ce script :
- charge l'index FAISS complet ;
- transforme une question avec mistral-embed ;
- applique un filtre géographique sur les métadonnées ;
- récupère les chunks sémantiquement les plus proches ;
- regroupe les résultats par événement ;
- récupère tous les chunks des événements sélectionnés ;
- construit un contexte complet prêt pour le LLM.

Aucun appel au modèle Chat Mistral n'est effectué ici.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

INDEX_PATH = Path(
    "vectorstore/faiss_index"
)

MODEL_NAME = "mistral-embed"

QUERY = (
    "Quelle exposition à Nantes parle de la Libération "
    "et de la Seconde Guerre mondiale ?"
)

CITY_FILTER = "Nantes"

# Nombre de chunks que l'on souhaite récupérer après filtrage.
SEARCH_K = 20

# FAISS examine davantage de voisins avant d'appliquer le filtre.
FETCH_K = 500

# Nombre final d'événements distincts.
TOP_EVENTS = 5


# ---------------------------------------------------------------------
# Utilitaires
# ---------------------------------------------------------------------

def get_all_event_documents(
    vectorstore: FAISS,
    uid: str,
) -> list:
    """Récupère tous les chunks d'un événement dans le docstore."""

    documents = []

    for document_id in (
        vectorstore
        .index_to_docstore_id
        .values()
    ):

        document = (
            vectorstore
            .docstore
            .search(
                document_id
            )
        )

        if (
            str(
                document.metadata.get(
                    "uid",
                    ""
                )
            )
            == uid
        ):
            documents.append(
                document
            )

    documents.sort(
        key=lambda document: int(
            document.metadata.get(
                "chunk_index",
                0,
            )
        )
    )

    return documents


def build_event_context(
    vectorstore: FAISS,
    event: dict,
    rank: int,
) -> str:
    """Construit le contexte complet d'un événement."""

    documents = get_all_event_documents(
        vectorstore=vectorstore,
        uid=event["uid"],
    )

    if not documents:
        raise RuntimeError(
            "Aucun document trouvé pour l'événement "
            f"{event['uid']}."
        )

    contents = []

    seen_contents = set()

    for document in documents:

        content = (
            document.page_content
            .strip()
        )

        if (
            content
            and content not in seen_contents
        ):
            contents.append(
                content
            )

            seen_contents.add(
                content
            )

    combined_content = (
        "\n".join(
            contents
        )
    )

    return (
        f"[ÉVÉNEMENT {rank}]\n"
        f"UID : {event['uid']}\n"
        f"Titre : {event['title']}\n"
        f"Ville : {event['city']}\n"
        f"Lieu : {event['location_text']}\n"
        f"Dates : {event['date_range']}\n"
        f"URL : {event['canonical_url']}\n"
        f"Informations :\n"
        f"{combined_content}"
    )


# ---------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------

def main() -> None:
    """Teste le retrieval filtré sur l'index FAISS complet."""

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
            f"Index FAISS introuvable : {INDEX_PATH}"
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

    print("=" * 78)
    print("TEST RETRIEVAL RAG FILTRE - PULS-EVENTS")
    print("=" * 78)

    print(
        f"\nVecteurs FAISS : "
        f"{vectorstore.index.ntotal:,}"
    )

    print(
        f"Question       : {QUERY}"
    )

    print(
        f"Filtre ville   : {CITY_FILTER}"
    )

    print(
        f"SEARCH_K       : {SEARCH_K}"
    )

    print(
        f"FETCH_K        : {FETCH_K}"
    )

    # -----------------------------------------------------------------
    # Recherche FAISS avec filtre metadata
    # -----------------------------------------------------------------

    raw_results = (
        vectorstore
        .similarity_search_with_score(
            QUERY,
            k=SEARCH_K,
            filter={
                "city": CITY_FILTER,
            },
            fetch_k=FETCH_K,
        )
    )

    if not raw_results:
        raise RuntimeError(
            "FAISS n'a retourné aucun résultat "
            "avec le filtre demandé."
        )

    # -----------------------------------------------------------------
    # Vérification stricte du filtre
    # -----------------------------------------------------------------

    invalid_cities = []

    for document, score in raw_results:

        city = str(
            document.metadata.get(
                "city",
                "",
            )
        ).strip()

        if city != CITY_FILTER:

            invalid_cities.append(
                city
            )

    if invalid_cities:
        raise RuntimeError(
            "Le filtre géographique n'a pas été respecté : "
            f"{invalid_cities}"
        )

    # -----------------------------------------------------------------
    # Regroupement par événement
    # -----------------------------------------------------------------

    grouped: OrderedDict[
        str,
        dict,
    ] = OrderedDict()

    for document, score in raw_results:

        metadata = (
            document.metadata
        )

        uid = str(
            metadata.get(
                "uid",
                "",
            )
        )

        if not uid:
            continue

        if uid not in grouped:

            grouped[uid] = {
                "uid": uid,
                "title": metadata.get(
                    "title"
                ),
                "city": metadata.get(
                    "city"
                ),
                "location_text": metadata.get(
                    "location_text"
                ),
                "date_range": metadata.get(
                    "date_range"
                ),
                "canonical_url": metadata.get(
                    "canonical_url"
                ),
                "source_agenda": metadata.get(
                    "source_agenda"
                ),
                "best_score": float(
                    score
                ),
                "matched_chunks": [],
            }

        grouped[
            uid
        ][
            "matched_chunks"
        ].append(
            {
                "chunk_id": metadata.get(
                    "chunk_id"
                ),
                "score": float(
                    score
                ),
            }
        )

        grouped[
            uid
        ][
            "best_score"
        ] = min(
            grouped[
                uid
            ][
                "best_score"
            ],
            float(
                score
            ),
        )

    # -----------------------------------------------------------------
    # Tri explicite par meilleur score
    # -----------------------------------------------------------------

    events = sorted(
        grouped.values(),
        key=lambda event: (
            event["best_score"]
        ),
    )

    events = events[
        :TOP_EVENTS
    ]

    if not events:
        raise RuntimeError(
            "Aucun événement distinct trouvé."
        )

    # -----------------------------------------------------------------
    # Affichage
    # -----------------------------------------------------------------

    print(
        "\n"
        + "=" * 78
    )

    print(
        f"TOP {len(events)} EVENEMENTS DISTINCTS"
    )

    print(
        "=" * 78
    )

    for rank, event in enumerate(
        events,
        start=1,
    ):

        full_documents = (
            get_all_event_documents(
                vectorstore=vectorstore,
                uid=event["uid"],
            )
        )

        print(
            "\n"
            + "-" * 78
        )

        print(
            f"Rang                 : {rank}"
        )

        print(
            f"Score L2             : "
            f"{event['best_score']:.6f}"
        )

        print(
            f"UID                  : "
            f"{event['uid']}"
        )

        print(
            f"Titre                : "
            f"{event['title']}"
        )

        print(
            f"Ville                : "
            f"{event['city']}"
        )

        print(
            f"Lieu                 : "
            f"{event['location_text']}"
        )

        print(
            f"Dates                : "
            f"{event['date_range']}"
        )

        print(
            "Chunks retrouvés FAISS : "
            f"{len(event['matched_chunks'])}"
        )

        print(
            "Chunks totaux événement : "
            f"{len(full_documents)}"
        )

        print(
            f"URL                  : "
            f"{event['canonical_url']}"
        )

    # -----------------------------------------------------------------
    # Construction du contexte RAG complet
    # -----------------------------------------------------------------

    context_blocks = []

    for rank, event in enumerate(
        events,
        start=1,
    ):

        block = build_event_context(
            vectorstore=vectorstore,
            event=event,
            rank=rank,
        )

        context_blocks.append(
            block
        )

    separator = (
        "\n\n"
        + "=" * 60
        + "\n\n"
    )

    rag_context = separator.join(
        context_blocks
    )

    # -----------------------------------------------------------------
    # Validation finale
    # -----------------------------------------------------------------

    for event in events:

        if (
            str(
                event["city"]
            ).strip()
            != CITY_FILTER
        ):
            raise RuntimeError(
                "Un événement hors Nantes "
                "est présent dans le Top final."
            )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "CONTEXTE RAG PRET POUR LE LLM"
    )

    print(
        "=" * 78
    )

    print(
        rag_context
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "RETRIEVAL RAG FILTRE : OK"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()