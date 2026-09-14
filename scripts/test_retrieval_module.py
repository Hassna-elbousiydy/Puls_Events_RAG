"""Smoke test du module de retrieval Puls-Events."""

from src.rag.retrieval import EventRetriever


QUESTION = (
    "Quelle exposition à Nantes parle de la Libération "
    "et de la Seconde Guerre mondiale ?"
)

CITY = "Nantes"


def main() -> None:
    """Teste le module réutilisable de retrieval."""

    print("=" * 78)
    print("TEST MODULE RETRIEVAL - PULS-EVENTS")
    print("=" * 78)

    retriever = EventRetriever()

    events, context = retriever.retrieve_context(
        question=QUESTION,
        city=CITY,
        top_events=5,
    )

    if not events:
        raise RuntimeError(
            "Aucun événement retourné."
        )

    for event in events:
        if (
            str(event["city"])
            .strip()
            .casefold()
            != CITY.casefold()
        ):
            raise RuntimeError(
                "Le filtre de ville n'a pas été respecté."
            )

    print(
        f"\nVecteurs FAISS : "
        f"{retriever.vectorstore.index.ntotal:,}"
    )

    print(
        f"Question       : {QUESTION}"
    )

    print(
        f"Ville          : {CITY}"
    )

    print(
        f"Événements     : {len(events)}"
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "RESULTATS"
    )

    print(
        "=" * 78
    )

    for rank, event in enumerate(
        events,
        start=1,
    ):
        print(
            f"\n{rank}. {event['title']}"
        )

        print(
            f"   UID        : {event['uid']}"
        )

        print(
            f"   Ville      : {event['city']}"
        )

        print(
            f"   Score L2   : "
            f"{event['best_score']:.6f}"
        )

        print(
            f"   Chunks     : "
            f"{event['total_chunks']}"
        )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "APERÇU DU CONTEXTE"
    )

    print(
        "=" * 78
    )

    print(
        context[:2000]
    )

    print(
        "\n..."
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        "MODULE RETRIEVAL : OK"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()