"""Smoke test du pipeline RAG complet Puls-Events."""

from src.rag.pipeline import PulsEventsRAG


QUESTION = (
    "Quelle exposition à Nantes parle de la Libération "
    "et de la Seconde Guerre mondiale ?"
)

CITY = "Nantes"


def main() -> None:
    """Teste retrieval FAISS + LangChain + Mistral."""

    print("=" * 78)
    print("TEST PIPELINE RAG COMPLET - PULS-EVENTS")
    print("=" * 78)

    rag = PulsEventsRAG()

    result = rag.ask(
    question=QUESTION,
    city=CITY,
    top_events=3,
    )

    events = result["events"]
    answer = result["answer"]

    if not events:
        raise RuntimeError(
            "Le pipeline RAG n'a récupéré aucun événement."
        )

    if not answer.strip():
        raise RuntimeError(
            "Le pipeline RAG a retourné une réponse vide."
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

    print(f"\nQuestion : {QUESTION}")
    print(f"Ville    : {CITY}")
    print(f"Sources  : {len(events)} événement(s)")

    print("\n" + "=" * 78)
    print("ÉVÉNEMENTS RÉCUPÉRÉS")
    print("=" * 78)

    for rank, event in enumerate(
        events,
        start=1,
    ):
        print(
            f"\n{rank}. {event['title']}"
        )
        print(
            f"   UID   : {event['uid']}"
        )
        print(
            f"   Ville : {event['city']}"
        )
        print(
            f"   Score : {event['best_score']:.6f}"
        )

    print("\n" + "=" * 78)
    print("RÉPONSE MISTRAL")
    print("=" * 78)

    print(f"\n{answer}")

    print("\n" + "=" * 78)
    print("PIPELINE RAG COMPLET : OK")
    print("=" * 78)


if __name__ == "__main__":
    main()