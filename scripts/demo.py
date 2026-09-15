"""Démo terminal : question, sources récupérées et génération Mistral."""
import argparse
from src.rag.pipeline import PulsEventsRAG
from src.rag.retrieval import EventRetriever
from src.rag.errors import MistralError, invoke_safely


def main() -> int:
    """Affiche les sources même si la génération est bloquée."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('question')
    parser.add_argument('--city')
    parser.add_argument('--start-date')
    parser.add_argument('--end-date')
    parser.add_argument('--index', default='vectorstore/faiss_index')
    args = parser.parse_args()
    try:
        retriever = EventRetriever(index_path=args.index)
        events, context = retriever.retrieve_context(args.question, city=args.city,
            start_date=args.start_date, end_date=args.end_date, top_events=3)
        print('QUESTION :', args.question)
        for event in events:
            print(f"{event['uid']} | {event['title']} | {event['city']} | {event['date_range']}\n{event['canonical_url']}")
        if not events:
            print('Aucun événement trouvé pour ces critères dans la base.'); return 0
        rag = PulsEventsRAG(retriever=retriever)
        response = invoke_safely(rag.chain.invoke, {'question': args.question, 'context': context})
        if not isinstance(response.content, str) or not response.content.strip():
            raise MistralError('Réponse Mistral vide ou non textuelle.')
        print('RÉPONSE MISTRAL :\n', response.content)
        return 0
    except MistralError as error:
        print(f'GÉNÉRATION NON VALIDÉE : {error}'); return 2


if __name__ == '__main__':
    raise SystemExit(main())
