"""Smoke test LangChain minimal ; toute erreur reste un échec explicite."""
from src.rag.config import make_chat
from src.rag.errors import invoke_safely, MistralError


def main() -> int:
    """Effectue un seul appel, sans retrieval ni retry."""
    try:
        response = invoke_safely(make_chat(max_tokens=5).invoke, 'Réponds uniquement par le mot OK.')
        if response.content.strip() != 'OK':
            raise MistralError('Réponse inattendue au test minimal.')
        print('TEST MISTRAL MINIMAL : OK')
        return 0
    except MistralError as error:
        print(f'BLOQUÉ / ÉCHEC : {error}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
