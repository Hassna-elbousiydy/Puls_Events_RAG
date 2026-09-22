"""Configuration partagée du chat Mistral."""
import os
import httpx
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

DEFAULT_CHAT_MODEL = 'ministral-8b-2512'


def make_chat(model_name: str | None = None, max_tokens: int = 600):
    """Construit le client compatible avec langchain-mistralai 1.1.6, sans retry."""
    load_dotenv()
    key = os.getenv('MISTRAL_API_KEY')
    if not key:
        raise RuntimeError('MISTRAL_API_KEY absente. Configurez votre environnement local.')
    # httpx charge les certificats SSL_CERT_FILE/SSL_CERT_DIR du système.
    # Certains environnements avec proxy TLS ont besoin de cette configuration,
    # que le contexte SSL interne de LangChain peut ignorer. Vérification TLS active.
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    client = httpx.Client(base_url='https://api.mistral.ai/v1', headers=headers, timeout=30)
    async_client = httpx.AsyncClient(base_url='https://api.mistral.ai/v1', headers=headers, timeout=30)
    return ChatMistralAI(client=client, async_client=async_client,model=model_name or os.getenv('MISTRAL_CHAT_MODEL') or DEFAULT_CHAT_MODEL,
                         api_key=key, temperature=0, max_tokens=max_tokens,
                         max_retries=0, timeout=30)
