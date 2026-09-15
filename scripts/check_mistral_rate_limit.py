"""Diagnostic direct reproductible : modèles puis un seul appel chat minimal."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import httpx
from dotenv import load_dotenv
from src.rag.config import DEFAULT_CHAT_MODEL
from src.rag.errors import invoke_safely, MistralError


def main() -> int:
    """Enregistre uniquement statut, modèle et code fournisseur ; jamais la clé."""
    load_dotenv()
    key = os.getenv('MISTRAL_API_KEY')
    if not key:
        print('BLOQUÉ : MISTRAL_API_KEY absente.'); return 2
    model = os.getenv('MISTRAL_CHAT_MODEL') or DEFAULT_CHAT_MODEL
    report = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'model': model,
              'method': 'HTTP direct, sans LangChain/FAISS, sans retry', 'status': 'failed'}
    try:
        with httpx.Client(timeout=30, headers={'Authorization': f'Bearer {key}'}) as client:
            models = invoke_safely(client.get, 'https://api.mistral.ai/v1/models')
            report['models_http'] = models.status_code
            invoke_safely(models.raise_for_status)
            report['model_available'] = any(m['id'] == model for m in models.json()['data'])
            if not report['model_available']:
                raise MistralError('Modèle configuré absent des modèles accessibles.')
            response = invoke_safely(client.post, 'https://api.mistral.ai/v1/chat/completions', json={
                'model': model, 'messages': [{'role': 'user', 'content': 'Réponds uniquement par le mot OK.'}],
                'temperature': 0, 'max_tokens': 5})
            report['chat_http'] = response.status_code
            if response.status_code == 429:
                report['provider_code'] = response.json().get('code')
                report['status'] = 'blocked_mistral_429'
            invoke_safely(response.raise_for_status)
            answer = response.json()['choices'][0]['message']['content']
            report['answer_is_OK'] = isinstance(answer, str) and answer.strip() == 'OK'
            if not report['answer_is_OK']:
                raise MistralError('Le test minimal n’a pas retourné exactement OK.')
            report['status'] = 'passed'
    except MistralError as error:
        report['error'] = str(error)
    path = Path('reports/generated/mistral_diagnostic.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['status'] == 'passed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
