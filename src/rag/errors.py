"""Erreurs publiques Mistral, sans détails susceptibles de contenir un secret."""
import httpx


class MistralError(RuntimeError):
    """Échec réel de l'API ; ne constitue jamais une réponse générée."""
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def translate_error(error: Exception) -> MistralError | None:
    """Classe les erreurs HTTP et transport sans masquer les erreurs de code."""
    response = getattr(error, 'response', None)
    status = getattr(error, 'status_code', None)
    if status is None and response is not None:
        status = response.status_code
    if status == 429:
        return MistralError('Limite Mistral atteinte (HTTP 429). Vérifiez les limites et l’usage du workspace. Aucune nouvelle tentative automatique.', 429)
    if status in (401, 403):
        return MistralError('Accès Mistral refusé : vérifiez la clé et les droits du workspace.', status)
    if status is not None:
        return MistralError(f'Échec Mistral HTTP {status}. Vérifiez le modèle configuré et la disponibilité du service.', status)
    if isinstance(error, httpx.TimeoutException):
        return MistralError('Délai de réponse Mistral dépassé.')
    if isinstance(error, httpx.TransportError):
        return MistralError('Connexion à Mistral impossible. Vérifiez le réseau et le proxy.')
    return None


def invoke_safely(operation, *args, **kwargs):
    """Exécute une seule opération et conserve les erreurs non reconnues."""
    try:
        return operation(*args, **kwargs)
    except Exception as error:
        public_error = translate_error(error)
        if public_error is None:
            raise
        raise public_error from None
