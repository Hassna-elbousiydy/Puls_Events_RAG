"""Diagnostic des limites API Mistral.

Effectue une requête minimale et affiche uniquement :
- le statut HTTP ;
- les en-têtes liés aux rate limits ;
- le message d'erreur retourné par Mistral.

La clé API n'est jamais affichée.
"""

from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv


MODEL = "mistral-small-2603"


def main() -> None:
    """Teste directement l'API Chat Mistral."""

    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY absente du fichier .env."
        )

    response = httpx.post(
        "https://api.mistral.ai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "Réponds uniquement par OK.",
                }
            ],
            "max_tokens": 5,
            "temperature": 0,
        },
        timeout=30,
    )

    print("=" * 70)
    print("DIAGNOSTIC API MISTRAL")
    print("=" * 70)

    print(f"\nModèle      : {MODEL}")
    print(f"HTTP status : {response.status_code}")

    print("\n--- HEADERS RATE LIMIT ---")

    found_header = False

    for key, value in response.headers.items():

        if (
            "ratelimit" in key.lower()
            or "retry-after" in key.lower()
        ):
            print(f"{key}: {value}")
            found_header = True

    if not found_header:
        print("Aucun header rate-limit exposé.")

    print("\n--- REPONSE API ---")

    try:
        print(response.json())
    except ValueError:
        print(response.text)

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()