"""Vérifie l'accès réel à l'API Mistral."""

import os
import sys

from dotenv import load_dotenv
from mistralai.client import Mistral


def main():
    """Teste l'authentification Mistral sans lancer de génération."""

    load_dotenv()

    api_key = os.getenv("MISTRAL_API_KEY")

    if not api_key:
        print("ERREUR : MISTRAL_API_KEY absente du fichier .env")
        sys.exit(1)

    try:
        with Mistral(api_key=api_key) as client:
            response = client.models.list()

        models = getattr(response, "data", [])

        print("=" * 60)
        print("PULS-EVENTS RAG - VERIFICATION API MISTRAL")
        print("=" * 60)
        print(f"Modeles accessibles : {len(models)}")
        print()
        print("ACCES MISTRAL : OK")

    except Exception as exc:
        print("ACCES MISTRAL : ECHEC")
        print(f"Erreur : {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
