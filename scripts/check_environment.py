"""Vérification de l'environnement de développement Puls-Events RAG."""

import sys
from importlib.metadata import version

import faiss
import numpy as np
import langchain
import langchain_community
import mistralai

from langchain_community.vectorstores import FAISS
from langchain_mistralai import ChatMistralAI
from mistralai.client import Mistral


def main():
    """Vérifie les imports et réalise un petit test FAISS CPU."""

    print("=" * 60)
    print("PULS-EVENTS RAG - VERIFICATION ENVIRONNEMENT")
    print("=" * 60)

    print(f"Python              : {sys.version.split()[0]}")
    print(f"LangChain           : {version('langchain')}")
    print(f"LangChain Community : {version('langchain-community')}")
    print(f"LangChain Mistral   : {version('langchain-mistralai')}")
    print(f"Mistral SDK         : {version('mistralai')}")
    print(f"FAISS CPU           : {version('faiss-cpu')}")
    print(f"NumPy               : {version('numpy')}")

    # Petit test de recherche vectorielle FAISS CPU.
    vectors = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [10.0, 10.0],
        ],
        dtype="float32",
    )

    index = faiss.IndexFlatL2(2)
    index.add(vectors)

    query = np.array([[0.1, 0.1]], dtype="float32")
    distances, indices = index.search(query, 1)

    assert index.ntotal == 3
    assert int(indices[0][0]) == 0

    # Ces références vérifient que les intégrations sont importables.
    assert FAISS is not None
    assert ChatMistralAI is not None
    assert Mistral is not None

    print()
    print("Test FAISS CPU       : OK")
    print("Imports LangChain    : OK")
    print("Integration Mistral  : OK")
    print()
    print("ENVIRONNEMENT LOCAL : OK")


if __name__ == "__main__":
    main()
