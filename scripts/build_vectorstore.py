"""Construction de la base vectorielle FAISS de Puls-Events.

Pipeline :
1. charge les chunks JSONL ;
2. calcule les embeddings avec Mistral ``mistral-embed`` ;
3. construit progressivement un index FAISS IndexFlatL2 ;
4. conserve le texte et les métadonnées de chaque chunk ;
5. sauvegarde l'index localement ;
6. recharge l'index afin de vérifier sa validité ;
7. génère un rapport JSON de construction.

Le script accepte ``--limit`` afin de tester la chaîne complète sur
un petit nombre de chunks avant la construction de l'index complet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import faiss
from dotenv import load_dotenv
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings


CHUNKS_PATH = Path(
    "data/processed/events_chunks.jsonl"
)

DEFAULT_OUTPUT_DIR = Path(
    "vectorstore/faiss_index"
)

DEFAULT_REPORT_PATH = Path(
    "reports/generated/vectorstore_build_report.json"
)

MODEL_NAME = "mistral-embed"

EXPECTED_DIMENSION = 1024

DEFAULT_BATCH_SIZE = 500


def parse_args() -> argparse.Namespace:
    """Lit les arguments de ligne de commande."""

    parser = argparse.ArgumentParser(
        description=(
            "Construit l'index FAISS Puls-Events "
            "à partir des chunks OpenAgenda."
        )
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Nombre maximal de chunks à indexer. "
            "À utiliser pour les smoke tests."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=(
            "Nombre de chunks traités par lot côté script. "
            "LangChain réalise ensuite son propre batching Mistral."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire de sauvegarde de l'index FAISS.",
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Chemin du rapport JSON.",
    )

    return parser.parse_args()


def sha256_file(
    path: Path,
) -> str:
    """Calcule le SHA-256 d'un fichier."""

    sha256 = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            sha256.update(
                block
            )

    return sha256.hexdigest()


def load_chunks(
    path: Path,
) -> list[dict[str, Any]]:
    """Charge les chunks depuis le fichier JSONL."""

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier de chunks introuvable : {path}"
        )

    records: list[
        dict[str, Any]
    ] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(
                    line
                )

            except json.JSONDecodeError as error:
                raise RuntimeError(
                    "JSON invalide à la ligne "
                    f"{line_number}: {error}"
                ) from error

            records.append(
                record
            )

    if not records:
        raise RuntimeError(
            "Aucun chunk n'a été chargé."
        )

    chunk_ids = [
        str(record["chunk_id"])
        for record in records
    ]

    if len(chunk_ids) != len(
        set(chunk_ids)
    ):
        raise RuntimeError(
            "Des chunk_id dupliqués sont présents."
        )

    return records


def build_metadata(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Construit les métadonnées stockées avec un chunk."""

    return {
        key: value
        for key, value in record.items()
        if key != "content"
    }


def validate_vectors(
    vectors: list[list[float]],
    expected_count: int,
) -> int:
    """Vérifie le nombre, la dimension et la validité des vecteurs."""

    if len(vectors) != expected_count:
        raise RuntimeError(
            "Nombre de vecteurs incorrect : "
            f"{len(vectors)} au lieu de {expected_count}."
        )

    if not vectors:
        raise RuntimeError(
            "Mistral n'a retourné aucun vecteur."
        )

    dimension = len(
        vectors[0]
    )

    if dimension != EXPECTED_DIMENSION:
        raise RuntimeError(
            "Dimension inattendue : "
            f"{dimension} au lieu de "
            f"{EXPECTED_DIMENSION}."
        )

    for vector_index, vector in enumerate(
        vectors
    ):

        if len(vector) != dimension:
            raise RuntimeError(
                "Dimension incohérente pour le vecteur "
                f"{vector_index}."
            )

        if not all(
            math.isfinite(value)
            for value in vector
        ):
            raise RuntimeError(
                "Valeur non finie détectée dans le vecteur "
                f"{vector_index}."
            )

    return dimension


def create_empty_vectorstore(
    embeddings: MistralAIEmbeddings,
    dimension: int,
) -> FAISS:
    """Crée un vector store FAISS vide utilisant IndexFlatL2."""

    index = faiss.IndexFlatL2(
        dimension
    )

    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )


def main() -> None:
    """Construit et valide l'index vectoriel FAISS."""

    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size doit être strictement positif."
        )

    if (
        args.limit is not None
        and args.limit <= 0
    ):
        raise ValueError(
            "--limit doit être strictement positif."
        )

    load_dotenv()

    api_key = os.getenv(
        "MISTRAL_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY est absente du fichier .env."
        )

    all_records = load_chunks(
        CHUNKS_PATH
    )

    total_available = len(
        all_records
    )

    if args.limit is not None:

        records = all_records[
            : args.limit
        ]

    else:

        records = all_records

    total_to_index = len(
        records
    )

    unique_events = len(
        {
            str(record["uid"])
            for record in records
        }
    )

    print("=" * 75)
    print("CONSTRUCTION VECTORSTORE FAISS - PULS-EVENTS")
    print("=" * 75)

    print(
        f"\nModèle embedding       : {MODEL_NAME}"
    )

    print(
        f"Dimension attendue     : {EXPECTED_DIMENSION}"
    )

    print(
        "Type index FAISS       : IndexFlatL2"
    )

    print(
        f"Chunks disponibles     : {total_available:,}"
    )

    print(
        f"Chunks à indexer       : {total_to_index:,}"
    )

    print(
        f"Événements représentés : {unique_events:,}"
    )

    print(
        f"Taille lot script      : {args.batch_size}"
    )

    print(
        f"Répertoire sortie      : {args.output_dir}"
    )

    final_output_dir = args.output_dir
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir = Path(tempfile.mkdtemp(prefix=".building-", dir=final_output_dir.parent))

    embeddings = MistralAIEmbeddings(
        model=MODEL_NAME,
        api_key=api_key,
        max_retries=0,
        timeout=120,
    )

    vectorstore: FAISS | None = None

    start_time = time.perf_counter()

    processed = 0

    number_of_batches = math.ceil(
        total_to_index
        / args.batch_size
    )

    print(
        "\nDébut de la vectorisation..."
    )

    for batch_number, start in enumerate(
        range(
            0,
            total_to_index,
            args.batch_size,
        ),
        start=1,
    ):

        end = min(
            start + args.batch_size,
            total_to_index,
        )

        batch = records[
            start:end
        ]

        texts = [
            str(record["content"])
            for record in batch
        ]

        ids = [
            str(record["chunk_id"])
            for record in batch
        ]

        metadatas = [
            build_metadata(
                record
            )
            for record in batch
        ]

        print(
            "\nLot "
            f"{batch_number}/{number_of_batches} "
            f"— chunks {start + 1:,} à {end:,}"
        )

        batch_start = time.perf_counter()

        vectors = embeddings.embed_documents(
            texts
        )

        dimension = validate_vectors(
            vectors=vectors,
            expected_count=len(texts),
        )

        if vectorstore is None:

            vectorstore = create_empty_vectorstore(
                embeddings=embeddings,
                dimension=dimension,
            )

        vectorstore.add_embeddings(
            text_embeddings=list(
                zip(
                    texts,
                    vectors,
                )
            ),
            metadatas=metadatas,
            ids=ids,
        )

        processed = end

        batch_duration = (
            time.perf_counter()
            - batch_start
        )

        print(
            f"Lot terminé en {batch_duration:.1f} s"
        )

        print(
            f"Progression : "
            f"{processed:,}/{total_to_index:,} "
            f"({processed / total_to_index:.1%})"
        )

        print(
            f"Vecteurs dans FAISS : "
            f"{vectorstore.index.ntotal:,}"
        )

    if vectorstore is None:
        raise RuntimeError(
            "Le vectorstore n'a pas été créé."
        )

    if processed != total_to_index:
        raise RuntimeError(
            "Tous les chunks n'ont pas été traités."
        )

    if (
        vectorstore.index.ntotal
        != total_to_index
    ):
        raise RuntimeError(
            "Le nombre de vecteurs FAISS "
            "ne correspond pas au nombre de chunks."
        )

    if len(
        vectorstore.index_to_docstore_id
    ) != total_to_index:
        raise RuntimeError(
            "Le mapping FAISS/docstore est incomplet."
        )

    # -----------------------------------------------------------------
    # Sauvegarde
    # -----------------------------------------------------------------

    print(
        "\nSauvegarde de l'index..."
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectorstore.save_local(
        str(
            args.output_dir
        )
    )

    faiss_file = (
        args.output_dir
        / "index.faiss"
    )

    pickle_file = (
        args.output_dir
        / "index.pkl"
    )

    if not faiss_file.exists():
        raise RuntimeError(
            "index.faiss n'a pas été créé."
        )

    if not pickle_file.exists():
        raise RuntimeError(
            "index.pkl n'a pas été créé."
        )

    # -----------------------------------------------------------------
    # Rechargement de contrôle
    # -----------------------------------------------------------------

    print(
        "Rechargement de contrôle..."
    )

    reloaded = FAISS.load_local(
        str(
            args.output_dir
        ),
        embeddings,
        allow_dangerous_deserialization=True,
    )

    if (
        reloaded.index.ntotal
        != total_to_index
    ):
        raise RuntimeError(
            "Le nombre de vecteurs après rechargement "
            "est incorrect."
        )

    if len(
        reloaded.index_to_docstore_id
    ) != total_to_index:
        raise RuntimeError(
            "Le docstore rechargé est incomplet."
        )

    if final_output_dir.exists():
        backup = final_output_dir.with_name(final_output_dir.name + ".previous-" + str(time.time_ns()))
        final_output_dir.rename(backup)
        try:
            args.output_dir.rename(final_output_dir)
        except Exception:
            backup.rename(final_output_dir)
            raise
    else:
        args.output_dir.rename(final_output_dir)
    args.output_dir = final_output_dir
    faiss_file = final_output_dir / "index.faiss"
    pickle_file = final_output_dir / "index.pkl"

    elapsed_seconds = (
        time.perf_counter()
        - start_time
    )

    # -----------------------------------------------------------------
    # Rapport
    # -----------------------------------------------------------------

    report = {
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "source_file": str(
            CHUNKS_PATH
        ),
        "source_sha256": sha256_file(
            CHUNKS_PATH
        ),
        "embedding_model": MODEL_NAME,
        "embedding_dimension": (
            EXPECTED_DIMENSION
        ),
        "faiss_index_type": (
            "IndexFlatL2"
        ),
        "distance_metric": (
            "euclidean_l2"
        ),
        "chunks_available": (
            total_available
        ),
        "chunks_indexed": (
            total_to_index
        ),
        "unique_events_indexed": (
            unique_events
        ),
        "faiss_ntotal": int(
            reloaded.index.ntotal
        ),
        "docstore_entries": len(
            reloaded.index_to_docstore_id
        ),
        "batch_size": int(
            args.batch_size
        ),
        "limit": (
            args.limit
        ),
        "output_directory": str(
            args.output_dir
        ),
        "index_faiss_bytes": (
            faiss_file.stat().st_size
        ),
        "index_pickle_bytes": (
            pickle_file.stat().st_size
        ),
        "elapsed_seconds": round(
            elapsed_seconds,
            2,
        ),
        "reload_validation": True,
    }

    args.report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n" + "=" * 75)
    print("CONSTRUCTION FAISS : SUCCES")
    print("=" * 75)

    print(
        f"\nChunks indexés     : {total_to_index:,}"
    )

    print(
        f"Événements uniques : {unique_events:,}"
    )

    print(
        f"Vecteurs FAISS     : {reloaded.index.ntotal:,}"
    )

    print(
        f"Dimension          : {EXPECTED_DIMENSION}"
    )

    print(
        "Index              : IndexFlatL2"
    )

    print(
        f"Durée              : {elapsed_seconds:.1f} s"
    )

    print(
        f"Index sauvegardé   : {args.output_dir}"
    )

    print(
        f"Rapport            : {args.report_path}"
    )


if __name__ == "__main__":
    main()