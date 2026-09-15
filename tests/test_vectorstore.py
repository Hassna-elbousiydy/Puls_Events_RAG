"""Tests de qualité de la base vectorielle FAISS Puls-Events.

Ces tests vérifient que :
- l'index FAISS complet existe et peut être rechargé ;
- tous les chunks attendus sont présents ;
- tous les événements sont représentés ;
- la dimension et le type de l'index sont corrects ;
- les textes et métadonnées correspondent aux chunks sources ;
- le périmètre géographique est Pays de la Loire ;
- la règle temporelle d'un an d'historique + événements à venir est respectée ;
- aucun événement annulé n'est présent ;
- la recherche FAISS fonctionne.

Aucun appel à l'API Mistral n'est effectué par ces tests.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import faiss
import numpy as np
import pandas as pd
import pytest
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

VECTORSTORE_PATH = Path(
    "vectorstore/faiss_index"
)

FAISS_PATH = (
    VECTORSTORE_PATH
    / "index.faiss"
)

PICKLE_PATH = (
    VECTORSTORE_PATH
    / "index.pkl"
)

CHUNKS_PATH = Path(
    "data/processed/events_chunks.jsonl"
)

REPORT_PATH = Path(
    "reports/generated/vectorstore_build_report.json"
)

EXPECTED_DIMENSION = 1024

EXPECTED_REGION = "Pays de la Loire"

EXPECTED_INDEX_TYPE = "IndexFlatL2"

REFERENCE_DATE = os.getenv(
    "PULS_REFERENCE_DATE",
    pd.Timestamp.now(tz="Europe/Paris").date().isoformat(),
)


# ---------------------------------------------------------------------
# Embeddings hors ligne
# ---------------------------------------------------------------------

class OfflineEmbeddings(Embeddings):
    """Embeddings factices utilisés uniquement pour recharger FAISS.

    Le chargement d'un index LangChain demande un objet Embeddings,
    mais aucun embedding n'est calculé pendant ces tests.
    """

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Interdit tout appel d'embedding pendant les tests."""

        raise RuntimeError(
            "Aucun embedding ne doit être calculé "
            "pendant test_vectorstore.py."
        )

    def embed_query(
        self,
        text: str,
    ) -> list[float]:
        """Interdit tout appel d'embedding pendant les tests."""

        raise RuntimeError(
            "Aucun embedding ne doit être calculé "
            "pendant test_vectorstore.py."
        )


# ---------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------

def sha256_file(
    path: Path,
) -> str:
    """Calcule le SHA-256 d'un fichier."""

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file:

        while True:

            block = file.read(
                1024 * 1024
            )

            if not block:
                break

            digest.update(
                block
            )

    return digest.hexdigest()


def historical_cutoff_utc() -> pd.Timestamp:
    """Retourne la limite historique d'un an en UTC.

    Exemple :
    référence 2026-09-08 en Europe/Paris
    -> cutoff 2025-09-08 00:00 Europe/Paris
    -> conversion UTC.
    """

    reference = datetime.strptime(
        REFERENCE_DATE,
        "%Y-%m-%d",
    ).replace(
        tzinfo=ZoneInfo(
            "Europe/Paris"
        )
    )

    cutoff = (
        pd.Timestamp(reference)
        - pd.DateOffset(years=1)
    )

    return cutoff.tz_convert(
        "UTC"
    )


# ---------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------

@pytest.fixture(scope="session")
def chunks() -> list[dict]:
    """Charge tous les chunks sources."""

    if not CHUNKS_PATH.exists():
        pytest.fail(
            f"Fichier absent : {CHUNKS_PATH}"
        )

    records = []

    with CHUNKS_PATH.open(
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

                pytest.fail(
                    "JSON invalide à la ligne "
                    f"{line_number} : {error}"
                )

            records.append(
                record
            )

    return records


@pytest.fixture(scope="session")
def chunk_by_id(
    chunks: list[dict],
) -> dict[str, dict]:
    """Indexe les chunks sources par chunk_id."""

    return {
        str(record["chunk_id"]): record
        for record in chunks
    }


@pytest.fixture(scope="session")
def build_report() -> dict:
    """Charge le rapport de construction de FAISS."""

    if not REPORT_PATH.exists():
        pytest.fail(
            f"Rapport absent : {REPORT_PATH}"
        )

    with REPORT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(
            file
        )


@pytest.fixture(scope="session")
def vectorstore() -> FAISS:
    """Recharge l'index FAISS sans appeler Mistral."""

    if not VECTORSTORE_PATH.exists():
        pytest.fail(
            "Répertoire FAISS absent : "
            f"{VECTORSTORE_PATH}"
        )

    embeddings = OfflineEmbeddings()

    return FAISS.load_local(
        str(VECTORSTORE_PATH),
        embeddings,
        allow_dangerous_deserialization=True,
    )


# ---------------------------------------------------------------------
# Tests fichiers / rapport
# ---------------------------------------------------------------------

def test_vectorstore_files_exist() -> None:
    """Vérifie que les deux fichiers LangChain/FAISS existent."""

    assert FAISS_PATH.exists()

    assert PICKLE_PATH.exists()

    assert FAISS_PATH.stat().st_size > 0

    assert PICKLE_PATH.stat().st_size > 0


def test_build_report_matches_source(
    chunks: list[dict],
    build_report: dict,
) -> None:
    """Vérifie que le rapport correspond aux chunks actuels."""

    assert (
        build_report[
            "source_sha256"
        ]
        == sha256_file(
            CHUNKS_PATH
        )
    )

    assert (
        build_report[
            "chunks_available"
        ]
        == len(chunks)
    )

    assert (
        build_report[
            "chunks_indexed"
        ]
        == len(chunks)
    )

    assert (
        build_report[
            "limit"
        ]
        is None
    )

    assert (
        build_report[
            "reload_validation"
        ]
        is True
    )


# ---------------------------------------------------------------------
# Tests structure FAISS
# ---------------------------------------------------------------------

def test_faiss_vector_count(
    vectorstore: FAISS,
    chunks: list[dict],
) -> None:
    """Vérifie qu'un vecteur existe pour chaque chunk."""

    assert (
        vectorstore.index.ntotal
        == len(chunks)
    )

    assert (
        vectorstore.index.ntotal
        == len(chunks)
    )


def test_faiss_dimension(
    vectorstore: FAISS,
) -> None:
    """Vérifie la dimension Mistral des vecteurs."""

    assert (
        vectorstore.index.d
        == EXPECTED_DIMENSION
    )


def test_faiss_uses_l2_metric(
    vectorstore: FAISS,
) -> None:
    """Vérifie que l'index utilise la distance L2."""

    assert (
        vectorstore.index.metric_type
        == faiss.METRIC_L2
    )


def test_faiss_index_type(
    vectorstore: FAISS,
    build_report: dict,
) -> None:
    """Vérifie le type d'index utilisé."""

    assert (
        build_report[
            "faiss_index_type"
        ]
        == EXPECTED_INDEX_TYPE
    )

    assert (
        "IndexFlat"
        in type(
            vectorstore.index
        ).__name__
    )


def test_docstore_count(
    vectorstore: FAISS,
    chunks: list[dict],
) -> None:
    """Vérifie qu'un document accompagne chaque vecteur."""

    assert (
        len(
            vectorstore.index_to_docstore_id
        )
        == len(chunks)
    )


# ---------------------------------------------------------------------
# Tests couverture
# ---------------------------------------------------------------------

def test_all_chunk_ids_are_indexed(
    vectorstore: FAISS,
    chunk_by_id: dict[str, dict],
) -> None:
    """Vérifie que tous les chunk_id attendus sont dans FAISS."""

    expected_ids = set(
        chunk_by_id
    )

    indexed_ids = set(
        str(value)
        for value
        in vectorstore.index_to_docstore_id.values()
    )

    assert indexed_ids == expected_ids


def test_all_events_are_indexed(
    vectorstore: FAISS,
    chunks: list[dict],
) -> None:
    """Vérifie que les 13 380 événements sont représentés."""

    event_uids = set()

    for document_id in (
        vectorstore
        .index_to_docstore_id
        .values()
    ):

        document = (
            vectorstore
            .docstore
            .search(
                document_id
            )
        )

        event_uids.add(
            str(
                document.metadata[
                    "uid"
                ]
            )
        )

    assert event_uids == {str(record["uid"]) for record in chunks}


# ---------------------------------------------------------------------
# Tests correspondance contenu / métadonnées
# ---------------------------------------------------------------------

def test_indexed_documents_match_chunks(
    vectorstore: FAISS,
    chunk_by_id: dict[str, dict],
) -> None:
    """Vérifie texte et métadonnées essentielles de chaque chunk."""

    fields = [
        "uid",
        "chunk_id",
        "chunk_index",
        "chunk_count",
        "title",
        "eligible_first_begin",
        "eligible_last_end",
        "region",
        "department",
        "location_text",
        "canonical_url",
        "source_agenda",
    ]

    mismatches = []

    for document_id in (
        vectorstore
        .index_to_docstore_id
        .values()
    ):

        document_id = str(
            document_id
        )

        source = chunk_by_id[
            document_id
        ]

        document = (
            vectorstore
            .docstore
            .search(
                document_id
            )
        )

        if (
            document.page_content
            != source["content"]
        ):

            mismatches.append(
                (
                    document_id,
                    "content",
                )
            )

            if len(mismatches) >= 20:
                break

        for field in fields:

            indexed_value = (
                document.metadata.get(
                    field
                )
            )

            source_value = (
                source.get(
                    field
                )
            )

            if indexed_value != source_value:

                mismatches.append(
                    (
                        document_id,
                        field,
                        indexed_value,
                        source_value,
                    )
                )

                if len(mismatches) >= 20:
                    break

        if len(mismatches) >= 20:
            break

    assert not mismatches, (
        "Différences détectées entre chunks "
        "et base vectorielle : "
        f"{mismatches}"
    )


# ---------------------------------------------------------------------
# Tests périmètre métier
# ---------------------------------------------------------------------

def test_all_indexed_events_are_in_target_region(
    vectorstore: FAISS,
) -> None:
    """Vérifie le périmètre Pays de la Loire."""

    invalid = []

    for document_id in (
        vectorstore
        .index_to_docstore_id
        .values()
    ):

        document = (
            vectorstore
            .docstore
            .search(
                document_id
            )
        )

        region = (
            str(
                document.metadata.get(
                    "region",
                    "",
                )
            )
            .strip()
        )

        if region != EXPECTED_REGION:

            invalid.append(
                (
                    document_id,
                    region,
                )
            )

            if len(invalid) >= 20:
                break

    assert not invalid, (
        "Documents hors Pays de la Loire : "
        f"{invalid}"
    )


def test_indexed_events_respect_one_year_history(
    vectorstore: FAISS,
) -> None:
    """Vérifie 1 an d'historique + événements à venir.

    Un événement est admissible si sa dernière occurrence
    éligible ne se termine pas avant la limite historique.
    """

    cutoff = historical_cutoff_utc()

    invalid = []

    for document_id in (
        vectorstore
        .index_to_docstore_id
        .values()
    ):

        document = (
            vectorstore
            .docstore
            .search(
                document_id
            )
        )

        raw_end = (
            document.metadata.get(
                "eligible_last_end"
            )
        )

        end = pd.to_datetime(
            raw_end,
            errors="coerce",
            utc=True,
        )

        if (
            pd.isna(end)
            or end < cutoff
        ):

            invalid.append(
                (
                    document_id,
                    raw_end,
                )
            )

            if len(invalid) >= 20:
                break

    assert not invalid, (
        "Événements hors fenêtre temporelle : "
        f"{invalid}"
    )


def test_no_cancelled_events_are_indexed(
    vectorstore: FAISS,
) -> None:
    """Vérifie qu'aucun événement annulé n'est indexé."""

    invalid = []

    for document_id in (
        vectorstore
        .index_to_docstore_id
        .values()
    ):

        document = (
            vectorstore
            .docstore
            .search(
                document_id
            )
        )

        status = (
            str(
                document.metadata.get(
                    "status_clean",
                    "",
                )
            )
            .strip()
            .casefold()
        )

        if status == "annulé":

            invalid.append(
                document_id
            )

            if len(invalid) >= 20:
                break

    assert not invalid


def test_location_metadata_is_available(
    vectorstore: FAISS,
) -> None:
    """Vérifie qu'une localisation est conservée pour chaque chunk."""

    missing = []

    for document_id in (
        vectorstore
        .index_to_docstore_id
        .values()
    ):

        document = (
            vectorstore
            .docstore
            .search(
                document_id
            )
        )

        location = (
            str(
                document.metadata.get(
                    "location_text",
                    "",
                )
            )
            .strip()
        )

        if not location:

            missing.append(
                document_id
            )

            if len(missing) >= 20:
                break

    assert not missing


# ---------------------------------------------------------------------
# Test du moteur FAISS sans API
# ---------------------------------------------------------------------

def test_faiss_self_search(
    vectorstore: FAISS,
) -> None:
    """Vérifie qu'un vecteur retrouve lui-même dans FAISS."""

    first_vector = (
        vectorstore.index.reconstruct(
            0
        )
    )

    query = np.asarray(
        [first_vector],
        dtype="float32",
    )

    distances, indices = (
        vectorstore.index.search(
            query,
            5,
        )
    )

    assert distances.shape == (
        1,
        5,
    )

    assert indices.shape == (
        1,
        5,
    )

    assert (
        indices[0][0]
        == 0
    )

    assert (
        distances[0][0]
        == pytest.approx(
            0.0,
            abs=1e-6,
        )
    )