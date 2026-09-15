"""Tests de qualité du chunking Puls-Events.

Ces tests vérifient que :
- tous les événements sources sont représentés ;
- aucun chunk n'est vide ;
- les identifiants de chunks sont uniques ;
- la taille maximale est respectée ;
- les métadonnées importantes sont conservées ;
- les chunks restent dans le périmètre Pays de la Loire ;
- les indices de chunks sont cohérents pour chaque événement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest


SOURCE_PATH = Path(
    "data/processed/events_processed.csv"
)

CHUNKS_PATH = Path(
    "data/processed/events_chunks.jsonl"
)

CHUNK_SIZE = 1500

TARGET_REGION = "Pays de la Loire"


@pytest.fixture(scope="session")
def source_events() -> pd.DataFrame:
    """Charge le dataset source pré-processé."""

    if not SOURCE_PATH.exists():
        pytest.fail(
            f"Dataset source introuvable : {SOURCE_PATH}"
        )

    df = pd.read_csv(
        SOURCE_PATH,
        low_memory=False,
    )

    df["uid"] = (
        df["uid"]
        .astype(str)
    )

    return df


@pytest.fixture(scope="session")
def chunks() -> list[dict]:
    """Charge tous les chunks JSONL."""

    if not CHUNKS_PATH.exists():
        pytest.fail(
            f"Fichier de chunks introuvable : {CHUNKS_PATH}"
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
def chunks_df(
    chunks: list[dict],
) -> pd.DataFrame:
    """Convertit les chunks en DataFrame."""

    return pd.DataFrame(
        chunks
    )


def test_chunks_file_is_not_empty(
    chunks: list[dict],
) -> None:
    """Vérifie que le chunking a produit des données."""

    assert chunks


def test_required_chunk_fields_exist(
    chunks_df: pd.DataFrame,
) -> None:
    """Vérifie les champs nécessaires au futur index FAISS."""

    required_fields = {
        "chunk_id",
        "uid",
        "chunk_index",
        "chunk_count",
        "content",
        "title",
        "eligible_first_begin",
        "eligible_last_end",
        "eligible_timings_json",
        "date_range",
        "location_name",
        "location_text",
        "city",
        "department",
        "region",
        "country_code",
        "latitude",
        "longitude",
        "attendance_mode",
        "status_clean",
        "registration",
        "canonical_url",
        "source_agenda",
    }

    missing_fields = (
        required_fields
        - set(chunks_df.columns)
    )

    assert not missing_fields, (
        "Champs manquants : "
        f"{sorted(missing_fields)}"
    )


def test_chunk_ids_are_unique(
    chunks_df: pd.DataFrame,
) -> None:
    """Vérifie que chaque chunk possède un identifiant unique."""

    assert (
        chunks_df["chunk_id"]
        .notna()
        .all()
    )

    assert (
        chunks_df["chunk_id"]
        .is_unique
    )


def test_all_source_events_are_represented(
    source_events: pd.DataFrame,
    chunks_df: pd.DataFrame,
) -> None:
    """Vérifie qu'aucun événement n'est perdu pendant le chunking."""

    source_uids = set(
        source_events["uid"]
        .astype(str)
    )

    chunk_uids = set(
        chunks_df["uid"]
        .astype(str)
    )

    missing_uids = (
        source_uids
        - chunk_uids
    )

    unexpected_uids = (
        chunk_uids
        - source_uids
    )

    assert not missing_uids, (
        "Événements absents des chunks : "
        f"{sorted(missing_uids)[:20]}"
    )

    assert not unexpected_uids, (
        "Chunks associés à des UID inconnus : "
        f"{sorted(unexpected_uids)[:20]}"
    )

    assert len(source_uids) == len(source_events)


def test_chunk_contents_are_not_empty(
    chunks_df: pd.DataFrame,
) -> None:
    """Vérifie qu'aucun contenu de chunk n'est vide."""

    contents = (
        chunks_df["content"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    assert contents.ne("").all()


def test_chunks_respect_maximum_size(
    chunks_df: pd.DataFrame,
) -> None:
    """Vérifie que la taille maximale de 1500 caractères est respectée."""

    lengths = (
        chunks_df["content"]
        .astype(str)
        .str.len()
    )

    oversized = chunks_df.loc[
        lengths.gt(CHUNK_SIZE),
        [
            "chunk_id",
            "uid",
            "title",
            "content",
        ],
    ]

    assert oversized.empty, (
        "Des chunks dépassent "
        f"{CHUNK_SIZE} caractères."
    )


def test_very_small_chunks_remain_exceptional(
    chunks_df: pd.DataFrame,
) -> None:
    """Vérifie que les très petits chunks restent exceptionnels.

    On accepte moins de 1 % de chunks sous 100 caractères afin
    de ne pas rendre le test trop dépendant d'un snapshot précis.
    """

    lengths = (
        chunks_df["content"]
        .astype(str)
        .str.len()
    )

    tiny_count = int(
        lengths.lt(100).sum()
    )

    tiny_ratio = (
        tiny_count
        / len(chunks_df)
    )

    assert tiny_ratio < 0.01, (
        f"Trop de petits chunks : "
        f"{tiny_count} "
        f"({tiny_ratio:.2%})"
    )


def test_chunk_counts_are_consistent(
    chunks_df: pd.DataFrame,
) -> None:
    """Vérifie que chunk_count correspond au nombre réel de chunks."""

    grouped = (
        chunks_df
        .groupby("uid")
    )

    invalid = []

    for uid, group in grouped:

        expected_count = len(
            group
        )

        declared_counts = set(
            group["chunk_count"]
            .astype(int)
        )

        if declared_counts != {
            expected_count
        }:
            invalid.append(
                (
                    str(uid),
                    expected_count,
                    sorted(
                        declared_counts
                    ),
                )
            )

    assert not invalid, (
        "chunk_count incohérent : "
        f"{invalid[:20]}"
    )


def test_chunk_indices_are_contiguous(
    chunks_df: pd.DataFrame,
) -> None:
    """Vérifie les indices 0, 1, 2... pour chaque événement."""

    invalid = []

    for uid, group in (
        chunks_df.groupby("uid")
    ):

        indices = sorted(
            group["chunk_index"]
            .astype(int)
            .tolist()
        )

        expected = list(
            range(
                len(group)
            )
        )

        if indices != expected:
            invalid.append(
                (
                    str(uid),
                    indices,
                    expected,
                )
            )

    assert not invalid, (
        "Indices de chunks incohérents : "
        f"{invalid[:20]}"
    )


def test_all_chunks_are_in_target_region(
    chunks_df: pd.DataFrame,
) -> None:
    """Vérifie que tous les chunks restent dans le périmètre choisi."""

    regions = (
        chunks_df["region"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    invalid = chunks_df.loc[
        regions.ne(TARGET_REGION),
        [
            "chunk_id",
            "uid",
            "title",
            "region",
        ],
    ]

    assert invalid.empty, (
        "Chunks hors Pays de la Loire :\n"
        f"{invalid.head(20)}"
    )


def test_location_metadata_is_available(
    chunks_df: pd.DataFrame,
) -> None:
    """Vérifie que chaque chunk conserve une localisation exploitable."""

    locations = (
        chunks_df["location_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    assert locations.ne("").all()


def test_titles_are_available(
    chunks_df: pd.DataFrame,
) -> None:
    """Vérifie que chaque chunk conserve son titre d'événement."""

    titles = (
        chunks_df["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    assert titles.ne("").all()


def test_dates_are_valid(
    chunks_df: pd.DataFrame,
) -> None:
    """Vérifie les bornes temporelles associées aux chunks."""

    first_begin = pd.to_datetime(
        chunks_df[
            "eligible_first_begin"
        ],
        errors="coerce",
        utc=True,
    )

    last_end = pd.to_datetime(
        chunks_df[
            "eligible_last_end"
        ],
        errors="coerce",
        utc=True,
    )

    assert first_begin.notna().all()

    assert last_end.notna().all()

    assert (
        first_begin
        <= last_end
    ).all()


def test_no_cancelled_event_in_chunks(
    chunks_df: pd.DataFrame,
) -> None:
    """Vérifie qu'aucun événement annulé n'arrive jusqu'aux chunks."""

    statuses = (
        chunks_df["status_clean"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    cancelled = chunks_df.loc[
        statuses.eq("annulé")
    ]

    assert cancelled.empty


def test_chunk_metadata_matches_source(
    source_events: pd.DataFrame,
    chunks_df: pd.DataFrame,
) -> None:
    """Vérifie que les métadonnées essentielles proviennent bien du CSV source."""

    source = (
        source_events
        .set_index("uid")
    )

    fields = [
        "title",
        "region",
        "department",
        "location_text",
        "canonical_url",
    ]

    mismatches = []

    for row in chunks_df.itertuples(
        index=False,
    ):

        uid = str(
            row.uid
        )

        source_row = source.loc[
            uid
        ]

        for field in fields:

            chunk_value = getattr(
                row,
                field,
            )

            source_value = source_row[
                field
            ]

            chunk_value = (
                ""
                if pd.isna(chunk_value)
                else str(chunk_value)
            )

            source_value = (
                ""
                if pd.isna(source_value)
                else str(source_value)
            )

            if chunk_value != source_value:

                mismatches.append(
                    (
                        uid,
                        field,
                        chunk_value,
                        source_value,
                    )
                )

                if len(mismatches) >= 20:
                    break

        if len(mismatches) >= 20:
            break

    assert not mismatches, (
        "Métadonnées différentes du dataset source : "
        f"{mismatches}"
    )