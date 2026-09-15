"""Tests de qualité du dataset OpenAgenda pré-processé.

Ces tests vérifient que les événements préparés pour la future
indexation FAISS respectent les principales contraintes du POC :

- périmètre géographique : Pays de la Loire ;
- historique limité à un an et événements futurs ;
- absence d'événements annulés ;
- identifiants uniques ;
- contenu textuel exploitable ;
- localisation exploitable ;
- créneaux temporels valides ;
- exclusion des agendas explicitement hors périmètre culturel.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest


DATA_PATH = Path(
    "data/processed/events_processed.csv"
)

TARGET_REGION = "Pays de la Loire"

REFERENCE_DATE_STR = os.getenv(
    "PULS_REFERENCE_DATE",
    pd.Timestamp.now(tz="Europe/Paris").date().isoformat(),
)

REFERENCE_DATE = pd.Timestamp(
    REFERENCE_DATE_STR,
    tz="Europe/Paris",
).tz_convert("UTC")

HISTORY_CUTOFF = (
    REFERENCE_DATE
    - pd.DateOffset(years=1)
)

EXCLUDED_AGENDA_TERMS = [
    "mes événements france travail",
    "semaine de l'industrie",
    "chambre d'agriculture",
    "challenges geovelo",
    "semaine des métiers du tourisme",
    "journées nationales de l'agriculture",
]


@pytest.fixture(scope="session")
def events() -> pd.DataFrame:
    """Charge une seule fois le dataset pré-processé."""

    if not DATA_PATH.exists():
        pytest.fail(
            "Le dataset pré-processé est introuvable. "
            "Exécutez d'abord scripts/preprocess_openagenda.py."
        )

    return pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )


def test_dataset_is_not_empty(
    events: pd.DataFrame,
) -> None:
    """Vérifie que le pré-processing produit des événements."""

    assert not events.empty


def test_required_columns_exist(
    events: pd.DataFrame,
) -> None:
    """Vérifie la présence des colonnes indispensables au futur RAG."""

    required_columns = {
        "uid",
        "title",
        "description",
        "long_description",
        "text_for_embedding",
        "eligible_first_begin",
        "eligible_last_end",
        "eligible_timings_json",
        "location_text",
        "city",
        "department",
        "region",
        "latitude",
        "longitude",
        "status_clean",
        "source_agenda",
        "canonical_url",
    }

    missing_columns = (
        required_columns
        - set(events.columns)
    )

    assert not missing_columns, (
        "Colonnes manquantes : "
        f"{sorted(missing_columns)}"
    )


def test_uids_are_present_and_unique(
    events: pd.DataFrame,
) -> None:
    """Vérifie qu'un événement possède un UID unique."""

    assert events["uid"].notna().all()

    assert events["uid"].is_unique


def test_all_events_are_in_target_region(
    events: pd.DataFrame,
) -> None:
    """Vérifie que tous les événements appartiennent à Pays de la Loire."""

    regions = (
        events["region"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    invalid_regions = events.loc[
        regions.ne(TARGET_REGION),
        [
            "uid",
            "title",
            "region",
        ],
    ]

    assert invalid_regions.empty, (
        "Des événements sont hors de la région "
        f"{TARGET_REGION} :\n"
        f"{invalid_regions.head(10)}"
    )


def test_no_cancelled_event(
    events: pd.DataFrame,
) -> None:
    """Vérifie qu'aucun événement annulé n'est conservé."""

    statuses = (
        events["status_clean"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    cancelled = events.loc[
        statuses.eq("annulé"),
        [
            "uid",
            "title",
            "status_clean",
        ],
    ]

    assert cancelled.empty, (
        "Des événements annulés sont encore présents :\n"
        f"{cancelled.head(10)}"
    )


def test_titles_are_not_empty(
    events: pd.DataFrame,
) -> None:
    """Vérifie que tous les événements possèdent un titre."""

    titles = (
        events["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    assert titles.ne("").all()


def test_events_have_description_content(
    events: pd.DataFrame,
) -> None:
    """Vérifie qu'une description courte ou longue est disponible."""

    descriptions = (
        events["description"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    long_descriptions = (
        events["long_description"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    valid_content = (
        descriptions.ne("")
        | long_descriptions.ne("")
    )

    assert valid_content.all()


def test_embedding_text_is_not_empty(
    events: pd.DataFrame,
) -> None:
    """Vérifie que chaque événement possède un texte à vectoriser."""

    texts = (
        events["text_for_embedding"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    assert texts.ne("").all()


def test_location_text_is_not_empty(
    events: pd.DataFrame,
) -> None:
    """Vérifie que chaque événement possède une localisation exploitable."""

    locations = (
        events["location_text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    assert locations.ne("").all()


def test_event_has_city_or_coordinates(
    events: pd.DataFrame,
) -> None:
    """Vérifie qu'aucun événement n'est totalement sans localisation.

    Une ville peut être absente si des coordonnées géographiques
    valides sont disponibles.
    """

    city_available = (
        events["city"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    coordinates_available = (
        events["latitude"].notna()
        & events["longitude"].notna()
    )

    valid_location = (
        city_available
        | coordinates_available
    )

    invalid = events.loc[
        ~valid_location,
        [
            "uid",
            "title",
            "city",
            "latitude",
            "longitude",
            "location_text",
        ],
    ]

    assert invalid.empty, (
        "Événements sans ville ni coordonnées :\n"
        f"{invalid.head(10)}"
    )


def test_eligible_date_columns_are_valid(
    events: pd.DataFrame,
) -> None:
    """Vérifie que les bornes temporelles admissibles sont valides."""

    first_begin = pd.to_datetime(
        events["eligible_first_begin"],
        errors="coerce",
        utc=True,
    )

    last_end = pd.to_datetime(
        events["eligible_last_end"],
        errors="coerce",
        utc=True,
    )

    assert first_begin.notna().all()
    assert last_end.notna().all()

    assert (
        first_begin
        <= last_end
    ).all()


def test_events_respect_one_year_history(
    events: pd.DataFrame,
) -> None:
    """Vérifie la contrainte d'un an d'historique.

    Chaque événement doit posséder au moins un créneau dont la fin
    est postérieure ou égale à la borne d'un an avant la date
    de référence.
    """

    last_end = pd.to_datetime(
        events["eligible_last_end"],
        errors="coerce",
        utc=True,
    )

    invalid = events.loc[
        last_end.lt(HISTORY_CUTOFF),
        [
            "uid",
            "title",
            "eligible_last_end",
        ],
    ]

    assert invalid.empty, (
        "Des événements dépassent la limite "
        "d'un an d'historique :\n"
        f"{invalid.head(10)}"
    )


def test_every_eligible_timing_respects_cutoff(
    events: pd.DataFrame,
) -> None:
    """Contrôle chaque créneau temporel conservé individuellement."""

    invalid_timings = []

    for row in events.itertuples():

        try:
            timings = json.loads(
                row.eligible_timings_json
            )

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            invalid_timings.append(
                (
                    row.uid,
                    row.title,
                    "JSON invalide",
                )
            )
            continue

        if not isinstance(
            timings,
            list,
        ) or not timings:

            invalid_timings.append(
                (
                    row.uid,
                    row.title,
                    "aucun créneau",
                )
            )
            continue

        for timing in timings:

            begin = pd.to_datetime(
                timing.get("begin"),
                errors="coerce",
                utc=True,
            )

            end = pd.to_datetime(
                timing.get("end"),
                errors="coerce",
                utc=True,
            )

            if (
                pd.isna(begin)
                or pd.isna(end)
            ):
                invalid_timings.append(
                    (
                        row.uid,
                        row.title,
                        "date invalide",
                    )
                )
                continue

            if begin > end:
                invalid_timings.append(
                    (
                        row.uid,
                        row.title,
                        "begin > end",
                    )
                )

            if end < HISTORY_CUTOFF:
                invalid_timings.append(
                    (
                        row.uid,
                        row.title,
                        "créneau trop ancien",
                    )
                )

    assert not invalid_timings, (
        "Créneaux invalides détectés : "
        f"{invalid_timings[:10]}"
    )


def test_excluded_agendas_are_absent(
    events: pd.DataFrame,
) -> None:
    """Vérifie que les agendas explicitement exclus sont absents."""

    agendas = (
        events["source_agenda"]
        .fillna("")
        .astype(str)
        .str.casefold()
    )

    invalid_mask = pd.Series(
        False,
        index=events.index,
    )

    for term in EXCLUDED_AGENDA_TERMS:
        invalid_mask = (
            invalid_mask
            | agendas.str.contains(
                term.casefold(),
                regex=False,
                na=False,
            )
        )

    invalid = events.loc[
        invalid_mask,
        [
            "uid",
            "title",
            "source_agenda",
        ],
    ]

    assert invalid.empty, (
        "Des événements issus d'agendas explicitement "
        "exclus sont présents :\n"
        f"{invalid.head(10)}"
    )