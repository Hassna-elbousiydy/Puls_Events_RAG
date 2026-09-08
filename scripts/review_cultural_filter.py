"""Validation manuelle du filtre culturel Puls-Events.

Ce script génère un échantillon reproductible d'événements :
- classés comme culturels ;
- classés comme non culturels.

L'objectif est de vérifier la pertinence du filtre culturel avant
de considérer le pré-processing comme validé.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from preprocess_openagenda import (
    TARGET_REGION,
    extract_json_label,
    get_reference_date,
    is_cultural_event,
    parse_and_filter_timings,
)


RAW_PATH = Path(
    "data/raw/evenements-publics-openagenda.csv"
)

OUTPUT_PATH = Path(
    "reports/generated/cultural_filter_review.csv"
)

SAMPLE_SIZE = 30
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    """Analyse les arguments de ligne de commande."""

    parser = argparse.ArgumentParser(
        description="Revue du filtre culturel Puls-Events."
    )

    parser.add_argument(
        "--reference-date",
        default=None,
        help=(
            "Date de référence au format YYYY-MM-DD. "
            "Par défaut : date actuelle en Europe/Paris."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Génère un échantillon culturel et non culturel."""

    args = parse_args()

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {RAW_PATH}"
        )

    reference_date = get_reference_date(
        args.reference_date
    )

    cutoff_date = (
        reference_date
        - pd.DateOffset(years=1)
    )

    print("=" * 75)
    print("REVUE DU FILTRE CULTUREL - PULS-EVENTS")
    print("=" * 75)

    print(
        f"\nDate de référence : {reference_date}"
    )

    print(
        f"Début historique  : {cutoff_date}"
    )

    # ---------------------------------------------------------------
    # Chargement
    # ---------------------------------------------------------------

    df = pd.read_csv(
        RAW_PATH,
        sep=";",
        encoding="utf-8-sig",
        encoding_errors="replace",
        low_memory=False,
    )

    print(
        f"\nÉvénements bruts : {len(df):,}"
    )

    # ---------------------------------------------------------------
    # Même périmètre géographique que le pré-processing
    # ---------------------------------------------------------------

    region_mask = (
        df["location_region"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq(TARGET_REGION)
    )

    df = df.loc[
        region_mask
    ].copy()

    # ---------------------------------------------------------------
    # Même périmètre temporel
    # ---------------------------------------------------------------

    df["eligible_timings"] = (
        df["timings"]
        .apply(
            lambda value: parse_and_filter_timings(
                value,
                cutoff_date,
            )
        )
    )

    df = df.loc[
        df["eligible_timings"]
        .map(len)
        .gt(0)
    ].copy()

    # ---------------------------------------------------------------
    # Même filtre sur le statut
    # ---------------------------------------------------------------

    df["status_clean"] = (
        df["status"]
        .apply(extract_json_label)
    )

    df = df.loc[
        ~df["status_clean"]
        .str.casefold()
        .eq("annulé")
    ].copy()

    print(
        "Événements admissibles avant filtre culturel : "
        f"{len(df):,}"
    )

    # ---------------------------------------------------------------
    # Application du filtre culturel
    # ---------------------------------------------------------------

    df["is_cultural"] = (
        df.apply(
            is_cultural_event,
            axis=1,
        )
    )

    cultural = df.loc[
        df["is_cultural"]
    ].copy()

    non_cultural = df.loc[
        ~df["is_cultural"]
    ].copy()

    print(
        f"Classés culturels     : {len(cultural):,}"
    )

    print(
        f"Classés non culturels : {len(non_cultural):,}"
    )

    # ---------------------------------------------------------------
    # Échantillons reproductibles
    # ---------------------------------------------------------------

    cultural_sample = cultural.sample(
        n=min(
            SAMPLE_SIZE,
            len(cultural),
        ),
        random_state=RANDOM_STATE,
    ).copy()

    non_cultural_sample = non_cultural.sample(
        n=min(
            SAMPLE_SIZE,
            len(non_cultural),
        ),
        random_state=RANDOM_STATE,
    ).copy()

    cultural_sample["filter_decision"] = (
        "CULTUREL"
    )

    non_cultural_sample["filter_decision"] = (
        "NON_CULTUREL"
    )

    review = pd.concat(
        [
            cultural_sample,
            non_cultural_sample,
        ],
        ignore_index=True,
    )

    # Colonnes utiles à la vérification humaine.
    review_columns = [
        "uid",
        "filter_decision",
        "title_fr",
        "originagenda_title",
        "keywords_fr",
        "description_fr",
        "location_city",
        "location_department",
        "location_region",
        "status_clean",
    ]

    review = review[
        review_columns
    ]

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    # ---------------------------------------------------------------
    # Affichage console
    # ---------------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("30 EVENEMENTS CLASSES CULTURELS")
    print("=" * 75)

    for _, row in cultural_sample.iterrows():

        print(
            f"\n- {row['title_fr']}"
        )

        print(
            "  Agenda : "
            f"{row['originagenda_title']}"
        )

        print(
            "  Mots-clés : "
            f"{row['keywords_fr']}"
        )

    print("\n")
    print("=" * 75)
    print("30 EVENEMENTS CLASSES NON CULTURELS")
    print("=" * 75)

    for _, row in non_cultural_sample.iterrows():

        print(
            f"\n- {row['title_fr']}"
        )

        print(
            "  Agenda : "
            f"{row['originagenda_title']}"
        )

        print(
            "  Mots-clés : "
            f"{row['keywords_fr']}"
        )

    print("\n")
    print("=" * 75)

    print(
        "Rapport enregistré dans : "
        f"{OUTPUT_PATH}"
    )

    print(
        "REVUE DU FILTRE CULTUREL : TERMINEE"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()