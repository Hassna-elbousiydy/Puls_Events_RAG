"""Audit des données géographiques manquantes du dataset traité.

Ce script analyse les événements dont la ville ou les coordonnées
sont manquantes afin de décider d'une règle de nettoyage adaptée.

Il ne modifie jamais le dataset.
"""

from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/processed/events_processed.csv"
)

REPORT_PATH = Path(
    "reports/generated/missing_locations_review.csv"
)


def main() -> None:
    """Analyse les informations géographiques manquantes."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    print("=" * 75)
    print("AUDIT DES DONNEES GEOGRAPHIQUES MANQUANTES")
    print("=" * 75)

    print(
        f"\nNombre total d'événements : {len(df):,}"
    )

    # ---------------------------------------------------------------
    # Valeurs manquantes
    # ---------------------------------------------------------------

    fields = [
        "location_name",
        "address",
        "postal_code",
        "city",
        "department",
        "region",
        "country_code",
        "latitude",
        "longitude",
    ]

    print("\n")
    print("=" * 75)
    print("1. VALEURS MANQUANTES")
    print("=" * 75)

    for field in fields:

        if field not in df.columns:
            print(
                f"{field:20s} : COLONNE ABSENTE"
            )
            continue

        if df[field].dtype == object:

            missing = (
                df[field]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq("")
                .sum()
            )

        else:

            missing = (
                df[field]
                .isna()
                .sum()
            )

        percent = (
            missing
            / len(df)
            * 100
        )

        print(
            f"{field:20s} : "
            f"{missing:5,d} "
            f"({percent:6.2f} %)"
        )

    # ---------------------------------------------------------------
    # Événements sans ville
    # ---------------------------------------------------------------

    missing_city_mask = (
        df["city"]
        .fillna("")
        .astype(str)
        .str.strip()
        .eq("")
    )

    missing_city = df.loc[
        missing_city_mask
    ].copy()

    print("\n")
    print("=" * 75)
    print("2. EVENEMENTS SANS VILLE")
    print("=" * 75)

    print(
        f"\nNombre : {len(missing_city):,}"
    )

    if len(missing_city) > 0:

        city_fields = [
            "title",
            "location_name",
            "address",
            "postal_code",
            "department",
            "region",
            "latitude",
            "longitude",
            "source_agenda",
        ]

        print(
            "\nDisponibilité d'autres informations "
            "pour les événements sans ville :"
        )

        for field in [
            "location_name",
            "address",
            "postal_code",
            "department",
            "latitude",
            "longitude",
        ]:

            if missing_city[field].dtype == object:

                available = (
                    missing_city[field]
                    .fillna("")
                    .astype(str)
                    .str.strip()
                    .ne("")
                    .sum()
                )

            else:

                available = (
                    missing_city[field]
                    .notna()
                    .sum()
                )

            print(
                f"{field:20s} : "
                f"{available:3,d} / "
                f"{len(missing_city):3,d}"
            )

        print(
            "\n30 premiers événements sans ville :\n"
        )

        print(
            missing_city[
                city_fields
            ]
            .head(30)
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------------
    # Événements sans coordonnées
    # ---------------------------------------------------------------

    missing_coordinates_mask = (
        df["latitude"].isna()
        | df["longitude"].isna()
    )

    missing_coordinates = df.loc[
        missing_coordinates_mask
    ].copy()

    print("\n")
    print("=" * 75)
    print("3. EVENEMENTS SANS COORDONNEES")
    print("=" * 75)

    print(
        f"\nNombre : {len(missing_coordinates):,}"
    )

    if len(missing_coordinates) > 0:

        print(
            "\n"
            + missing_coordinates[
                [
                    "title",
                    "location_name",
                    "address",
                    "postal_code",
                    "city",
                    "department",
                    "region",
                    "source_agenda",
                ]
            ]
            .to_string(
                index=False
            )
        )

    # ---------------------------------------------------------------
    # Export pour contrôle
    # ---------------------------------------------------------------

    review = pd.concat(
        [
            missing_city,
            missing_coordinates,
        ],
        ignore_index=True,
    )

    review = (
        review
        .drop_duplicates(
            subset=["uid"]
        )
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    review.to_csv(
        REPORT_PATH,
        index=False,
        encoding="utf-8",
    )

    print("\n")
    print("=" * 75)

    print(
        "Rapport enregistré dans : "
        f"{REPORT_PATH}"
    )

    print(
        "AUDIT GEOGRAPHIQUE : TERMINE"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()