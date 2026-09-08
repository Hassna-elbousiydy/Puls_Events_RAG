"""Audit initial du fichier brut OpenAgenda.

Ce script inspecte les données avant tout nettoyage ou filtrage.
Il ne modifie jamais le fichier source.
"""

from pathlib import Path
import csv

import pandas as pd


DATA_PATH = Path("data/raw/evenements-publics-openagenda.csv")


def detect_separator(path: Path) -> str:
    """Détecte automatiquement le séparateur du fichier CSV."""
    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
        newline="",
    ) as file:
        sample = file.read(100_000)

    dialect = csv.Sniffer().sniff(
        sample,
        delimiters=",;\t|",
    )

    return dialect.delimiter


def main() -> None:
    """Affiche les principales caractéristiques du dataset OpenAgenda."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATA_PATH}"
        )

    separator = detect_separator(DATA_PATH)

    print("=" * 70)
    print("AUDIT INITIAL - DONNEES OPENAGENDA")
    print("=" * 70)

    print(f"\nFichier : {DATA_PATH}")
    print(
        f"Taille  : {DATA_PATH.stat().st_size / (1024 ** 2):.2f} Mo"
    )
    print(f"Séparateur détecté : {repr(separator)}")

    print("\nChargement du fichier...")

    df = pd.read_csv(
        DATA_PATH,
        sep=separator,
        encoding="utf-8-sig",
        encoding_errors="replace",
        low_memory=False,
    )

    print("\n--- DIMENSIONS ---")
    print(f"Nombre de lignes   : {len(df):,}")
    print(f"Nombre de colonnes : {len(df.columns):,}")

    print("\n--- COLONNES ---")
    for index, column in enumerate(df.columns, start=1):
        print(f"{index:02d}. {column}")

    print("\n--- TYPES DE DONNEES ---")
    print(df.dtypes.to_string())

    print("\n--- DOUBLONS COMPLETS ---")
    duplicate_count = int(df.duplicated().sum())
    print(f"Nombre de lignes dupliquées : {duplicate_count:,}")

    print("\n--- VALEURS MANQUANTES ---")

    missing = pd.DataFrame(
        {
            "missing_count": df.isna().sum(),
            "missing_percent": (
                df.isna().mean() * 100
            ).round(2),
        }
    )

    missing = missing.sort_values(
        "missing_percent",
        ascending=False,
    )

    print(missing.to_string())

    date_keywords = [
        "date",
        "begin",
        "end",
        "start",
        "time",
        "schedule",
    ]

    date_columns = [
        column
        for column in df.columns
        if any(
            keyword in column.lower()
            for keyword in date_keywords
        )
    ]

    print("\n--- COLONNES POTENTIELLEMENT TEMPORELLES ---")

    if date_columns:
        for column in date_columns:
            print(f"- {column}")
    else:
        print("Aucune colonne temporelle détectée automatiquement.")

    geo_keywords = [
        "city",
        "ville",
        "postal",
        "region",
        "department",
        "departement",
        "location",
        "address",
        "latitude",
        "longitude",
        "lat",
        "lon",
    ]

    geo_columns = [
        column
        for column in df.columns
        if any(
            keyword in column.lower()
            for keyword in geo_keywords
        )
    ]

    print("\n--- COLONNES POTENTIELLEMENT GEOGRAPHIQUES ---")

    if geo_columns:
        for column in geo_columns:
            print(f"- {column}")
    else:
        print("Aucune colonne géographique détectée automatiquement.")

    print("\n--- APERCU DES 3 PREMIERES LIGNES ---")

    pd.set_option(
        "display.max_columns",
        None,
    )

    print(df.head(3).to_string())

    print("\n" + "=" * 70)
    print("AUDIT OPENAGENDA : TERMINE")
    print("=" * 70)


if __name__ == "__main__":
    main()