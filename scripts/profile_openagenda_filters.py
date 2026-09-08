"""Profil des filtres géographiques et temporels OpenAgenda.

Ce script analyse le périmètre réel du fichier brut avant de choisir
les règles définitives de filtrage du pipeline.
"""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/evenements-publics-openagenda.csv")


def print_value_counts(
    df: pd.DataFrame,
    column: str,
    top_n: int = 30,
) -> None:
    """Affiche les valeurs les plus fréquentes d'une colonne."""

    print(f"\n--- {column.upper()} ---")

    values = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    print(f"Valeurs renseignées : {len(values):,}")
    print(f"Valeurs uniques     : {values.nunique():,}")

    print("\nValeurs les plus fréquentes :")
    print(values.value_counts().head(top_n).to_string())


def main() -> None:
    """Analyse la répartition géographique et temporelle du dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATA_PATH}"
        )

    print("=" * 70)
    print("PROFIL DES FILTRES OPENAGENDA")
    print("=" * 70)

    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        encoding="utf-8-sig",
        encoding_errors="replace",
        low_memory=False,
    )

    print(f"\nNombre total d'événements : {len(df):,}")

    # ------------------------------------------------------------------
    # Géographie
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("1. PROFIL GEOGRAPHIQUE")
    print("=" * 70)

    country = (
        df["location_countrycode"]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    print("\n--- PAYS ---")
    print(country.value_counts().to_string())

    print_value_counts(
        df,
        "location_region",
        top_n=30,
    )

    print_value_counts(
        df,
        "location_department",
        top_n=30,
    )

    print_value_counts(
        df,
        "location_city",
        top_n=30,
    )

    # ------------------------------------------------------------------
    # Dates
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("2. PROFIL TEMPOREL")
    print("=" * 70)

    date_columns = [
        "firstdate_begin",
        "firstdate_end",
        "lastdate_begin",
        "lastdate_end",
        "updatedat",
    ]

    for column in date_columns:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
            utc=True,
        )

    print("\n--- PLAGE DES DATES ---")

    print(
        "Premier début :",
        df["firstdate_begin"].min(),
    )

    print(
        "Dernier début :",
        df["lastdate_begin"].max(),
    )

    print(
        "Première fin  :",
        df["firstdate_end"].min(),
    )

    print(
        "Dernière fin  :",
        df["lastdate_end"].max(),
    )

    # Date actuelle utilisée pour le contrôle.
    today = (
        pd.Timestamp.now(tz="Europe/Paris")
        .normalize()
        .tz_convert("UTC")
    )

    one_year_ago = today - pd.DateOffset(years=1)

    print("\n--- DATE DE REFERENCE ---")
    print(f"Aujourd'hui      : {today}")
    print(f"Il y a un an     : {one_year_ago}")

    first_begin = df["firstdate_begin"]
    last_end = df["lastdate_end"]

    older_than_one_year = last_end < one_year_ago

    recent_past = (
        (last_end >= one_year_ago)
        & (last_end < today)
    )

    spans_reference_date = (
    (first_begin <= today)
    & (last_end >= today)
    )

    future = first_begin > today

    print("\n--- REPARTITION TEMPORELLE ---")

    print(
        "Terminés depuis plus d'un an : "
        f"{int(older_than_one_year.sum()):,}"
    )

    print(
        "Historique de moins d'un an  : "
        f"{int(recent_past.sum()):,}"
    )

    print(
    "Plage globale couvrant la référence : "
    f"{int(spans_reference_date.sum()):,}"
    )

    print(
        "Événements à venir           : "
        f"{int(future.sum()):,}"
    )

    print("\n--- DATES MANQUANTES ---")

    for column in [
        "firstdate_begin",
        "lastdate_end",
    ]:
        print(
            f"{column}: "
            f"{int(df[column].isna().sum()):,}"
        )

    # ------------------------------------------------------------------
    # Dates futures extrêmes
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("3. EVENEMENTS LES PLUS ELOIGNES DANS LE FUTUR")
    print("=" * 70)

    columns_to_display = [
        "uid",
        "title_fr",
        "location_city",
        "location_department",
        "location_region",
        "firstdate_begin",
        "lastdate_end",
    ]

    future_extreme = (
        df[columns_to_display]
        .sort_values(
            "lastdate_end",
            ascending=False,
        )
        .head(10)
    )

    print(
        "\n"
        + future_extreme.to_string(
            index=False,
        )
    )

    # ------------------------------------------------------------------
    # Statuts
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("4. STATUTS")
    print("=" * 70)

    print(
        "\n"
        + df["status"]
        .value_counts(dropna=False)
        .head(20)
        .to_string()
    )

    print("\n")
    print("=" * 70)
    print("PROFIL DES FILTRES : TERMINE")
    print("=" * 70)


if __name__ == "__main__":
    main()