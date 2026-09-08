"""Audit de la pertinence culturelle des données OpenAgenda.

Le script étudie les agendas sources et les mots-clés avant de définir
une règle de sélection des événements culturels.
"""

from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/raw/evenements-publics-openagenda.csv")


def main() -> None:
    """Analyse les sources et mots-clés du corpus OpenAgenda."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Fichier introuvable : {DATA_PATH}")

    df = pd.read_csv(
        DATA_PATH,
        sep=";",
        encoding="utf-8-sig",
        encoding_errors="replace",
        low_memory=False,
    )

    print("=" * 75)
    print("AUDIT DU PERIMETRE CULTUREL OPENAGENDA")
    print("=" * 75)

    print(f"\nNombre total d'événements : {len(df):,}")

    # ---------------------------------------------------------------
    # Agendas sources
    # ---------------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("1. AGENDAS SOURCES")
    print("=" * 75)

    agendas = (
        df["originagenda_title"]
        .fillna("[MANQUANT]")
        .astype(str)
        .str.strip()
    )

    print(f"\nNombre d'agendas différents : {agendas.nunique():,}")

    print("\n50 agendas les plus fréquents :\n")

    print(
        agendas
        .value_counts()
        .head(50)
        .to_string()
    )

    # ---------------------------------------------------------------
    # Mots-clés
    # ---------------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("2. COUVERTURE DES MOTS-CLES")
    print("=" * 75)

    keyword_available = df["keywords_fr"].notna()

    print(
        "\nÉvénements avec keywords_fr : "
        f"{int(keyword_available.sum()):,} "
        f"({keyword_available.mean() * 100:.2f} %)"
    )

    print(
        "Événements sans keywords_fr  : "
        f"{int((~keyword_available).sum()):,}"
    )

    # Transforme les listes de mots-clés en valeurs individuelles
    keywords = (
        df["keywords_fr"]
        .dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
        .str.lower()
    )

    keywords = keywords[keywords != ""]

    print(
        f"\nNombre de mots-clés distincts : "
        f"{keywords.nunique():,}"
    )

    print("\n80 mots-clés les plus fréquents :\n")

    print(
        keywords
        .value_counts()
        .head(80)
        .to_string()
    )

    # ---------------------------------------------------------------
    # Agendas + exemples
    # ---------------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("3. EXEMPLES PARMI LES PRINCIPAUX AGENDAS")
    print("=" * 75)

    top_agendas = (
        agendas
        .value_counts()
        .head(20)
        .index
        .tolist()
    )

    for agenda in top_agendas:
        if agenda == "[MANQUANT]":
            subset = df[df["originagenda_title"].isna()]
        else:
            subset = df[
                df["originagenda_title"]
                .fillna("")
                .astype(str)
                .str.strip()
                .eq(agenda)
            ]

        titles = (
            subset["title_fr"]
            .dropna()
            .astype(str)
            .head(3)
            .tolist()
        )

        print("\n" + "-" * 75)
        print(f"AGENDA : {agenda}")
        print(f"Nombre : {len(subset):,}")

        for title in titles:
            print(f"  - {title}")

    # ---------------------------------------------------------------
    # Champs textuels utiles au futur RAG
    # ---------------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("4. CHAMPS TEXTUELS")
    print("=" * 75)

    fields = [
        "title_fr",
        "description_fr",
        "longdescription_fr",
        "keywords_fr",
    ]

    for field in fields:
        available = df[field].notna().sum()

        print(
            f"{field:25s} : "
            f"{available:6,d} renseignés "
            f"({available / len(df) * 100:6.2f} %)"
        )

    print("\n")
    print("=" * 75)
    print("AUDIT CULTUREL : TERMINE")
    print("=" * 75)


if __name__ == "__main__":
    main()