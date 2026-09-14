"""Analyse des longueurs de textes avant le chunking Puls-Events.

Ce script analyse le champ ``text_for_embedding`` du dataset nettoyé
afin de choisir une stratégie de découpage adaptée aux données réelles.

Aucune donnée n'est modifiée et aucun appel à l'API Mistral n'est effectué.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


DATA_PATH = Path(
    "data/processed/events_processed.csv"
)

REPORT_PATH = Path(
    "reports/generated/chunking_profile.json"
)


def main() -> None:
    """Analyse les longueurs des textes préparés pour l'indexation."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}\n"
            "Exécutez d'abord le pipeline de pré-processing."
        )

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    if "text_for_embedding" not in df.columns:
        raise RuntimeError(
            "La colonne text_for_embedding est absente."
        )

    texts = (
        df["text_for_embedding"]
        .fillna("")
        .astype(str)
    )

    df["text_length_chars"] = (
        texts.str.len()
    )

    df["text_length_words"] = (
        texts
        .str.split()
        .str.len()
    )

    total_events = len(df)

    empty_texts = int(
        texts.str.strip().eq("").sum()
    )

    quantiles_chars = (
        df["text_length_chars"]
        .quantile(
            [
                0.00,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                1.00,
            ]
        )
    )

    quantiles_words = (
        df["text_length_words"]
        .quantile(
            [
                0.00,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                1.00,
            ]
        )
    )

    thresholds = [
        500,
        1000,
        1500,
        2000,
        3000,
        5000,
    ]

    above_thresholds: dict[str, dict[str, float | int]] = {}

    for threshold in thresholds:

        count = int(
            (
                df["text_length_chars"]
                > threshold
            ).sum()
        )

        percentage = (
            count
            / total_events
            * 100
            if total_events
            else 0
        )

        above_thresholds[
            str(threshold)
        ] = {
            "count": count,
            "percentage": round(
                percentage,
                2,
            ),
        }

    longest = (
        df[
            [
                "uid",
                "title",
                "text_length_chars",
                "text_length_words",
            ]
        ]
        .sort_values(
            "text_length_chars",
            ascending=False,
        )
        .head(20)
    )

    report = {
        "dataset": str(
            DATA_PATH
        ),
        "total_events": int(
            total_events
        ),
        "empty_texts": empty_texts,
        "character_lengths": {
            "min": int(
                df["text_length_chars"].min()
            ),
            "mean": round(
                float(
                    df["text_length_chars"].mean()
                ),
                2,
            ),
            "median": round(
                float(
                    df["text_length_chars"].median()
                ),
                2,
            ),
            "max": int(
                df["text_length_chars"].max()
            ),
            "p25": round(
                float(
                    quantiles_chars.loc[0.25]
                ),
                2,
            ),
            "p75": round(
                float(
                    quantiles_chars.loc[0.75]
                ),
                2,
            ),
            "p90": round(
                float(
                    quantiles_chars.loc[0.90]
                ),
                2,
            ),
            "p95": round(
                float(
                    quantiles_chars.loc[0.95]
                ),
                2,
            ),
            "p99": round(
                float(
                    quantiles_chars.loc[0.99]
                ),
                2,
            ),
        },
        "word_lengths": {
            "min": int(
                df["text_length_words"].min()
            ),
            "mean": round(
                float(
                    df["text_length_words"].mean()
                ),
                2,
            ),
            "median": round(
                float(
                    df["text_length_words"].median()
                ),
                2,
            ),
            "max": int(
                df["text_length_words"].max()
            ),
            "p90": round(
                float(
                    quantiles_words.loc[0.90]
                ),
                2,
            ),
            "p95": round(
                float(
                    quantiles_words.loc[0.95]
                ),
                2,
            ),
            "p99": round(
                float(
                    quantiles_words.loc[0.99]
                ),
                2,
            ),
        },
        "events_above_character_thresholds": (
            above_thresholds
        ),
    }

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("=" * 75)
    print("ANALYSE DU CHUNKING - PULS-EVENTS")
    print("=" * 75)

    print(
        f"\nÉvénements analysés : {total_events:,}"
    )

    print(
        f"Textes vides         : {empty_texts:,}"
    )

    print("\n--- LONGUEUR EN CARACTERES ---")

    for key, value in report[
        "character_lengths"
    ].items():

        print(
            f"{key:10s}: {value}"
        )

    print("\n--- LONGUEUR EN MOTS ---")

    for key, value in report[
        "word_lengths"
    ].items():

        print(
            f"{key:10s}: {value}"
        )

    print(
        "\n--- EVENEMENTS DEPASSANT "
        "LES SEUILS ---"
    )

    for threshold, values in (
        above_thresholds.items()
    ):

        print(
            f"> {threshold:5s} caractères : "
            f"{values['count']:5,d} "
            f"({values['percentage']:6.2f} %)"
        )

    print(
        "\n--- 20 TEXTES LES PLUS LONGS ---\n"
    )

    print(
        longest.to_string(
            index=False,
        )
    )

    print(
        "\nRapport enregistré dans : "
        f"{REPORT_PATH}"
    )

    print(
        "\nANALYSE DU CHUNKING : TERMINEE"
    )


if __name__ == "__main__":
    main()