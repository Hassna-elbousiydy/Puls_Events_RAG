"""Compare deux snapshots CSV OpenAgenda.

Le script vérifie :
- le nombre de lignes et de colonnes ;
- les schémas ;
- les UID présents dans chaque fichier ;
- les doublons d'UID ;
- les différences de contenu pour les UID communs.

Aucun fichier n'est modifié.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


ORIGINAL_PATH = Path(
    "data/raw/evenements-publics-openagenda.csv"
)

API_PATH = Path(
    "data/raw/evenements-publics-openagenda-api.csv"
)


def load_csv(path: Path) -> pd.DataFrame:
    """Charge un CSV OpenAgenda en conservant les valeurs textuelles."""

    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}"
        )

    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        encoding_errors="replace",
        dtype=str,
        keep_default_na=False,
        low_memory=False,
    )


def normalize_value(value: object) -> str:
    """Normalise uniquement les fins de ligne pour la comparaison."""

    return (
        str(value)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )


def main() -> None:
    """Compare les deux snapshots OpenAgenda."""

    print("=" * 75)
    print("COMPARAISON DES SNAPSHOTS OPENAGENDA")
    print("=" * 75)

    original = load_csv(
        ORIGINAL_PATH
    )

    api = load_csv(
        API_PATH
    )

    print("\n--- DIMENSIONS ---")

    print(
        f"Original : {len(original):,} lignes, "
        f"{len(original.columns)} colonnes"
    )

    print(
        f"API      : {len(api):,} lignes, "
        f"{len(api.columns)} colonnes"
    )

    print("\n--- SCHEMA ---")

    same_columns = (
        list(original.columns)
        == list(api.columns)
    )

    print(
        f"Colonnes identiques et dans le même ordre : "
        f"{same_columns}"
    )

    original_only_columns = sorted(
        set(original.columns)
        - set(api.columns)
    )

    api_only_columns = sorted(
        set(api.columns)
        - set(original.columns)
    )

    print(
        f"Colonnes uniquement original : "
        f"{original_only_columns}"
    )

    print(
        f"Colonnes uniquement API      : "
        f"{api_only_columns}"
    )

    if "uid" not in original.columns:
        raise RuntimeError(
            "La colonne uid manque dans le fichier original."
        )

    if "uid" not in api.columns:
        raise RuntimeError(
            "La colonne uid manque dans le fichier API."
        )

    print("\n--- UID ---")

    original_duplicates = int(
        original["uid"].duplicated().sum()
    )

    api_duplicates = int(
        api["uid"].duplicated().sum()
    )

    print(
        f"Doublons UID original : {original_duplicates:,}"
    )

    print(
        f"Doublons UID API      : {api_duplicates:,}"
    )

    original_uids = set(
        original["uid"]
    )

    api_uids = set(
        api["uid"]
    )

    only_original = sorted(
        original_uids - api_uids
    )

    only_api = sorted(
        api_uids - original_uids
    )

    common_uids = (
        original_uids
        & api_uids
    )

    print(
        f"UID communs             : {len(common_uids):,}"
    )

    print(
        f"UID uniquement original : {len(only_original):,}"
    )

    print(
        f"UID uniquement API      : {len(only_api):,}"
    )

    if only_original:
        print(
            "\nExemples UID uniquement original :"
        )

        print(
            only_original[:20]
        )

    if only_api:
        print(
            "\nExemples UID uniquement API :"
        )

        print(
            only_api[:20]
        )

    # ---------------------------------------------------------------
    # Comparaison des valeurs des UID communs
    # ---------------------------------------------------------------

    common_columns = [
        column
        for column in original.columns
        if column in api.columns
    ]

    original_common = (
        original.loc[
            original["uid"].isin(common_uids),
            common_columns,
        ]
        .copy()
        .set_index("uid")
        .sort_index()
    )

    api_common = (
        api.loc[
            api["uid"].isin(common_uids),
            common_columns,
        ]
        .copy()
        .set_index("uid")
        .sort_index()
    )

    for column in original_common.columns:

        original_common[column] = (
            original_common[column]
            .map(normalize_value)
        )

        api_common[column] = (
            api_common[column]
            .map(normalize_value)
        )

    differences = (
        original_common
        .ne(api_common)
    )

    rows_with_difference = (
        differences
        .any(axis=1)
    )

    differing_row_count = int(
        rows_with_difference.sum()
    )

    print("\n--- CONTENU ---")

    print(
        "Lignes communes avec au moins une différence : "
        f"{differing_row_count:,}"
    )

    column_difference_counts = (
        differences
        .sum()
        .sort_values(
            ascending=False
        )
    )

    column_difference_counts = (
        column_difference_counts[
            column_difference_counts > 0
        ]
    )

    if column_difference_counts.empty:

        print(
            "Aucune différence de contenu détectée "
            "après alignement par UID."
        )

    else:

        print(
            "\nNombre de différences par colonne :"
        )

        print(
            column_difference_counts.to_string()
        )

        example_uids = (
            differences.index[
                rows_with_difference
            ]
            .tolist()[:10]
        )

        print(
            "\nExemples de lignes différentes :"
        )

        for uid in example_uids:

            print("\n" + "-" * 75)
            print(
                f"UID : {uid}"
            )

            changed_columns = (
                differences.columns[
                    differences.loc[uid]
                ]
                .tolist()
            )

            print(
                "Colonnes différentes : "
                f"{changed_columns}"
            )

            for column in changed_columns[:5]:

                old_value = (
                    original_common
                    .loc[uid, column]
                )

                new_value = (
                    api_common
                    .loc[uid, column]
                )

                print(
                    f"\n  {column}"
                )

                print(
                    "    Original : "
                    f"{old_value[:200]}"
                )

                print(
                    "    API      : "
                    f"{new_value[:200]}"
                )

    print("\n")
    print("=" * 75)

    if (
        not only_original
        and not only_api
        and differing_row_count == 0
    ):

        print(
            "CONCLUSION : LES DEUX DATASETS SONT "
            "SEMANTIQUEMENT IDENTIQUES"
        )

        print(
            "La différence SHA-256 provient uniquement "
            "de la sérialisation du fichier CSV."
        )

    elif (
        not only_original
        and not only_api
    ):

        print(
            "CONCLUSION : MEMES EVENEMENTS, "
            "MAIS CERTAINES VALEURS ONT CHANGE"
        )

    else:

        print(
            "CONCLUSION : LES DEUX SNAPSHOTS "
            "NE CONTIENNENT PAS EXACTEMENT "
            "LES MEMES EVENEMENTS"
        )

    print("=" * 75)


if __name__ == "__main__":
    main()