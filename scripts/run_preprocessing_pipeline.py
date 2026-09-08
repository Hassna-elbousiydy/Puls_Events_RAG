"""Pipeline reproductible de préparation des données Puls-Events.

Le pipeline exécute successivement :
1. l'acquisition des données OpenAgenda via l'API ;
2. le pré-processing des événements ;
3. les tests unitaires de qualité.

La même date de référence est utilisée pendant toutes les étapes.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo


RAW_OUTPUT = (
    "data/raw/evenements-publics-openagenda.csv"
)


def parse_args() -> argparse.Namespace:
    """Analyse les arguments du pipeline."""

    parser = argparse.ArgumentParser(
        description=(
            "Exécute l'acquisition, le pré-processing "
            "et les tests de qualité Puls-Events."
        )
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


def run_command(
    command: list[str],
    env: dict[str, str] | None = None,
) -> None:
    """Exécute une commande et arrête le pipeline en cas d'échec."""

    print(
        "\n" + "=" * 75,
        flush=True,
    )

    print(
        "EXECUTION : "
        + " ".join(command),
        flush=True,
    )

    print(
        "=" * 75,
        flush=True,
    )

    subprocess.run(
        command,
        check=True,
        env=env,
    )


def get_reference_date(
    requested_date: str | None,
) -> str:
    """Retourne la date fournie ou la date actuelle en Europe/Paris."""

    if requested_date:

        try:
            datetime.strptime(
                requested_date,
                "%Y-%m-%d",
            )

        except ValueError as error:
            raise ValueError(
                "La date doit respecter le format YYYY-MM-DD."
            ) from error

        return requested_date

    return datetime.now(
        ZoneInfo("Europe/Paris")
    ).date().isoformat()


def main() -> None:
    """Exécute le pipeline Data complet de Puls-Events."""

    args = parse_args()

    reference_date = get_reference_date(
        args.reference_date
    )

    print(
        "=" * 75,
        flush=True,
    )

    print(
        "PIPELINE DATA - PULS-EVENTS",
        flush=True,
    )

    print(
        "=" * 75,
        flush=True,
    )

    print(
        f"\nDate de référence : {reference_date}",
        flush=True,
    )

    pipeline_env = os.environ.copy()

    pipeline_env[
        "PULS_REFERENCE_DATE"
    ] = reference_date

    # ---------------------------------------------------------------
    # Étape 1 : acquisition OpenAgenda
    # ---------------------------------------------------------------

    run_command(
        [
            sys.executable,
            "scripts/fetch_openagenda.py",
            "--reference-date",
            reference_date,
            "--output",
            RAW_OUTPUT,
        ],
        env=pipeline_env,
    )

    # ---------------------------------------------------------------
    # Étape 2 : pré-processing
    # ---------------------------------------------------------------

    run_command(
        [
            sys.executable,
            "scripts/preprocess_openagenda.py",
            "--reference-date",
            reference_date,
        ],
        env=pipeline_env,
    )

    # ---------------------------------------------------------------
    # Étape 3 : tests unitaires
    # ---------------------------------------------------------------

    run_command(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_preprocessing.py",
        ],
        env=pipeline_env,
    )

    # ---------------------------------------------------------------
    # Succès
    # ---------------------------------------------------------------

    print(
        "\n" + "=" * 75,
        flush=True,
    )

    print(
        "PIPELINE DATA : SUCCES",
        flush=True,
    )

    print(
        "=" * 75,
        flush=True,
    )

    print(
        "\nAcquisition OpenAgenda : OK",
        flush=True,
    )

    print(
        "Pré-processing         : OK",
        flush=True,
    )

    print(
        "Tests de qualité       : OK",
        flush=True,
    )

    print(
        "\nLe dataset est prêt pour "
        "l'étape de vectorisation.",
        flush=True,
    )


if __name__ == "__main__":
    main()