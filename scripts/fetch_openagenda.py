"""Téléchargement reproductible des événements OpenAgenda pour Puls-Events.

Ce script récupère les événements depuis l'API Explore v2.1
du jeu public "Événements Publics - OpenAgenda".

Le téléchargement est filtré selon :
- la région Pays de la Loire ;
- un an d'historique par rapport à une date de référence ;
- les événements futurs, sans borne supérieure arbitraire.

Le résultat est enregistré sous forme de CSV brut.
Aucun nettoyage métier n'est réalisé ici : il appartient au script
``preprocess_openagenda.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


DATASET_ID = "evenements-publics-openagenda"

API_BASE = (
    "https://public.opendatasoft.com/"
    "api/explore/v2.1/catalog/datasets/"
    f"{DATASET_ID}"
)

RECORDS_ENDPOINT = (
    f"{API_BASE}/records"
)

EXPORT_ENDPOINT = (
    f"{API_BASE}/exports/csv"
)

DEFAULT_OUTPUT_PATH = Path(
    "data/raw/evenements-publics-openagenda.csv"
)

REPORT_PATH = Path(
    "reports/generated/acquisition_report.json"
)

TARGET_REGION = "Pays de la Loire"


def parse_args() -> argparse.Namespace:
    """Analyse les arguments de ligne de commande."""

    parser = argparse.ArgumentParser(
        description=(
            "Télécharge les événements OpenAgenda "
            "pour le projet Puls-Events."
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

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help=(
            "Chemin du CSV de sortie. "
            "Par défaut : "
            "data/raw/evenements-publics-openagenda.csv"
        ),
    )

    return parser.parse_args()


def get_reference_date(
    value: str | None,
) -> pd.Timestamp:
    """Retourne la date de référence en Europe/Paris."""

    if value:
        try:
            reference = pd.Timestamp(
                value,
                tz="Europe/Paris",
            )
        except ValueError as error:
            raise ValueError(
                "La date doit respecter le format YYYY-MM-DD."
            ) from error

    else:
        reference = pd.Timestamp.now(
            tz="Europe/Paris"
        ).normalize()

    return reference


def build_where_clause(
    cutoff_date: pd.Timestamp,
) -> str:
    """Construit le filtre ODSQL de l'API."""

    cutoff_string = (
        cutoff_date
        .strftime("%Y-%m-%d")
    )

    return (
        f'location_region = "{TARGET_REGION}" '
        f"and lastdate_end >= date'{cutoff_string}'"
    )


def request_json(
    url: str,
) -> dict:
    """Effectue une requête HTTP et retourne une réponse JSON."""

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Puls-Events-RAG/1.0"
            ),
        },
    )

    try:
        with urlopen(
            request,
            timeout=60,
        ) as response:
            content = response.read()

    except HTTPError as error:
        raise RuntimeError(
            f"Erreur HTTP {error.code} pour {url}"
        ) from error

    except URLError as error:
        raise RuntimeError(
            f"Impossible de contacter l'API : {error.reason}"
        ) from error

    return json.loads(
        content.decode("utf-8")
    )


def get_expected_count(
    where_clause: str,
) -> int:
    """Récupère le nombre d'enregistrements correspondant au filtre."""

    params = {
        "limit": 1,
        "where": where_clause,
        "timezone": "Europe/Paris",
    }

    url = (
        RECORDS_ENDPOINT
        + "?"
        + urlencode(params)
    )

    response = request_json(
        url
    )

    return int(
        response["total_count"]
    )


def download_csv(
    where_clause: str,
    output_path: Path,
) -> None:
    """Télécharge l'export CSV filtré dans un fichier temporaire."""

    params = {
        "where": where_clause,
        "timezone": "Europe/Paris",
        "delimiter": ";",
        "with_bom": "true",
        "use_labels": "false",
    }

    url = (
        EXPORT_ENDPOINT
        + "?"
        + urlencode(params)
    )

    request = Request(
        url,
        headers={
            "User-Agent": (
                "Puls-Events-RAG/1.0"
            ),
        },
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            dir=output_path.parent,
            suffix=".tmp",
        ) as temporary_file:

            temporary_path = Path(
                temporary_file.name
            )

            with urlopen(
                request,
                timeout=180,
            ) as response:

                shutil.copyfileobj(
                    response,
                    temporary_file,
                )

        if (
            temporary_path is None
            or not temporary_path.exists()
            or temporary_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "Le fichier téléchargé est vide."
            )

        temporary_path.replace(
            output_path
        )

    except HTTPError as error:
        raise RuntimeError(
            f"Erreur HTTP {error.code} pendant l'export CSV."
        ) from error

    except URLError as error:
        raise RuntimeError(
            "Impossible de télécharger l'export CSV : "
            f"{error.reason}"
        ) from error

    finally:
        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            temporary_path.unlink()


def validate_download(
    path: Path,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """Valide le CSV téléchargé avant utilisation."""

    if not path.exists():
        raise FileNotFoundError(
            f"CSV téléchargé introuvable : {path}"
        )

    df = pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        encoding_errors="replace",
        low_memory=False,
    )

    required_columns = {
        "uid",
        "title_fr",
        "description_fr",
        "timings",
        "lastdate_end",
        "location_region",
        "location_city",
        "location_coordinates",
        "status",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Colonnes attendues absentes de l'export : "
            f"{sorted(missing_columns)}"
        )

    regions = (
        df["location_region"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    invalid_regions = (
        regions.ne(TARGET_REGION)
        .sum()
    )

    if invalid_regions:
        raise RuntimeError(
            f"{invalid_regions} lignes sont hors de "
            f"la région {TARGET_REGION}."
        )

    last_end = pd.to_datetime(
        df["lastdate_end"],
        errors="coerce",
        utc=True,
    )

    cutoff_utc = (
        cutoff_date
        .tz_convert("UTC")
    )

    invalid_dates = (
        last_end.isna()
        | last_end.lt(cutoff_utc)
    )

    if invalid_dates.any():
        raise RuntimeError(
            f"{int(invalid_dates.sum())} lignes "
            "ne respectent pas le filtre temporel API."
        )

    return df


def save_report(
    output_path: Path,
    reference_date: pd.Timestamp,
    cutoff_date: pd.Timestamp,
    where_clause: str,
    expected_count: int,
    actual_count: int,
    file_size: int,
) -> None:
    """Enregistre les métadonnées de l'acquisition."""

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = {
        "dataset_id": DATASET_ID,
        "source": (
            "public.opendatasoft.com - "
            "Événements Publics OpenAgenda"
        ),
        "target_region": TARGET_REGION,
        "reference_date": (
            reference_date.isoformat()
        ),
        "history_cutoff": (
            cutoff_date.isoformat()
        ),
        "where_clause": where_clause,
        "api_count_before_export": int(
            expected_count
        ),
        "downloaded_rows": int(
            actual_count
        ),
        "output_file": str(
            output_path
        ),
        "output_size_bytes": int(
            file_size
        ),
        "downloaded_at_utc": (
            datetime.now(timezone.utc)
            .isoformat()
        ),
    }

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


def main() -> None:
    """Télécharge et valide les données OpenAgenda."""

    args = parse_args()

    output_path = Path(
        args.output
    )

    reference_date = get_reference_date(
        args.reference_date
    )

    cutoff_date = (
        reference_date
        - pd.DateOffset(years=1)
    )

    where_clause = build_where_clause(
        cutoff_date
    )

    print("=" * 75)
    print("ACQUISITION OPENAGENDA - PULS-EVENTS")
    print("=" * 75)

    print(
        f"\nRégion          : {TARGET_REGION}"
    )

    print(
        f"Date référence  : {reference_date.date()}"
    )

    print(
        f"Début historique: {cutoff_date.date()}"
    )

    print(
        f"Filtre API      : {where_clause}"
    )

    print("\nComptage des événements via l'API...")

    expected_count = get_expected_count(
        where_clause
    )

    print(
        f"Événements annoncés par l'API : "
        f"{expected_count:,}"
    )

    print("\nTéléchargement de l'export CSV...")

    download_csv(
        where_clause,
        output_path,
    )

    print("\nValidation du fichier téléchargé...")

    df = validate_download(
        output_path,
        cutoff_date,
    )

    actual_count = len(df)

    file_size = (
        output_path.stat().st_size
    )

    save_report(
        output_path=output_path,
        reference_date=reference_date,
        cutoff_date=cutoff_date,
        where_clause=where_clause,
        expected_count=expected_count,
        actual_count=actual_count,
        file_size=file_size,
    )

    print("\n--- RESULTATS ---")

    print(
        f"Lignes téléchargées : {actual_count:,}"
    )

    print(
        f"Colonnes            : {len(df.columns):,}"
    )

    print(
        f"Taille du fichier   : "
        f"{file_size / (1024 ** 2):.2f} Mo"
    )

    if actual_count != expected_count:
        print(
            "\nATTENTION : le nombre d'enregistrements "
            "a changé entre le comptage et l'export."
        )

        print(
            "Cela peut arriver lorsque le jeu OpenAgenda "
            "est mis à jour pendant le téléchargement."
        )

    print("\nFichier téléchargé :")
    print(
        output_path
    )

    print("\nRapport d'acquisition :")
    print(
        REPORT_PATH
    )

    print("\nACQUISITION OPENAGENDA : OK")


if __name__ == "__main__":
    main()