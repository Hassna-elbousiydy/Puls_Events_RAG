"""Création des chunks d'événements pour Puls-Events.

Ce script :
- charge le dataset OpenAgenda pré-processé ;
- découpe le champ ``text_for_embedding`` en chunks ;
- conserve les métadonnées de chaque événement ;
- génère un identifiant unique pour chaque chunk ;
- écrit les chunks au format JSONL ;
- produit un rapport de contrôle.

Aucun appel à l'API Mistral n'est réalisé ici.
Aucun embedding et aucun index FAISS ne sont encore créés.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ---------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------

DATA_PATH = Path(
    "data/processed/events_processed.csv"
)

OUTPUT_PATH = Path(
    "data/processed/events_chunks.jsonl"
)

REPORT_PATH = Path(
    "reports/generated/chunking_report.json"
)


# ---------------------------------------------------------------------
# Paramètres de chunking
# ---------------------------------------------------------------------

CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


# ---------------------------------------------------------------------
# Métadonnées conservées
# ---------------------------------------------------------------------

METADATA_COLUMNS = [
    "title",
    "eligible_first_begin",
    "eligible_last_end",
    "eligible_timings_json",
    "date_range",
    "location_name",
    "location_text",
    "city",
    "department",
    "region",
    "country_code",
    "latitude",
    "longitude",
    "attendance_mode",
    "status_clean",
    "registration",
    "canonical_url",
    "source_agenda",
]


def clean_metadata_value(
    value: object,
) -> object:
    """Convertit une valeur Pandas en valeur compatible JSON."""

    if pd.isna(value):
        return None

    if isinstance(
        value,
        (
            int,
            float,
            bool,
        ),
    ):
        return value

    return str(
        value
    )


def create_splitter() -> RecursiveCharacterTextSplitter:
    """Crée le splitter LangChain utilisé pour Puls-Events.

    Le retour à la ligne simple n'est volontairement pas utilisé comme
    séparateur prioritaire afin d'éviter de créer de très petits chunks
    à partir des lignes Titre / Description / Détails / Localisation.

    Le découpage privilégie les limites de paragraphes et de phrases.
    """

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=[
            "\n\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
            "",
        ],
    )


def main() -> None:
    """Découpe tous les événements et génère les fichiers de sortie."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {DATA_PATH}\n"
            "Exécutez d'abord le pipeline de pré-processing."
        )

    df = pd.read_csv(
        DATA_PATH,
        low_memory=False,
    )

    required_columns = {
        "uid",
        "text_for_embedding",
        *METADATA_COLUMNS,
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise RuntimeError(
            "Colonnes nécessaires manquantes : "
            f"{sorted(missing_columns)}"
        )

    if df["uid"].duplicated().any():
        raise RuntimeError(
            "Le dataset source contient des UID dupliqués."
        )

    splitter = create_splitter()

    output_records: list[
        dict[str, object]
    ] = []

    chunks_per_event: list[int] = []

    events_without_chunks: list[str] = []

    print("=" * 75)
    print("CREATION DES CHUNKS - PULS-EVENTS")
    print("=" * 75)

    print(
        f"\nÉvénements sources : {len(df):,}"
    )

    print(
        f"Chunk size         : {CHUNK_SIZE}"
    )

    print(
        f"Chunk overlap      : {CHUNK_OVERLAP}"
    )

    # -----------------------------------------------------------------
    # Chunking événement par événement
    # -----------------------------------------------------------------

    for row in df.itertuples(
        index=False,
    ):

        uid = str(
            row.uid
        )

        source_text = (
            str(
                row.text_for_embedding
            )
            if not pd.isna(
                row.text_for_embedding
            )
            else ""
        ).strip()

        if not source_text:
            events_without_chunks.append(
                uid
            )
            continue

        chunks = splitter.split_text(
            source_text
        )

        chunks = [
            chunk.strip()
            for chunk in chunks
            if chunk.strip()
        ]

        if not chunks:
            events_without_chunks.append(
                uid
            )
            continue

        chunk_count = len(
            chunks
        )

        chunks_per_event.append(
            chunk_count
        )

        metadata = {}

        for column in METADATA_COLUMNS:

            metadata[
                column
            ] = clean_metadata_value(
                getattr(
                    row,
                    column,
                )
            )

        for chunk_index, chunk_text in enumerate(
            chunks
        ):

            chunk_id = (
                f"{uid}_"
                f"{chunk_index:03d}"
            )

            record = {
                "chunk_id": chunk_id,
                "uid": uid,
                "chunk_index": int(
                    chunk_index
                ),
                "chunk_count": int(
                    chunk_count
                ),
                "content": chunk_text,
                **metadata,
            }

            output_records.append(
                record
            )

    # -----------------------------------------------------------------
    # Contrôles
    # -----------------------------------------------------------------

    if events_without_chunks:
        raise RuntimeError(
            "Certains événements n'ont produit aucun chunk : "
            f"{events_without_chunks[:20]}"
        )

    chunks_df = pd.DataFrame(
        output_records
    )

    if chunks_df.empty:
        raise RuntimeError(
            "Aucun chunk n'a été produit."
        )

    if chunks_df["chunk_id"].duplicated().any():
        raise RuntimeError(
            "Des identifiants de chunks sont dupliqués."
        )

    source_uid_count = int(
        df["uid"]
        .astype(str)
        .nunique()
    )

    chunk_uid_count = int(
        chunks_df["uid"]
        .nunique()
    )

    if source_uid_count != chunk_uid_count:
        raise RuntimeError(
            "Tous les événements sources ne sont pas "
            "représentés dans les chunks."
        )

    chunks_df[
        "chunk_length_chars"
    ] = (
        chunks_df["content"]
        .str.len()
    )

    if (
        chunks_df["chunk_length_chars"]
        > CHUNK_SIZE
    ).any():
        raise RuntimeError(
            "Au moins un chunk dépasse CHUNK_SIZE."
        )

    # -----------------------------------------------------------------
    # Export JSONL
    # -----------------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in output_records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
            )

            file.write(
                "\n"
            )

    # -----------------------------------------------------------------
    # Statistiques
    # -----------------------------------------------------------------

    chunks_per_event_series = pd.Series(
        chunks_per_event,
        dtype="int64",
    )

    single_chunk_events = int(
        (
            chunks_per_event_series
            == 1
        ).sum()
    )

    multi_chunk_events = int(
        (
            chunks_per_event_series
            > 1
        ).sum()
    )

    chunks_under_100 = int(
        (
            chunks_df["chunk_length_chars"]
            < 100
        ).sum()
    )

    chunks_under_200 = int(
        (
            chunks_df["chunk_length_chars"]
            < 200
        ).sum()
    )

    chunks_under_300 = int(
        (
            chunks_df["chunk_length_chars"]
            < 300
        ).sum()
    )

    report = {
        "source_file": str(
            DATA_PATH
        ),
        "output_file": str(
            OUTPUT_PATH
        ),
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "source_events": int(
            len(df)
        ),
        "source_unique_uids": (
            source_uid_count
        ),
        "events_represented_in_chunks": (
            chunk_uid_count
        ),
        "events_without_chunks": int(
            len(events_without_chunks)
        ),
        "total_chunks": int(
            len(chunks_df)
        ),
        "single_chunk_events": (
            single_chunk_events
        ),
        "multi_chunk_events": (
            multi_chunk_events
        ),
        "average_chunks_per_event": round(
            float(
                chunks_per_event_series.mean()
            ),
            3,
        ),
        "min_chunks_per_event": int(
            chunks_per_event_series.min()
        ),
        "max_chunks_per_event": int(
            chunks_per_event_series.max()
        ),
        "min_chunk_chars": int(
            chunks_df[
                "chunk_length_chars"
            ].min()
        ),
        "mean_chunk_chars": round(
            float(
                chunks_df[
                    "chunk_length_chars"
                ].mean()
            ),
            2,
        ),
        "max_chunk_chars": int(
            chunks_df[
                "chunk_length_chars"
            ].max()
        ),
        "chunks_under_100_chars": (
            chunks_under_100
        ),
        "chunks_under_200_chars": (
            chunks_under_200
        ),
        "chunks_under_300_chars": (
            chunks_under_300
        ),
        "unique_chunk_ids": int(
            chunks_df[
                "chunk_id"
            ].nunique()
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

    # -----------------------------------------------------------------
    # Résumé
    # -----------------------------------------------------------------

    print("\n--- RESULTATS ---")

    for key, value in report.items():

        print(
            f"{key}: {value}"
        )

    print(
        "\nFichier des chunks :"
    )

    print(
        OUTPUT_PATH
    )

    print(
        "\nRapport :"
    )

    print(
        REPORT_PATH
    )

    print(
        "\nCREATION DES CHUNKS : TERMINEE"
    )


if __name__ == "__main__":
    main()