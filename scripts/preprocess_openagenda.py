"""Pré-processing des événements OpenAgenda pour Puls-Events.

Ce script :
- charge les données OpenAgenda brutes ;
- normalise les données utiles ;
- filtre le périmètre géographique ;
- applique la contrainte temporelle de la mission sur les créneaux réels ;
- supprime les événements annulés ;
- sélectionne un sous-ensemble d'événements culturels ;
- nettoie les champs textuels ;
- conserve les métadonnées nécessaires au futur système RAG ;
- construit une localisation textuelle exploitable même si la ville manque ;
- prépare un texte exploitable pour la future vectorisation ;
- produit un fichier CSV nettoyé ;
- produit un rapport JSON de contrôle.

Le fichier source brut n'est jamais modifié.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------

RAW_PATH = Path(
    "data/raw/evenements-publics-openagenda.csv"
)

OUTPUT_PATH = Path(
    "data/processed/events_processed.csv"
)

REPORT_PATH = Path(
    "reports/generated/preprocessing_report.json"
)


# ---------------------------------------------------------------------
# Périmètre géographique
# ---------------------------------------------------------------------

TARGET_REGION = "Pays de la Loire"


# ---------------------------------------------------------------------
# Périmètre culturel
# ---------------------------------------------------------------------

CULTURAL_TERMS = [
    "art",
    "arts",
    "artiste",
    "artistes",
    "artistique",
    "architecture",
    "bibliothèque",
    "cinéma",
    "concert",
    "conservatoire",
    "culture",
    "culturel",
    "danse",
    "exposition",
    "festival",
    "film",
    "galerie",
    "jazz",
    "littérature",
    "livre",
    "lecture",
    "médiathèque",
    "musée",
    "musique",
    "musical",
    "opéra",
    "orchestre",
    "patrimoine",
    "photographie",
    "poésie",
    "spectacle",
    "théâtre",
    "cirque",
    "conte",
    "chanson",
    "comédie",
    "humour",
    "vernissage",
    "arts plastiques",
    "bande dessinée",
    "visite guidée",
    "visite commentée",
    "improvisation",
    "artisanat",
]


CULTURAL_AGENDA_TERMS = [
    "journées européennes du patrimoine",
    "bibliothèque",
    "musée",
    "muséum",
    "conservatoire",
    "festival culture bar bars",
    "scare",
    "stereolux",
    "biblis en folie",
    "rendez-vous aux jardins",
    "agenda patrimoine",
    "agenda régional de l'architecture",
    "la bouche d'air",
    "cosmopolis",
]


NON_CULTURAL_AGENDA_TERMS = [
    "mes événements france travail",
    "semaine de l'industrie",
    "chambre d'agriculture",
    "challenges geovelo",
    "semaine des métiers du tourisme",
    "journées nationales de l'agriculture",
]


# ---------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Analyse les arguments de ligne de commande."""

    parser = argparse.ArgumentParser(
        description=(
            "Pré-processing des événements OpenAgenda "
            "pour le POC Puls-Events."
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


# ---------------------------------------------------------------------
# Nettoyage des textes
# ---------------------------------------------------------------------

def clean_text(value: object) -> str:
    """Nettoie un champ textuel.

    Les balises HTML sont supprimées, les entités HTML sont décodées
    et les espaces successifs sont normalisés.
    """

    if pd.isna(value):
        return ""

    text = str(value)

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    text = html.unescape(text)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def normalize_for_matching(value: object) -> str:
    """Normalise un texte pour les comparaisons lexicales."""

    text = clean_text(value).casefold()

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    text = re.sub(
        r"[^a-z0-9]+",
        " ",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def contains_term(
    text: object,
    term: str,
) -> bool:
    """Teste la présence d'un mot ou d'une expression entière.

    Cette fonction évite notamment que le terme "art"
    corresponde accidentellement au mot "quartier".
    """

    normalized_text = normalize_for_matching(text)
    normalized_term = normalize_for_matching(term)

    if not normalized_text or not normalized_term:
        return False

    pattern = (
        rf"(?<![a-z0-9])"
        rf"{re.escape(normalized_term)}"
        rf"(?![a-z0-9])"
    )

    return bool(
        re.search(
            pattern,
            normalized_text,
        )
    )


# ---------------------------------------------------------------------
# Champs JSON OpenAgenda
# ---------------------------------------------------------------------

def extract_json_label(value: object) -> str:
    """Extrait le label français d'un champ JSON OpenAgenda."""

    if pd.isna(value):
        return ""

    try:
        data = json.loads(
            str(value)
        )

        label = data.get(
            "label",
            {},
        )

        if isinstance(label, dict):
            return str(
                label.get("fr")
                or label.get("en")
                or ""
            ).strip()

    except (
        json.JSONDecodeError,
        TypeError,
        AttributeError,
    ):
        pass

    return clean_text(value)


# ---------------------------------------------------------------------
# Normalisation géographique
# ---------------------------------------------------------------------

def normalize_department(value: object) -> str:
    """Normalise les variantes observées des départements."""

    text = clean_text(value)

    normalized = normalize_for_matching(
        text
    )

    mapping = {
        "loire atlantique": "Loire-Atlantique",
        "maine et loire": "Maine-et-Loire",
        "vendee": "Vendée",
        "sarthe": "Sarthe",
        "mayenne": "Mayenne",
    }

    return mapping.get(
        normalized,
        text,
    )


def parse_coordinates(
    value: object,
) -> tuple[float | None, float | None]:
    """Sépare les coordonnées OpenAgenda en latitude et longitude."""

    if pd.isna(value):
        return None, None

    parts = str(value).split(",")

    if len(parts) != 2:
        return None, None

    try:
        latitude = float(
            parts[0].strip()
        )

        longitude = float(
            parts[1].strip()
        )

        return latitude, longitude

    except ValueError:
        return None, None


def build_location_text(
    row: pd.Series,
) -> str:
    """Construit une localisation textuelle à partir des données disponibles.

    Aucun lieu n'est déduit ou inventé. La fonction concatène uniquement
    les informations géographiques présentes dans les données OpenAgenda.
    """

    components: list[str] = []

    fields = [
        "location_name",
        "address",
        "postal_code",
        "city",
        "department",
        "region",
    ]

    for field in fields:

        value = row.get(
            field,
            "",
        )

        if pd.isna(value):
            continue

        text = str(
            value
        ).strip()

        if (
            text
            and text.lower() != "nan"
        ):
            components.append(
                text
            )

    unique_components = list(
        dict.fromkeys(
            components
        )
    )

    return ", ".join(
        unique_components
    )


# ---------------------------------------------------------------------
# Créneaux temporels
# ---------------------------------------------------------------------

def parse_and_filter_timings(
    value: object,
    cutoff_date: pd.Timestamp,
) -> list[dict[str, str]]:
    """Parse les créneaux OpenAgenda et conserve les créneaux admissibles.

    Un créneau est conservé lorsque sa date de fin est postérieure
    ou égale à la borne correspondant à un an avant la date de référence.

    Le contrôle porte sur les créneaux réels contenus dans ``timings``
    et non uniquement sur firstdate_begin / lastdate_end.
    """

    if pd.isna(value):
        return []

    try:
        timings = json.loads(
            str(value)
        )

    except (
        json.JSONDecodeError,
        TypeError,
    ):
        return []

    if not isinstance(
        timings,
        list,
    ):
        return []

    eligible_timings: list[
        dict[str, str]
    ] = []

    for timing in timings:

        if not isinstance(
            timing,
            dict,
        ):
            continue

        begin = pd.to_datetime(
            timing.get("begin"),
            errors="coerce",
            utc=True,
        )

        end = pd.to_datetime(
            timing.get("end"),
            errors="coerce",
            utc=True,
        )

        if (
            pd.isna(begin)
            or pd.isna(end)
        ):
            continue

        if begin <= end and end >= cutoff_date:
            eligible_timings.append(
                {
                    "begin": begin.isoformat(),
                    "end": end.isoformat(),
                }
            )

    return eligible_timings


# ---------------------------------------------------------------------
# Sélection culturelle
# ---------------------------------------------------------------------

def is_cultural_event(
    row: pd.Series,
) -> bool:
    """Détermine si un événement appartient au périmètre culturel.

    La règle appliquée est :
    1. exclusion prioritaire des agendas explicitement non culturels ;
    2. inclusion des agendas explicitement culturels ;
    3. analyse lexicale du contenu pour les agendas généralistes.
    """

    agenda_text = clean_text(
        row.get(
            "originagenda_title"
        )
    )

    if any(
        contains_term(
            agenda_text,
            term,
        )
        for term in NON_CULTURAL_AGENDA_TERMS
    ):
        return False

    if any(
        contains_term(
            agenda_text,
            term,
        )
        for term in CULTURAL_AGENDA_TERMS
    ):
        return True

    event_text = " ".join(
        [
            clean_text(
                row.get("title_fr")
            ),
            clean_text(
                row.get("keywords_fr")
            ),
            clean_text(
                row.get("description_fr")
            ),
            clean_text(
                row.get("longdescription_fr")
            ),
        ]
    )

    return any(
        contains_term(
            event_text,
            term,
        )
        for term in CULTURAL_TERMS
    )


# ---------------------------------------------------------------------
# Texte destiné à la future vectorisation
# ---------------------------------------------------------------------

def build_embedding_text(
    row: pd.Series,
) -> str:
    """Construit le texte qui sera vectorisé ultérieurement."""

    components: list[str] = []

    if row["title"]:
        components.append(
            f"Titre : {row['title']}"
        )

    if row["description"]:
        components.append(
            f"Description : {row['description']}"
        )

    if row["long_description"]:
        components.append(
            f"Détails : {row['long_description']}"
        )

    if row["keywords"]:
        components.append(
            f"Mots-clés : {row['keywords']}"
        )

    if row["location_text"]:
        components.append(
            f"Localisation : {row['location_text']}"
        )

    if row["date_range"]:
        components.append(
            f"Dates : {row['date_range']}"
        )

    return "\n".join(
        components
    )


# ---------------------------------------------------------------------
# Date de référence
# ---------------------------------------------------------------------

def get_reference_date(
    value: str | None,
) -> pd.Timestamp:
    """Retourne la date de référence en UTC."""

    if value:
        reference = pd.Timestamp(
            value,
            tz="Europe/Paris",
        )

    else:
        reference = pd.Timestamp.now(
            tz="Europe/Paris"
        ).normalize()

    return reference.tz_convert(
        "UTC"
    )


# ---------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------

def main() -> None:
    """Exécute le pipeline complet de pré-processing OpenAgenda."""

    args = parse_args()

    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Fichier source introuvable : {RAW_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print(
        "PRE-PROCESSING OPENAGENDA - PULS-EVENTS"
    )
    print("=" * 70)

    # -----------------------------------------------------------------
    # Chargement
    # -----------------------------------------------------------------

    df = pd.read_csv(
        RAW_PATH,
        sep=";",
        encoding="utf-8-sig",
        encoding_errors="replace",
        low_memory=False,
    )

    initial_count = len(df)

    print(
        f"\nLignes brutes : {initial_count:,}"
    )

    reference_date = get_reference_date(
        args.reference_date
    )

    cutoff_date = (
        reference_date
        - pd.DateOffset(
            years=1
        )
    )

    print(
        f"Date de référence : {reference_date}"
    )

    print(
        f"Début historique  : {cutoff_date}"
    )

    # -----------------------------------------------------------------
    # Conversion des dates globales
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Filtre géographique
    # -----------------------------------------------------------------

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

    after_region = len(df)

    # -----------------------------------------------------------------
    # Filtre temporel sur les créneaux réels
    # -----------------------------------------------------------------

    df["eligible_timings"] = (
        df["timings"]
        .apply(
            lambda value: parse_and_filter_timings(
                value,
                cutoff_date,
            )
        )
    )

    has_eligible_timing = (
        df["eligible_timings"]
        .map(len)
        .gt(0)
    )

    df = df.loc[
        has_eligible_timing
    ].copy()

    df["eligible_first_begin"] = (
        df["eligible_timings"]
        .apply(
            lambda timings: min(
                pd.Timestamp(
                    item["begin"]
                )
                for item in timings
            )
        )
    )

    df["eligible_last_end"] = (
        df["eligible_timings"]
        .apply(
            lambda timings: max(
                pd.Timestamp(
                    item["end"]
                )
                for item in timings
            )
        )
    )

    df["eligible_timings_json"] = (
        df["eligible_timings"]
        .apply(
            lambda timings: json.dumps(
                timings,
                ensure_ascii=False,
            )
        )
    )

    after_date = len(df)

    # -----------------------------------------------------------------
    # Statut
    # -----------------------------------------------------------------

    df["status_clean"] = (
        df["status"]
        .apply(
            extract_json_label
        )
    )

    not_cancelled = (
        ~df["status_clean"]
        .str.casefold()
        .eq("annulé")
    )

    df = df.loc[
        not_cancelled
    ].copy()

    after_status = len(df)

    # -----------------------------------------------------------------
    # Filtre culturel
    # -----------------------------------------------------------------

    df["is_cultural"] = (
        df.apply(
            is_cultural_event,
            axis=1,
        )
    )

    cultural_candidates_before_filter = int(
        df["is_cultural"].sum()
    )

    df = df.loc[
        df["is_cultural"]
    ].copy()

    after_cultural = len(df)

    # -----------------------------------------------------------------
    # Nettoyage des textes
    # -----------------------------------------------------------------

    text_mapping = {
        "title_fr": "title",
        "description_fr": "description",
        "longdescription_fr": "long_description",
        "keywords_fr": "keywords",
        "location_name": "location_name",
        "location_address": "address",
        "location_city": "city",
        "originagenda_title": "source_agenda",
    }

    for source, target in text_mapping.items():
        df[target] = (
            df[source]
            .apply(
                clean_text
            )
        )

    # -----------------------------------------------------------------
    # Géographie normalisée
    # -----------------------------------------------------------------

    df["department"] = (
        df["location_department"]
        .apply(
            normalize_department
        )
    )

    df["region"] = (
        df["location_region"]
        .apply(
            clean_text
        )
    )

    df["country_code"] = (
        df["location_countrycode"]
        .fillna("")
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # -----------------------------------------------------------------
    # Coordonnées
    # -----------------------------------------------------------------

    coordinates = (
        df["location_coordinates"]
        .apply(
            parse_coordinates
        )
    )

    df["latitude"] = (
        coordinates
        .apply(
            lambda value: value[0]
        )
    )

    df["longitude"] = (
        coordinates
        .apply(
            lambda value: value[1]
        )
    )

    # -----------------------------------------------------------------
    # Autres métadonnées
    # -----------------------------------------------------------------

    df["attendance_mode"] = (
        df["attendancemode"]
        .apply(
            extract_json_label
        )
    )

    df["canonical_url"] = (
        df["canonicalurl"]
        .apply(
            clean_text
        )
    )

    df["registration"] = (
        df["registration"]
        .apply(
            clean_text
        )
    )

    df["date_range"] = (
        df["daterange_fr"]
        .apply(
            clean_text
        )
    )

    df["timings_json"] = (
        df["timings"]
        .fillna("")
        .astype(str)
    )

    df["postal_code"] = (
        df["location_postalcode"]
        .astype("Int64")
        .astype("string")
        .fillna("")
    )

    # -----------------------------------------------------------------
    # Construction d'une localisation exploitable
    # -----------------------------------------------------------------

    df["location_text"] = (
        df.apply(
            build_location_text,
            axis=1,
        )
    )

    # -----------------------------------------------------------------
    # Qualité textuelle minimale
    # -----------------------------------------------------------------

    text_available = (
        df["title"].ne("")
        & (
            df["description"].ne("")
            | df["long_description"].ne("")
        )
    )

    df = df.loc[
        text_available
    ].copy()

    after_text_quality = len(df)

    # -----------------------------------------------------------------
    # Déduplication par UID
    # -----------------------------------------------------------------

    before_uid_dedup = len(df)

    df = (
        df
        .sort_values(
            "updatedat",
            ascending=False,
        )
        .drop_duplicates(
            subset=["uid"],
            keep="first",
        )
        .copy()
    )

    duplicates_removed = (
        before_uid_dedup
        - len(df)
    )

    # -----------------------------------------------------------------
    # Texte pour la future vectorisation
    # -----------------------------------------------------------------

    df["text_for_embedding"] = (
        df.apply(
            build_embedding_text,
            axis=1,
        )
    )

    # -----------------------------------------------------------------
    # Colonnes finales
    # -----------------------------------------------------------------

    final_columns = [
        "uid",
        "title",
        "description",
        "long_description",
        "keywords",
        "text_for_embedding",
        "firstdate_begin",
        "firstdate_end",
        "lastdate_begin",
        "lastdate_end",
        "eligible_first_begin",
        "eligible_last_end",
        "date_range",
        "eligible_timings_json",
        "timings_json",
        "location_name",
        "address",
        "postal_code",
        "city",
        "location_text",
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
        "updatedat",
    ]

    result = (
        df[final_columns]
        .sort_values(
            [
                "eligible_first_begin",
                "uid",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # -----------------------------------------------------------------
    # Export CSV
    # -----------------------------------------------------------------

    result.to_csv(
        OUTPUT_PATH,
        index=False,
        encoding="utf-8",
    )

    # -----------------------------------------------------------------
    # Rapport de qualité
    # -----------------------------------------------------------------

    report = {
        "source_file": str(
            RAW_PATH
        ),
        "output_file": str(
            OUTPUT_PATH
        ),
        "target_region": TARGET_REGION,
        "reference_date_utc": (
            reference_date.isoformat()
        ),
        "history_cutoff_utc": (
            cutoff_date.isoformat()
        ),
        "initial_rows": int(
            initial_count
        ),
        "after_region_filter": int(
            after_region
        ),
        "after_timing_filter": int(
            after_date
        ),
        "events_removed_by_timing_filter": int(
            after_region - after_date
        ),
        "after_status_filter": int(
            after_status
        ),
        "events_removed_as_cancelled": int(
            after_date - after_status
        ),
        "cultural_candidates": int(
            cultural_candidates_before_filter
        ),
        "after_cultural_filter": int(
            after_cultural
        ),
        "events_removed_as_non_cultural": int(
            after_status - after_cultural
        ),
        "after_text_quality_filter": int(
            after_text_quality
        ),
        "events_removed_for_text_quality": int(
            after_cultural
            - after_text_quality
        ),
        "duplicate_uids_removed": int(
            duplicates_removed
        ),
        "final_rows": int(
            len(result)
        ),
        "missing_title": int(
            result["title"]
            .eq("")
            .sum()
        ),
        "missing_city": int(
            result["city"]
            .eq("")
            .sum()
        ),
        "missing_coordinates": int(
            (
                result["latitude"].isna()
                | result["longitude"].isna()
            ).sum()
        ),
        "missing_location_text": int(
            result["location_text"]
            .eq("")
            .sum()
        ),
        "events_without_city_or_coordinates": int(
            (
                result["city"]
                .eq("")
                & result["latitude"].isna()
                & result["longitude"].isna()
            ).sum()
        ),
        "unique_uids": int(
            result["uid"]
            .nunique()
        ),
        "min_eligible_begin": (
            result["eligible_first_begin"]
            .min()
            .isoformat()
            if not result.empty
            else None
        ),
        "max_eligible_end": (
            result["eligible_last_end"]
            .max()
            .isoformat()
            if not result.empty
            else None
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

    # -----------------------------------------------------------------
    # Résumé console
    # -----------------------------------------------------------------

    print("\n--- RESULTATS ---")

    for key, value in report.items():
        print(
            f"{key}: {value}"
        )

    print("\nFichier produit :")
    print(
        OUTPUT_PATH
    )

    print("\nRapport produit :")
    print(
        REPORT_PATH
    )

    print(
        "\nPRE-PROCESSING : TERMINE"
    )


if __name__ == "__main__":
    main()