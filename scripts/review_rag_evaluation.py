"""Relecture humaine interactive du jeu d'évaluation RAG Puls-Events.

Ce script affiche chaque cas du jeu de référence et demande une validation
explicite à une personne. Seuls les cas effectivement validés sont marqués
human_reviewed=true.

Le script crée une sauvegarde avant la première modification et sauvegarde
la progression après chaque cas.
"""

from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path


DATASET_PATH = Path("data/evaluation/rag_evaluation.jsonl")
README_PATH = Path("data/evaluation/README.md")
BACKUP_PATH = Path(
    "data/evaluation/rag_evaluation_before_human_review.jsonl"
)

REVIEW_DATE = date.today().isoformat()


def load_records() -> list[dict]:
    """Charge les cas JSONL."""

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {DATASET_PATH}"
        )

    records: list[dict] = []

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                records.append(
                    json.loads(line)
                )
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"JSON invalide ligne {line_number}"
                ) from exc

    if len(records) != 25:
        raise RuntimeError(
            "Le jeu d'évaluation devrait contenir "
            f"25 cas, mais {len(records)} ont été trouvés."
        )

    return records


def save_records(
    records: list[dict],
) -> None:
    """Sauvegarde le jeu JSONL."""

    with DATASET_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def print_case(
    record: dict,
    position: int,
    total: int,
) -> None:
    """Affiche les informations nécessaires à la relecture."""

    print("\n" + "=" * 90)
    print(
        f"CAS {position}/{total} — "
        f"{record.get('id', '')}"
    )
    print("=" * 90)

    print(
        "\nQUESTION :\n"
        f"{record.get('question', '')}"
    )

    print(
        "\nRÉPONSE DE RÉFÉRENCE :\n"
        f"{record.get('expected_answer', '')}"
    )

    print(
        "\nUID ATTENDU(S) : "
        f"{record.get('expected_uids', [])}"
    )

    print(
        "VILLE ATTENDUE : "
        f"{record.get('expected_city', '') or '(aucune)'}"
    )

    print(
        "CATÉGORIE : "
        f"{record.get('category', '')}"
    )

    references = record.get(
        "references",
        [],
    )

    if not references:
        print(
            "\nRÉFÉRENCE OPENAGENDA : "
            "aucune — cas d'absence ou d'ambiguïté."
        )
        return

    print("\nRÉFÉRENCE(S) OPENAGENDA :")

    for index, reference in enumerate(
        references,
        start=1,
    ):
        print(f"\n  Référence {index}")
        print(
            "  UID   : "
            f"{reference.get('uid', '')}"
        )
        print(
            "  Titre : "
            f"{reference.get('title', '')}"
        )
        print(
            "  Ville : "
            f"{reference.get('city', '')}"
        )
        print(
            "  Dates : "
            f"{reference.get('date_range', '')}"
        )
        print(
            "  Lieu  : "
            f"{reference.get('location_text', '')}"
        )
        print(
            "  URL   : "
            f"{reference.get('canonical_url', '')}"
        )


def update_annotation_method(
    record: dict,
) -> None:
    """Met à jour la provenance de l'annotation."""

    old_method = str(
        record.get(
            "annotation_method",
            "",
        )
    ).strip()

    if (
        "revue humaine à effectuer"
        in old_method.casefold()
    ):
        old_method = old_method.replace(
            "revue humaine à effectuer",
            f"revue humaine effectuée le {REVIEW_DATE}",
        )
        old_method = old_method.replace(
            "Revue humaine à effectuer",
            f"Revue humaine effectuée le {REVIEW_DATE}",
        )

    elif (
        "revue humaine effectuée"
        not in old_method.casefold()
    ):
        if old_method:
            old_method += "; "

        old_method += (
            "revue humaine effectuée "
            f"le {REVIEW_DATE}"
        )

    record["annotation_method"] = (
        old_method
    )


def update_readme(
    reviewed_count: int,
    total: int,
) -> None:
    """Documente la méthode de relecture humaine."""

    if reviewed_count != total:
        return

    content = f"""# Jeu de référence Puls-Events

25 cas rédigés à partir du snapshot OpenAgenda utilisé par le projet.

Aucune réponse générée par Mistral n'est utilisée comme vérité terrain.
Les réponses de référence proviennent des informations structurées des
événements OpenAgenda : UID, titre, ville, lieu, dates et URL.

## Composition du jeu

Le jeu contient 25 questions couvrant plusieurs scénarios :

- recherche d'un événement précis ;
- recherche par ville ;
- exposition, concert, cinéma, patrimoine et autres thèmes culturels ;
- recherche par période ;
- plusieurs événements possibles ;
- absence de résultat dans le périmètre Pays de la Loire ;
- question utilisateur ambiguë.

Les cas avec événements de référence contiennent les UID OpenAgenda
attendus ainsi que leurs métadonnées sources.

Les cas d'absence ou d'ambiguïté sont évalués séparément afin de ne pas
leur attribuer artificiellement des métriques de retrieval classiques.

## Relecture humaine

Date de relecture : {REVIEW_DATE}

Nombre de cas relus : {reviewed_count}/{total}.

Méthode de relecture :

Chaque cas a été relu manuellement par l'auteure du projet.

Pour chaque question, la relecture a porté sur :

- la formulation de la question ;
- la réponse de référence ;
- le ou les UID attendus ;
- la ville ;
- le titre de l'événement ;
- les dates ;
- le lieu ;
- l'URL OpenAgenda ;
- la cohérence globale entre la question et la réponse attendue.

Pour les cas sans événement attendu, la cohérence de la réponse
d'abstention ou de clarification a également été vérifiée.

Les cas effectivement validés portent :

`"human_reviewed": true`

ainsi que la date et la méthode de relecture.

## Limites

Pour les questions ouvertes, les listes d'événements pertinents ne sont
pas nécessairement exhaustives. Une faible précision de retrieval peut
donc refléter l'existence d'autres événements pertinents qui n'ont pas
été inclus dans la référence annotée.

Les réponses annotées servent de vérité terrain pour comparer le sens
et les informations des réponses produites par le système RAG.
"""

    README_PATH.write_text(
        content,
        encoding="utf-8",
    )


def main() -> None:
    """Lance la relecture humaine interactive."""

    records = load_records()

    if not BACKUP_PATH.exists():
        shutil.copy2(
            DATASET_PATH,
            BACKUP_PATH,
        )

        print(
            "Sauvegarde créée : "
            f"{BACKUP_PATH}"
        )

    total = len(records)

    print("\n" + "=" * 90)
    print("RELECTURE HUMAINE — JEU Q/R PULS-EVENTS")
    print("=" * 90)

    print(
        "\nPour chaque cas :"
        "\n  O = je valide ce cas après l'avoir relu"
        "\n  N = je ne valide pas ce cas"
        "\n  Q = quitter et reprendre plus tard"
    )

    for position, record in enumerate(
        records,
        start=1,
    ):
        if record.get(
            "human_reviewed"
        ) is True:
            print(
                f"\n{record.get('id')} "
                "déjà validé — ignoré."
            )
            continue

        print_case(
            record,
            position,
            total,
        )

        while True:
            choice = input(
                "\nAprès avoir relu ce cas, "
                "validez-vous la référence ? "
                "[O/N/Q] : "
            ).strip().casefold()

            if choice in {
                "o",
                "oui",
            }:
                record[
                    "human_reviewed"
                ] = True

                record[
                    "human_reviewed_at"
                ] = REVIEW_DATE

                record[
                    "human_review_method"
                ] = (
                    "Relecture manuelle par "
                    "l'auteure du projet : "
                    "vérification de la question, "
                    "de la réponse attendue, des UID, "
                    "de la ville, des dates, du lieu, "
                    "de l'URL et de la cohérence avec "
                    "les références OpenAgenda."
                )

                record.pop(
                    "human_review_note",
                    None,
                )

                update_annotation_method(
                    record
                )

                print(
                    "✓ Cas validé humainement."
                )
                break

            if choice in {
                "n",
                "non",
            }:
                record[
                    "human_reviewed"
                ] = False

                note = input(
                    "Explique brièvement "
                    "ce qui doit être corrigé : "
                ).strip()

                record[
                    "human_review_note"
                ] = (
                    note
                    or "Cas à corriger."
                )

                print(
                    "✗ Cas laissé non validé."
                )
                break

            if choice in {
                "q",
                "quitter",
            }:
                save_records(
                    records
                )

                reviewed_count = sum(
                    bool(
                        item.get(
                            "human_reviewed"
                        )
                    )
                    for item in records
                )

                print(
                    "\nProgression sauvegardée : "
                    f"{reviewed_count}/{total} "
                    "cas validés."
                )
                return

            print(
                "Réponse invalide. "
                "Tapez O, N ou Q."
            )

        # Sauvegarde après chaque décision.
        save_records(
            records
        )

    reviewed_count = sum(
        bool(
            item.get(
                "human_reviewed"
            )
        )
        for item in records
    )

    update_readme(
        reviewed_count,
        total,
    )

    print("\n" + "=" * 90)
    print("RELECTURE TERMINÉE")
    print("=" * 90)

    print(
        f"\nCas validés humainement : "
        f"{reviewed_count}/{total}"
    )

    if reviewed_count == total:
        print(
            "\n✓ Les 25 références ont été "
            "relues et validées humainement."
        )
        print(
            "✓ data/evaluation/README.md "
            "a été mis à jour."
        )
    else:
        print(
            "\nCertains cas restent à corriger "
            "avant validation finale."
        )


if __name__ == "__main__":
    main()