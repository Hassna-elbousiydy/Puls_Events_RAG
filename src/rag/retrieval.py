"""Retrieval RAG réutilisable pour Puls-Events.

Ce module :
- charge l'index FAISS ;
- utilise mistral-embed pour vectoriser la question ;
- applique éventuellement un filtre de ville ;
- récupère les chunks sémantiquement proches ;
- regroupe les chunks par événement ;
- reconstruit le contexte complet des événements sélectionnés.

Il n'effectue aucun appel au modèle Chat Mistral.
"""

from __future__ import annotations

import os
import json
import pandas as pd
from src.rag.errors import invoke_safely
from collections import OrderedDict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_mistralai import MistralAIEmbeddings


DEFAULT_INDEX_PATH = Path("vectorstore/faiss_index")
DEFAULT_EMBEDDING_MODEL = "mistral-embed"


class EventRetriever:
    """Retriever sémantique pour les événements Puls-Events."""

    def __init__(
        self,
        index_path: Path | str = DEFAULT_INDEX_PATH,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embeddings: Any | None = None,
    ) -> None:
        """Charge les embeddings et l'index FAISS."""

        load_dotenv()

        api_key = os.getenv("MISTRAL_API_KEY")

        if not api_key and embeddings is None:
            raise RuntimeError(
                "MISTRAL_API_KEY est absente du fichier .env."
            )

        self.index_path = Path(index_path)

        if not self.index_path.exists():
            raise FileNotFoundError(
                f"Index FAISS introuvable : {self.index_path}"
            )

        if embedding_model != DEFAULT_EMBEDDING_MODEL:
            raise ValueError("L’index exige mistral-embed ; un autre modèle nécessite une reconstruction.")
        self.embeddings = embeddings if embeddings is not None else MistralAIEmbeddings(
            model=embedding_model,
            api_key=api_key,
            max_retries=0,
            timeout=30,
        )

        self.vectorstore = FAISS.load_local(
            str(self.index_path),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )

        if self.vectorstore.index.d != 1024:
            raise ValueError("Dimension de l’index incompatible avec mistral-embed.")
        self.documents_by_uid = self._index_documents_by_uid()

    def _index_documents_by_uid(self) -> dict[str, list]:
        """Regroupe une fois tous les chunks du docstore par UID."""

        documents_by_uid: dict[str, list] = {}

        for document_id in (
            self.vectorstore.index_to_docstore_id.values()
        ):
            document = self.vectorstore.docstore.search(
                document_id
            )

            if not hasattr(document, "metadata"):
                continue

            uid = str(
                document.metadata.get("uid", "")
            ).strip()

            if not uid:
                continue

            documents_by_uid.setdefault(
                uid,
                [],
            ).append(document)

        for documents in documents_by_uid.values():
            documents.sort(
                key=self._chunk_order
            )

        return documents_by_uid

    @staticmethod
    def _chunk_order(document: Any) -> int:
        """Retourne l'ordre d'un chunk dans son événement."""

        try:
            return int(
                document.metadata.get(
                    "chunk_index",
                    0,
                )
            )
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _city_filter(city: str):
        """Construit un filtre de ville insensible à la casse."""

        expected_city = city.strip().casefold()

        def matches_city(metadata: dict[str, Any]) -> bool:
            actual_city = str(
                metadata.get(
                    "city",
                    "",
                )
            ).strip().casefold()

            return actual_city == expected_city

        return matches_city

    def retrieve_events(
        self,
        question: str,
        city: str | None = None,
        search_k: int = 20,
        fetch_k: int = 500,
        top_events: int = 5,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Recherche les événements les plus pertinents.

        Parameters
        ----------
        question:
            Question de l'utilisateur.
        city:
            Ville à filtrer. Aucun filtre si None.
        search_k:
            Nombre maximum de chunks conservés après filtrage.
        fetch_k:
            Nombre de voisins FAISS examinés avant filtrage.
        top_events:
            Nombre maximum d'événements distincts retournés.
        """

        if not question.strip():
            raise ValueError(
                "La question ne peut pas être vide."
            )

        if min(search_k, fetch_k, top_events) <= 0:
            raise ValueError("Les nombres de résultats doivent être strictement positifs.")
        reference = pd.Timestamp(os.getenv("PULS_REFERENCE_DATE") or pd.Timestamp.now(tz="Europe/Paris").date(), tz="Europe/Paris")
        cutoff = (reference - pd.DateOffset(years=1)).tz_convert("UTC")
        lower = pd.Timestamp(start_date, tz="Europe/Paris").tz_convert("UTC") if start_date else cutoff
        upper = (pd.Timestamp(end_date, tz="Europe/Paris") + pd.DateOffset(days=1)).tz_convert("UTC") if end_date else None
        if upper is not None and lower >= upper:
            raise ValueError("La période demandée est inversée.")
        city_matches = self._city_filter(city) if city else lambda metadata: True

        def metadata_filter(metadata):
            if metadata.get("region") != "Pays de la Loire" or not city_matches(metadata):
                return False
            try:
                timings = json.loads(metadata.get("eligible_timings_json", "[]"))
                return any(pd.Timestamp(t["end"]) >= max(lower, cutoff)
                           and (upper is None or pd.Timestamp(t["begin"]) < upper)
                           for t in timings)
            except (ValueError, TypeError, KeyError):
                return False

        # IndexFlatL2 réalise déjà une recherche exhaustive. Examiner tous les
        # voisins évite de perdre une ville rare ou une période après filtrage.
        raw_results = invoke_safely(
            self.vectorstore.similarity_search_with_score,
            question, k=search_k, filter=metadata_filter,
            fetch_k=self.vectorstore.index.ntotal,
        )

        grouped: OrderedDict[
            str,
            dict[str, Any],
        ] = OrderedDict()

        for document, score in raw_results:
            metadata = document.metadata

            uid = str(
                metadata.get(
                    "uid",
                    "",
                )
            ).strip()

            if not uid:
                continue

            if uid not in grouped:
                grouped[uid] = {
                    "uid": uid,
                    "title": metadata.get(
                        "title"
                    ),
                    "city": metadata.get(
                        "city"
                    ),
                    "location_text": metadata.get(
                        "location_text"
                    ),
                    "date_range": metadata.get(
                        "date_range"
                    ),
                    "canonical_url": metadata.get(
                        "canonical_url"
                    ),
                    "source_agenda": metadata.get(
                        "source_agenda"
                    ),
                    "eligible_timings_json": metadata.get("eligible_timings_json", "[]"),
                    "best_score": float(score),
                    "matched_chunk_ids": [],
                }

            grouped[uid][
                "matched_chunk_ids"
            ].append(
                metadata.get(
                    "chunk_id"
                )
            )

            grouped[uid]["best_score"] = min(
                grouped[uid]["best_score"],
                float(score),
            )

        events = sorted(
            grouped.values(),
            key=lambda event: event[
                "best_score"
            ],
        )

        events = events[:top_events]

        for event in events:
            all_documents = (
                self.documents_by_uid.get(
                    event["uid"],
                    [],
                )
            )

            event["total_chunks"] = len(
                all_documents
            )

            event["contents"] = [
                document.page_content.strip()
                for document in all_documents
                if document.page_content.strip()
            ]

        return events

    def build_context(
        self,
        events: list[dict[str, Any]],
    ) -> str:
        """Construit le contexte qui sera transmis au LLM."""

        context_blocks = []

        for rank, event in enumerate(
            events,
            start=1,
        ):
            unique_contents = []

            seen_contents = set()

            for content in event.get(
                "contents",
                [],
            ):
                if (
                    content
                    and content not in seen_contents
                ):
                    unique_contents.append(
                        content
                    )

                    seen_contents.add(
                        content
                    )

            combined_content = "\n".join(
                unique_contents
            )

            block = (
                f"[ÉVÉNEMENT {rank}]\n"
                f"UID : {event['uid']}\n"
                f"Titre : {event['title']}\n"
                f"Ville : {event['city']}\n"
                f"Lieu : {event['location_text']}\n"
                f"Dates : {event['date_range']}\n"
                f"URL : {event['canonical_url']}\n"
                f"Créneaux admissibles (UTC) : {event.get('eligible_timings_json', 'non disponibles')}\n"
                f"Informations descriptives (peuvent évoquer des dates historiques) :\n"
                f"{combined_content}"
            )

            context_blocks.append(
                block
            )

        separator = (
            "\n\n"
            + "=" * 60
            + "\n\n"
        )

        return separator.join(
            context_blocks
        )

    def retrieve_context(
        self,
        question: str,
        city: str | None = None,
        search_k: int = 20,
        fetch_k: int = 500,
        top_events: int = 5,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """Effectue retrieval + construction du contexte."""

        events = self.retrieve_events(
            question=question,
            city=city,
            search_k=search_k,
            fetch_k=fetch_k,
            top_events=top_events,
            start_date=start_date,
            end_date=end_date,
        )

        context = self.build_context(
            events
        )

        return events, context