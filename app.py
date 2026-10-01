from __future__ import annotations

import html
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import streamlit as st

from src.rag.pipeline import PulsEventsRAG


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Puls Events — Découvrez vos sorties",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CSS
# ============================================================

STYLE_PATH = Path("assets/styles.css")

if STYLE_PATH.exists():
    st.markdown(
        f"<style>{STYLE_PATH.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


# ============================================================
# HTML HELPER
# ============================================================

def render_html(content: str) -> None:
    """Affiche proprement un bloc HTML dans Streamlit."""

    clean_html = "".join(
        line.strip()
        for line in content.splitlines()
        if line.strip()
    )

    st.markdown(
        clean_html,
        unsafe_allow_html=True,
    )


# ============================================================
# RAG
# ============================================================

@st.cache_resource(show_spinner=False)
def load_rag() -> PulsEventsRAG:
    """Charge le moteur de recommandation une seule fois."""
    return PulsEventsRAG()


def available_cities(rag: PulsEventsRAG) -> list[str]:
    """Retourne les villes présentes dans le catalogue."""

    cities: set[str] = set()

    for documents in rag.retriever.documents_by_uid.values():

        if not documents:
            continue

        city = str(
            documents[0].metadata.get("city", "")
        ).strip()

        if city:
            cities.add(city)

    return sorted(cities, key=str.casefold)


def safe(value: Any) -> str:
    """Échappe une valeur avant insertion HTML."""

    if value is None:
        return ""

    return html.escape(str(value))


def event_card(
    event: dict[str, Any],
    rank: int,
) -> str:
    """Construit une carte publique pour un événement."""

    title = safe(
        event.get("title")
        or "Événement culturel"
    )

    city = safe(
        event.get("city")
        or "Ville non précisée"
    )

    location = safe(
        event.get("location_text")
        or "Lieu à préciser"
    )

    dates = safe(
        event.get("date_range")
        or "Dates à préciser"
    )

    raw_url = str(
        event.get("canonical_url")
        or ""
    ).strip()

    if raw_url.startswith(
        ("https://", "http://")
    ):

        url = html.escape(
            raw_url,
            quote=True,
        )

        link = (
            f'<a class="event-link" '
            f'href="{url}" '
            f'target="_blank" '
            f'rel="noopener noreferrer">'
            f'Découvrir l’événement ↗'
            f'</a>'
        )

    else:
        link = ""

    return f"""
    <article class="event-card">

        <div>

            <div class="event-number">
                {rank:02d}
            </div>

            <div class="event-title">
                {title}
            </div>

            <div class="event-meta">

                <div class="meta-row">
                    <span class="meta-icon">⌖</span>
                    <span>{location}</span>
                </div>

                <div class="meta-row">
                    <span class="meta-icon">●</span>
                    <span>{city}</span>
                </div>

                <div class="meta-row">
                    <span class="meta-icon">◷</span>
                    <span>{dates}</span>
                </div>

            </div>

        </div>

        {link}

    </article>
    """


def render_events(
    events: list[dict[str, Any]],
) -> None:
    """Affiche les événements proposés à l'utilisateur."""

    if not events:
        return

    with st.expander(
        f"Voir les événements proposés · {len(events)}",
        expanded=False,
    ):

        cards = "".join(
            event_card(event, rank)
            for rank, event
            in enumerate(events, start=1)
        )

        render_html(
            f"""
            <div class="source-grid">
                {cards}
            </div>
            """
        )


# ============================================================
# SESSION
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# ============================================================
# LOAD APPLICATION
# ============================================================

try:

    rag = load_rag()

except Exception as exc:

    # L'erreur complète reste disponible dans le terminal
    # mais n'est pas exposée dans l'interface publique.
    print(f"[Puls Events] Erreur de chargement : {exc}")

    st.error(
        "Puls Events est momentanément indisponible. "
        "Veuillez réessayer dans quelques instants."
    )

    st.stop()


cities = available_cities(rag)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">
            <div class="sidebar-logo">✦</div>

            <div>
                <div class="sidebar-brand-name">
                    Puls Events
                </div>

                <div class="sidebar-brand-tagline">
                    Votre guide culturel
                </div>
            </div>
        </div>
        """
    )

    st.markdown(
        "### Personnalisez votre recherche"
    )

    selected_city = st.selectbox(
        "Ville",
        ["Toutes les villes"] + cities,
    )

    city_filter = (
        None
        if selected_city == "Toutes les villes"
        else selected_city
    )

    use_dates = st.toggle(
        "Choisir une période",
        value=False,
    )

    start_iso: str | None = None
    end_iso: str | None = None

    if use_dates:

        start_value = st.date_input(
            "Du",
            value=date.today(),
        )

        end_value = st.date_input(
            "Au",
            value=(
                date.today()
                + timedelta(days=30)
            ),
        )

        start_iso = start_value.isoformat()
        end_iso = end_value.isoformat()

    top_events = st.slider(
        "Nombre de suggestions",
        min_value=1,
        max_value=5,
        value=3,
    )

    st.divider()

    st.markdown(
        "### À propos"
    )

    st.caption(
        "Indiquez simplement ce que vous recherchez : "
        "une ville, une période ou une envie. "
        "Puls Events vous propose des sorties culturelles "
        "adaptées à votre demande."
    )

    if st.button(
        "Nouvelle recherche",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# HERO
# ============================================================

render_html(
    """
    <section class="hero">

        <div class="hero-inner">

            <div class="hero-copy">

                <div class="hero-badge">
                    PAYS DE LA LOIRE · SORTIES CULTURELLES
                </div>

                <h1>Et si votre prochaine <span class="gradient-text">découverte</span> était juste ici ?</h1>

                <p class="hero-subtitle">Concerts, expositions, spectacles, festivals et découvertes locales. Dites-nous ce qui vous fait envie, Puls Events s’occupe du reste.</p>

                <div class="hero-location-line">
                    Nantes · Angers · Laval · Le Mans ·
                    et toute la région
                </div>

            </div>


            <div class="hero-art" aria-hidden="true">

                <div class="art-glow art-glow-one"></div>
                <div class="art-glow art-glow-two"></div>

                <div class="floating-card floating-card-one">
                    <div class="floating-icon">♪</div>
                    <div>
                        <strong>Concerts</strong>
                        <span>Vibrer</span>
                    </div>
                </div>

                <div class="floating-card floating-card-two">
                    <div class="floating-icon">◫</div>
                    <div>
                        <strong>Expositions</strong>
                        <span>Explorer</span>
                    </div>
                </div>

                <div class="floating-card floating-card-three">
                    <div class="floating-icon">✦</div>
                    <div>
                        <strong>Spectacles</strong>
                        <span>S’émerveiller</span>
                    </div>
                </div>

                <div class="art-center">
                    <span>Découvrir</span>
                    <strong>autrement.</strong>
                </div>

            </div>

        </div>

    </section>
    """
)


# ============================================================
# DISCOVERY
# ============================================================

render_html(
    """
    <section class="discovery-heading">

        <div class="section-label">
            Quelques idées
        </div>

        <h2 class="section-title">
            Que souhaitez-vous découvrir aujourd’hui ?
        </h2>

        <p class="section-description">
            Inspirez-vous d’une suggestion ou posez
            directement votre propre question.
        </p>

    </section>
    """
)


suggestions = [
    "Un concert à Nantes",
    "Une exposition à Angers",
    "Une idée pour ce week-end",
    "Surprenez-moi",
]


suggestion_cols = st.columns(4)


selected_prompt: str | None = None


suggestion_questions = {
    "Un concert à Nantes":
        "Quels concerts puis-je découvrir à Nantes ?",

    "Une exposition à Angers":
        "Quelles expositions puis-je voir à Angers ?",

    "Une idée pour ce week-end":
        "Que puis-je faire culturellement ce week-end ?",

    "Surprenez-moi":
        "Recommande-moi une sortie culturelle intéressante.",
}


for idx, suggestion in enumerate(
    suggestions
):

    with suggestion_cols[idx]:

        if st.button(
            suggestion,
            key=f"suggestion_{idx}",
            use_container_width=True,
        ):

            selected_prompt = (
                suggestion_questions[suggestion]
            )


# ============================================================
# HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"]
            == "assistant"
        ):

            render_events(
                message.get(
                    "events",
                    [],
                )
            )


# ============================================================
# CHAT
# ============================================================

typed_prompt = st.chat_input(
    "Ex. Que puis-je faire à Nantes ce week-end ?"
)


prompt = (
    selected_prompt
    or typed_prompt
)


if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)


    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Je cherche les meilleures idées pour vous…"
            ):

                result = rag.ask(
                    question=prompt,
                    city=city_filter,
                    start_date=start_iso,
                    end_date=end_iso,
                    top_events=top_events,
                )

            answer = result["answer"]

            events = result.get(
                "events",
                [],
            )

            st.markdown(answer)

            render_events(events)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "events": events,
                }
            )

        except Exception as exc:

            print(
                "[Puls Events] "
                f"Erreur pendant la requête : {exc}"
            )

            st.error(
                "Je n’ai pas pu terminer cette recherche. "
                "Veuillez réessayer dans quelques instants."
            )


# ============================================================
# FOOTER
# ============================================================

render_html(
    """
    <footer class="custom-footer">

        <span class="footer-brand">
            ✦ Puls Events
        </span>

        <span class="footer-text">
            Trouvez une sortie qui vous ressemble.
        </span>

    </footer>
    """
)