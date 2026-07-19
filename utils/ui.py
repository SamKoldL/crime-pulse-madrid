"""HTML presentation components for the premium Streamlit interface."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import TYPE_CHECKING

import streamlit as st

from utils.metrics import DashboardSnapshot

if TYPE_CHECKING:
    from utils.home_data import HomeSignal, HomeSnapshot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STYLES_PATH = PROJECT_ROOT / "assets" / "styles.css"


def _format_integer(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _format_decimal(value: float, digits: int = 1) -> str:
    formatted = f"{value:,.{digits}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def inject_global_styles() -> None:
    """Load the shared design system into Streamlit."""
    css = STYLES_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown(
        """
        <section class="hero-shell">
            <div class="hero-grid"></div>
            <div class="hero-orbit hero-orbit-one"></div>
            <div class="hero-orbit hero-orbit-two"></div>
            <div class="hero-scanline"></div>
            <div class="hero-content">
                <div class="brand-mark"><span></span> INTELIGENCIA URBANA · COMUNIDAD DE MADRID</div>
                <h1>CRIME PULSE <em>MADRID</em></h1>
                <h2>Inteligencia territorial para una seguridad pública basada en datos</h2>
                <p>Análisis multidimensional de criminalidad, entorno urbano y factores
                socioeconómicos en la Comunidad de Madrid.</p>
                <div class="hero-meta">
                    <span>37 MUNICIPIOS</span><i></i><span>2023—2025</span><i></i><span>VISIÓN TERRITORIAL</span>
                </div>
            </div>
            <div class="hero-coordinate">40.4168° N &nbsp; 3.7038° W</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _build_kpi_html(
    label: str,
    value: str,
    details: tuple[str, ...],
    *,
    featured: bool = False,
) -> str:
    """Build exactly one compact, fully closed KPI card."""
    modifier = " kpi-card-featured" if featured else ""
    detail_html = "".join(
        f'<div class="kpi-card-detail">{escape(detail)}</div>' for detail in details
    )
    return f'<article class="kpi-card{modifier}" aria-label="{escape(label)}"><div class="kpi-card-label">{escape(label)}</div><div class="kpi-card-value">{escape(value)}</div><div class="kpi-card-details">{detail_html}</div></article>'


def _render_kpi_with_fallback(
    card_html: str,
    label: str,
    value: str,
    details: tuple[str, ...],
) -> None:
    """Render one premium KPI card in the main Streamlit document."""
    try:
        # Compact single-line HTML avoids Markdown code-block interpretation and
        # remains in the main document, where the global .kpi-card CSS applies.
        st.markdown(card_html, unsafe_allow_html=True)
    except Exception:
        with st.container(border=True):
            st.metric(label, value)
            for detail in details:
                st.caption(detail)

def render_kpi_grid(snapshot: DashboardSnapshot) -> None:
    """Render four isolated premium cards with a native safety fallback."""
    columns = st.columns(4, gap="small")
    cards = (
        (
            "Municipios analizados",
            _format_integer(snapshot.municipality_count),
            ("Cobertura territorial anual",),
            False,
        ),
        (
            "Población analizada",
            _format_integer(snapshot.population),
            ("Habitantes en el ámbito analizado",),
            False,
        ),
        (
            "Delitos totales registrados",
            _format_integer(snapshot.total_crimes),
            (f"Registros agregados · {snapshot.year}",),
            False,
        ),
        (
            "Mayor criminalidad relativa",
            snapshot.top_municipality,
            (
                f"{_format_decimal(snapshot.top_index)} · Índice ponderado / 10.000 hab.",
                f"{_format_decimal(snapshot.top_crime_share)}% de los delitos anuales",
                f"Ranking anual: #{snapshot.top_rank} de {snapshot.municipality_count} municipios",
            ),
            True,
        ),
    )

    for column, (label, value, details, featured) in zip(columns, cards):
        with column:
            card_html = _build_kpi_html(label, value, details, featured=featured)
            _render_kpi_with_fallback(card_html, label, value, details)


def render_section_heading(
    eyebrow: str,
    title: str,
    description: str,
    *,
    home_compact: bool = False,
) -> None:
    css_class = "section-heading section-heading-home" if home_compact else "section-heading"
    st.markdown(
        f"""
        <header class="{css_class}">
            <div><span>{escape(eyebrow)}</span><h3>{escape(title)}</h3></div>
            <p>{escape(description)}</p>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_insights(snapshot: DashboardSnapshot) -> None:
    """Generate descriptive, non-causal annual observations."""
    if snapshot.annual_change is None:
        evolution_text = (
            f"No existe un año anterior comparable para describir la evolución de "
            f"{escape(snapshot.top_municipality)}."
        )
    else:
        direction = "por encima" if snapshot.annual_change >= 0 else "por debajo"
        evolution_text = (
            f"Su índice se sitúa un <strong>{_format_decimal(abs(snapshot.annual_change))}% "
            f"{direction}</strong> del registrado en {snapshot.previous_year}."
        )

    st.markdown(
        f"""
        <section class="insights-shell">
            <div class="insights-title">
                <span>INSIGHTS AUTOMÁTICOS</span>
                <h3>Señales del periodo</h3>
                <p>Lecturas descriptivas generadas a partir de los datos seleccionados. Las diferencias observadas no implican causalidad.</p>
            </div>
            <div class="insight-list">
                <article><b>01</b><div><strong>Máximo relativo</strong><p>{escape(snapshot.top_municipality)} presenta el mayor índice criminal ponderado del periodo, con {_format_decimal(snapshot.top_index)} puntos por 10.000 habitantes.</p></div></article>
                <article><b>02</b><div><strong>Distancia frente al centro</strong><p>El valor máximo está un {_format_decimal(snapshot.difference_from_median)}% por encima de la mediana municipal ({_format_decimal(snapshot.median_index)}).</p></div></article>
                <article><b>03</b><div><strong>Evolución interanual</strong><p>{evolution_text}</p></div></article>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_footer(year: int, *, home_compact: bool = False) -> None:
    css_class = "app-footer app-footer-home" if home_compact else "app-footer"
    st.markdown(
        f"""
        <footer class="{css_class}">
            <span>CRIME PULSE MADRID</span>
            <p>Índice criminal ponderado por gravedad y normalizado por población · Vista {year}</p>
            <span class="footer-status"><i></i> SISTEMA OPERATIVO</span>
        </footer>
        """,
        unsafe_allow_html=True,
    )


def _render_home_metric(
    column,
    key: str,
    label: str,
    value: str,
    details: tuple[str, ...],
    *,
    featured: bool = False,
) -> None:
    with column:
        suffix = "_featured" if featured else ""
        with st.container(border=True, key=f"home_kpi_{key}{suffix}"):
            st.metric(label, value)
            for detail in details:
                st.caption(detail)


def render_home_kpis(snapshot: HomeSnapshot) -> None:
    """Cuatro KPIs ejecutivos con contenido contextual y componentes nativos."""
    if snapshot.yoy_change is None:
        yoy_value = "SIN HISTÓRICO COMPARABLE"
        yoy_detail = "No existe una base anterior con el mismo ámbito válido"
    else:
        yoy_value = f"{snapshot.yoy_change:+.1f}%".replace(".", ",")
        yoy_detail = (
            f"{snapshot.year} vs {snapshot.previous_year} · "
            f"{snapshot.comparable_municipalities} municipios comunes"
        )

    relative_value = (
        _format_decimal(snapshot.relative_metric_value, 1)
        if snapshot.relative_metric_value is not None
        else "NO DISPONIBLE"
    )
    if snapshot.municipality == "TODOS LOS MUNICIPIOS":
        context_label = "Mayor índice territorial"
        context_value = snapshot.focus_municipality or "NO DISPONIBLE"
        context_detail = (
            f"{_format_decimal(snapshot.focus_metric, 1)} / 10.000 hab. · "
            f"#1 de {snapshot.selected_rank_total}"
            if snapshot.focus_metric is not None
            else f"Sin municipios comparables en {snapshot.year}"
        )
    else:
        context_label = "Posición territorial"
        context_value = (
            f"#{snapshot.selected_rank} de {snapshot.selected_rank_total}"
            if snapshot.selected_rank is not None
            else "NO DISPONIBLE"
        )
        context_detail = (
            f"Índice territorial: {relative_value} / 10.000 hab."
            if snapshot.selected_rank is not None and snapshot.relative_metric_value is not None
            else f"Fuera del universo criminal válido de {snapshot.year}"
        )

    columns = st.columns(4, gap="small")
    cards = (
        (
            "crimes",
            "Delitos registrados",
            _format_integer(snapshot.total_crimes),
            (f"Suma según el filtro activo · {snapshot.year}",),
            False,
        ),
        ("yoy", "Variación interanual", yoy_value, (yoy_detail,), False),
        (
            "index",
            snapshot.relative_metric_label,
            relative_value,
            (snapshot.relative_metric_detail,),
            True,
        ),
        ("context", context_label, context_value, (context_detail,), False),
    )
    for column, (key, label, value, details, featured) in zip(columns, cards):
        _render_home_metric(
            column, key, label, value, details, featured=featured
        )

    st.markdown(
        '<div class="home-coverage-strip">'
        f'<span><b>{_format_integer(snapshot.municipality_count)}</b> municipios incluidos</span>'
        f'<i></i><span><b>{_format_integer(snapshot.population_covered)}</b> habitantes cubiertos</span>'
        f'<i></i><span><b>{escape(snapshot.municipality)}</b> · {escape(snapshot.crime_type)}</span>'
        '</div>',
        unsafe_allow_html=True,
    )


def render_specific_type_context(snapshot: HomeSnapshot) -> None:
    """Sustituye el ranking de drivers cuando ya hay una tipología seleccionada."""
    if snapshot.yoy_change is None:
        st.markdown(
            '<div class="home-type-context"><span>LECTURA ESPECÍFICA</span>'
            f'<h4>{escape(snapshot.crime_type)}</h4>'
            '<p>Sin histórico comparable para cuantificar su contribución interanual en el ámbito seleccionado.</p></div>',
            unsafe_allow_html=True,
        )
        return
    absolute_change = float(snapshot.yoy_current or 0) - float(snapshot.yoy_previous or 0)
    direction = "aumenta" if absolute_change >= 0 else "disminuye"
    st.markdown(
        '<div class="home-type-context"><span>LECTURA ESPECÍFICA</span>'
        f'<h4>{escape(snapshot.crime_type)}</h4>'
        f'<p>En la cohorte comparable, {direction} en <strong>{_format_integer(abs(int(round(absolute_change))))} casos</strong> '
        f'({_format_decimal(abs(snapshot.yoy_change), 1)}%) frente a {snapshot.previous_year}. '
        'La lectura describe evolución observada y no implica causalidad.</p></div>',
        unsafe_allow_html=True,
    )


def render_home_signals(signals: tuple[HomeSignal, ...]) -> None:
    articles = "".join(
        '<article>'
        f'<span>{escape(signal.signal_type)}</span>'
        f'<h4>{escape(signal.title)}</h4>'
        f'<p>{escape(signal.text)}</p>'
        f'<b>PROFUNDIZAR · {escape(signal.module)}</b>'
        '</article>'
        for signal in signals[:4]
    )
    st.markdown(
        '<section class="home-signals"><div class="home-signals-title">'
        '<span>INSIGHTS DINÁMICOS · SIN CAUSALIDAD</span><h3>Señales del periodo</h3>'
        '<p>Lecturas ejecutivas calculadas desde el filtro activo.</p></div>'
        f'<div class="home-signal-grid">{articles}</div></section>',
        unsafe_allow_html=True,
    )
