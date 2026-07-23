"""Navegación superior reutilizable de Crime Pulse Madrid."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape

import streamlit as st


@dataclass(frozen=True)
class NavigationItem:
    """Define una entrada del menú principal en un único lugar."""

    item_id: str
    label: str
    page: str | None
    weight: float


# Para activar un módulo futuro solo hay que sustituir ``None`` por su ruta.
NAVIGATION_ITEMS = (
    NavigationItem("home", "HOME", "app.py", 1.0),
    NavigationItem("profile", "PERFIL DELICTIVO", "pages/2_Perfil_Delictivo.py", 1.35),
    NavigationItem("map", "MAPA CRIMINAL", "pages/1_Mapa_Criminal.py", 1.25),
    NavigationItem(
        "optimization",
        "OPTIMIZACIÓN",
        "pages/3_Optimizacion_Policial.py",
        1.35,
    ),
    NavigationItem(
        "predictions",
        "PREDICCIONES",
        "pages/4_Predicciones.py",
        1.25,
    ),
    NavigationItem(
        "simulator",
        "SIMULADOR",
        "pages/5_Simulador.py",
        1.15,
    ),
)


def _render_navigation_item(column, item: NavigationItem, active: str) -> None:
    """Renderiza un enlace real o una entrada futura claramente inactiva."""
    with column:
        if item.page is None:
            st.markdown(
                f'<div class="top-nav-disabled" aria-disabled="true">{escape(item.label)}</div>',
                unsafe_allow_html=True,
            )
            return

        state = "active" if item.item_id == active else "idle"
        # La clave del contenedor crea una clase estable para el estado visual.
        # El enlace sigue activo incluso cuando representa la página actual.
        with st.container(key=f"nav_item_{item.item_id}_{state}"):
            st.page_link(item.page, label=item.label, width="stretch")


def render_top_navigation(active: str) -> None:
    """Renderiza la barra sticky con enlaces nativos de Streamlit."""
    column_weights = [2.2, *(item.weight for item in NAVIGATION_ITEMS)]

    with st.container(key="top_nav"):
        brand_col, *navigation_columns = st.columns(
            column_weights,
            vertical_alignment="center",
        )

        with brand_col:
            st.markdown(
                '''
                <a href="/" target="_self" class="top-nav-brand-link" aria-label="Ir a Home">\n                <div class="top-nav-brand">
                    <div class="brand-symbol" aria-hidden="true">
                        <svg viewBox="0 0 100 100" role="img">
                            <circle cx="50" cy="50" r="35" class="pulse-ring"></circle>
                            <path d="M12 52 H31 L39 34 L50 69 L61 43 L69 52 H88"
                                  class="pulse-line"></path>
                        </svg>
                    </div>
                    <div class="brand-divider"></div>
                    <div class="brand-copy">
                        <div class="brand-name">CRIME PULSE</div>
                        <div class="brand-city">MADRID</div>
                    </div>
                </div>
                </a>
                ''',
                unsafe_allow_html=True,
            )

        for column, item in zip(navigation_columns, NAVIGATION_ITEMS):
            _render_navigation_item(column, item, active)
