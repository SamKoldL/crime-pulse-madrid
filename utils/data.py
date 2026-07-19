"""Read-only access and validation for the project master dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "tabla_maestra.csv"

REQUIRED_COLUMNS = {
    "Municipio",
    "Año",
    "Población",
    "Delitos_totales",
    "Indice_criminal_ponderado_por_10000_hab",
    "Ranking_criminal_anual",
}


@st.cache_data(show_spinner=False)
def load_master_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the source CSV without transforming or persisting its values."""
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo {path.name}.")

    frame = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Faltan columnas requeridas: {missing}.")

    return frame


def get_available_years(frame: pd.DataFrame) -> list[int]:
    """Return valid years in reverse chronological order."""
    return sorted(frame["Año"].dropna().astype(int).unique().tolist(), reverse=True)
