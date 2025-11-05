# app.py
import io
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

# Importa tus módulos del proyecto (layout actual: src/…)
from src.scenarios import baseline, bloqueo, anchos
from src.metrics import plot_curva, plot_curvas_comparadas

# -------------------------------------------------------
# Config general de la app
# -------------------------------------------------------
st.set_page_config(page_title="ModSim Evacuación", layout="wide")

st.title("🏃‍♀️💨 Simulación de Evacuación — ModSim")

st.caption(
    "Frontend Streamlit sobre tu modelo ABM (Mesa). "
    "Corre escenarios, visualiza curvas y descarga resultados."
)

# -------------------------------------------------------
# Helpers UI
# -------------------------------------------------------
def _plot_line(ts, perc, title="Curva de evacuación"):
    fig, ax = plt.subplots()
    ax.plot(ts, perc)
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("% evacuado")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    st.pyplot(fig, clear_figure=True)


def _download_df_button(df: pd.DataFrame, filename: str, label: str = "Descargar CSV"):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv,
        file_name=filename,
        mime="text/csv",
    )


@st.cache_data(show_spinner=False)
def run_baseline_cached(N, width, height, num_exits, seed, max_steps):
    return baseline(N=N, width=width, height=height, num_exits=num_exits, seed=seed, max_steps=max_steps)

@st.cache_data(show_spinner=False)
def run_bloqueo_cached(N, width, height, num_exits, seed, t_bloqueo, exit_index, max_steps):
    return bloqueo(
        N=N, width=width, height=height, num_exits=num_exits, seed=seed,
        t_bloqueo=t_bloqueo, exit_index=exit_index, max_steps=max_steps
    )

@st.cache_data(show_spinner=False)
def run_anchos_cached(N, width, height, lista_anchos, seed, max_steps):
    return anchos(
        N=N, width=width, height=height, lista_anchos=tuple(lista_anchos), seed=seed, max_steps=max_steps
    )


# -------------------------------------------------------
# Sidebar: parámetros comunes
# -------------------------------------------------------
st.sidebar.header("Parámetros comunes")
agents = st.sidebar.number_input("Agentes (N)", min_value=10, max_value=5000, value=300, step=50)
width = st.sidebar.number_input("Ancho grid (celdas)", min_value=5, max_value=200, value=25, step=5)
height = st.sidebar.number_input("Alto grid (celdas)", min_value=5, max_value=200, value=25, step=5)
seed = st.sidebar.number_input("Semilla", min_value=0, max_value=10_000, value=42, step=1)
max_steps = st.sidebar.number_input("Max steps", min_value=100, max_value=200_000, value=5000, step=500)

st.sidebar.markdown("---")
escenario = st.sidebar.selectbox("Escenario", ["Baseline", "Bloqueo", "Anchos (proxy)"])

# -------------------------------------------------------
# Escenario: BASELINE
# -------------------------------------------------------
if escenario == "Baseline":
    st.subheader("Escenario: Baseline")
    num_exits = st.number_input("Número de salidas", min_value=1, max_value=12, value=3, step=1)

    colA, colB = st.columns([1, 2])
    with colA:
        run = st.button("▶️ Ejecutar baseline", type="primary")
    if run:
        with st.spinner("Simulando baseline..."):
            df, ts, perc, metrics = run_baseline_cached(
                N=agents, width=width, height=height, num_exits=num_exits, seed=seed, max_steps=max_steps
            )

        # Métricas
        mdf = pd.DataFrame([metrics])
        st.success("✅ Simulación completada.")
        st.markdown("### Métricas")
        st.dataframe(mdf, use_container_width=True)
        _download_df_button(mdf, "baseline_metrics.csv", "Descargar métricas (CSV)")

        # Curva
        st.markdown("### Curva de evacuación")
        _plot_line(ts, perc, "Curva de evacuación — Baseline")

        # Tiempos individuales
        st.markdown("### Tiempos individuales")
        if df is not None and not df.empty:
            st.dataframe(df.head(500), use_container_width=True)
            _download_df_button(df, "baseline_times.csv", "Descargar tiempos (CSV)")
        else:
            st.info("No hay tiempos registrados (¿todos atrapados o max_steps muy bajo?).")


# -------------------------------------------------------
# Escenario: BLOQUEO
# -------------------------------------------------------
elif escenario == "Bloqueo":
    st.subheader("Escenario: Bloqueo de una salida")
    num_exits = st.number_input("Número de salidas", min_value=1, max_value=12, value=3, step=1)
    t_bloqueo = st.number_input("Tiempo de bloqueo (s)", min_value=0.0, max_value=10_000.0, value=60.0, step=5.0)
    exit_index = st.number_input("Índice de salida a bloquear (0..n-1)", min_value=0, max_value=max(0, num_exits-1), value=0, step=1)

    run = st.button("🚧 Ejecutar bloqueo", type="primary")
    if run:
        with st.spinner("Simulando bloqueo..."):
            df, ts, perc, metrics = run_bloqueo_cached(
                N=agents, width=width, height=height, num_exits=num_exits,
                seed=seed, t_bloqueo=t_bloqueo, exit_index=exit_index, max_steps=max_steps
            )

        mdf = pd.DataFrame([metrics])
        st.success("✅ Simulación completada.")
        st.markdown("### Métricas")
        st.dataframe(mdf, use_container_width=True)
        _download_df_button(mdf, f"bloqueo_e{exit_index}_t{int(t_bloqueo)}_metrics.csv", "Descargar métricas (CSV)")

        st.markdown("### Curva de evacuación")
        _plot_line(ts, perc, f"Bloqueo — salida {exit_index} a {t_bloqueo}s")

        st.markdown("### Tiempos individuales")
        if df is not None and not df.empty:
            st.dataframe(df.head(500), use_container_width=True)
            _download_df_button(df, f"bloqueo_e{exit_index}_t{int(t_bloqueo)}_times.csv", "Descargar tiempos (CSV)")
        else:
            st.info("No hay tiempos registrados.")


# -------------------------------------------------------
# Escenario: ANCHOS (proxy)
# -------------------------------------------------------
elif escenario == "Anchos (proxy)":
    st.subheader("Escenario: Comparación de 'anchos' (proxy = número de salidas)")
    anchos_input = st.text_input("Lista de 'anchos' (proxy = número de salidas, separados por coma)", "1,2,3")
    try:
        lista_anchos = [int(x.strip()) for x in anchos_input.split(",") if x.strip()]
        lista_anchos = [x for x in lista_anchos if x >= 1]
    except Exception:
        st.error("Formato inválido. Usa números enteros separados por coma, p. ej.: 1,2,3")
        st.stop()

    run = st.button("📈 Ejecutar comparación", type="primary")
    if run:
        with st.spinner("Simulando anchos (proxy)..."):
            resultados = run_anchos_cached(
                N=agents, width=width, height=height, lista_anchos=tuple(lista_anchos),
                seed=seed, max_steps=max_steps
            )

        # Curvas comparadas
        series, rows = [], []
        for a, df, ts, perc, met in resultados:
            series.append((f"ancho{a}", ts, perc))
            rows.append({**met, "ancho_proxy": a})
        st.success("✅ Simulación completada.")

        st.markdown("### Curvas comparadas")
        fig, ax = plt.subplots()
        for label, ts, perc in series:
            ax.plot(ts, perc, label=label)
        ax.set_xlabel("Tiempo (s)")
        ax.set_ylabel("% evacuado")
        ax.set_title("Curvas por 'ancho' (proxy)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig, clear_figure=True)

        # Tabla de métricas
        resumo = pd.DataFrame(rows).sort_values("ancho_proxy")
        st.markdown("### Métricas por 'ancho'")
        st.dataframe(resumo, use_container_width=True)
        _download_df_button(resumo, "anchos_metrics.csv", "Descargar métricas (CSV)")

        # Descarga de tiempos individuales por ancho (ZIP simple en memoria)
        import zipfile
        mem_zip = io.BytesIO()
        with zipfile.ZipFile(mem_zip, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for a, df, _, _, _ in resultados:
                csv_bytes = df.to_csv(index=False).encode("utf-8")
                zf.writestr(f"anchos_{a}_times.csv", csv_bytes)
        st.download_button(
            "Descargar tiempos por ancho (ZIP)",
            data=mem_zip.getvalue(),
            file_name="anchos_times.zip",
            mime="application/zip",
        )

# -------------------------------------------------------
# Pie
# -------------------------------------------------------
st.markdown("---")
st.caption("Hecho con ❤️ usando Streamlit + Mesa. Estructura portable (sin paths absolutos).")
