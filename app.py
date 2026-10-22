# -*- coding: utf-8 -*-
"""
Frontend (Streamlit) del Sistema de Inferencia Difusa en cascada para la
alerta temprana de rancha en papa.

Ejecutar:  streamlit run app.py
Requiere rancha_fis.py en la misma carpeta (backend sin modificar).
"""

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import skfuzzy as fuzz

from rancha_fis import (evaluar, reglas_activas, superficie, sistema_1, sistema_2,
                        FAM_1, FAM_2, T, HR, S, RC_out, RC_in, RF,
                        U_T, U_HR, U_RC, U_S, U_RF)

# ----------------------------------------------------------------------
# CONFIGURACION
# ----------------------------------------------------------------------
st.set_page_config(page_title="Alerta de Rancha en Papa", page_icon="🥔", layout="wide")

NIVELES = {
    "SinRiesgo": ("Sin Riesgo", "#2E7D32"),
    "Leve": ("Leve", "#F9A825"),
    "Moderado": ("Moderado", "#EF6C00"),
    "Critico": ("Crítico", "#C62828"),
}

PRESETS = {
    "Personalizado": None,
    "Ejemplo 4.8 · Alerta preventiva (14 °C, 85 %, 0 %)": (14.0, 85.0, 0.0),
    "Corrida 1 · Época seca sin síntomas (22 °C, 35 %, 2 %)": (22.0, 35.0, 2.0),
    "Corrida 2 · Condiciones intermedias (19 °C, 70 %, 15 %)": (19.0, 70.0, 15.0),
    "Corrida 3 · Campaña húmeda, lesiones avanzadas (17 °C, 93 %, 55 %)": (17.0, 93.0, 55.0),
    "Corrida 4 · Caso atípico, posible helada (4 °C, 45 %, 60 %)": (4.0, 45.0, 60.0),
}

# Valores iniciales de los controles
for clave, valor in (("t", 14.0), ("hr", 85.0), ("s", 0.0), ("preset", "Personalizado")):
    st.session_state.setdefault(clave, valor)


def aplicar_preset():
    p = PRESETS.get(st.session_state.preset)
    if p:
        st.session_state.t, st.session_state.hr, st.session_state.s = p


def marcar_personalizado():
    st.session_state.preset = "Personalizado"


# ----------------------------------------------------------------------
# FUNCIONES AUXILIARES
# ----------------------------------------------------------------------
def agregada(fam, ant1, ant2, u1, u2, u_sal, sal, x1, x2):
    """Agregacion de Mamdani (implicacion = min, agregacion = max)."""
    agg = np.zeros_like(u_sal, dtype=float)
    for (e1, e2), cons in fam.items():
        alfa = min(fuzz.interp_membership(u1, ant1[e1].mf, x1),
                   fuzz.interp_membership(u2, ant2[e2].mf, x2))
        if alfa > 0:
            agg = np.fmax(agg, np.fmin(alfa, sal[cons].mf))
    return agg


def fig_membresias(variable, titulo, valor):
    fig, ax = plt.subplots(figsize=(5, 3))
    for et, term in variable.terms.items():
        ax.plot(variable.universe, term.mf, linewidth=2, label=et)
    ax.axvline(valor, color="black", linestyle="--", linewidth=1.5)
    ax.text(valor, 1.12, f" {valor:.2f}", fontsize=9, fontweight="bold")
    ax.set_title(titulo, fontsize=11)
    ax.set_ylim(-0.03, 1.3)
    ax.set_ylabel("Membresía")
    ax.legend(fontsize=8, loc="upper center", ncol=len(variable.terms))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def fig_agregacion(variable, universo, agg, centroide, titulo, xlabel):
    fig, ax = plt.subplots(figsize=(6, 3.4))
    for et, term in variable.terms.items():
        ax.plot(universo, term.mf, "--", linewidth=1, alpha=0.6, label=et)
    ax.fill_between(universo, agg, color="tab:red", alpha=0.35, label="Agregado")
    ax.axvline(centroide, color="black", linewidth=1.8)
    ax.text(centroide, 1.1, f" {centroide:.2f} %", fontsize=10, fontweight="bold")
    ax.set_title(titulo, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Membresía")
    ax.set_ylim(0, 1.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper center", ncol=5)
    fig.tight_layout()
    return fig


@st.cache_data(show_spinner="Calculando superficies de control (solo la primera vez)…")
def calcular_superficies():
    s1 = superficie(sistema_1, "T", "HR", "RC", U_T, U_HR)
    s2 = superficie(sistema_2, "RC_in", "S", "RF", U_RC, U_S)
    return s1, s2


def fig_superficie(xs, ys, Z, lx, ly, lz, punto):
    X, Y = np.meshgrid(xs, ys)
    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    sup = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.92)
    ax.scatter(*punto, color="red", s=90, depthshade=False, label="Entrada actual")
    ax.set_xlabel(lx)
    ax.set_ylabel(ly)
    ax.set_zlabel(lz)
    ax.view_init(elev=30, azim=-130)
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(sup, shrink=0.6, aspect=12)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# BARRA LATERAL: ENTRADAS
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("Entradas")
    st.selectbox("Escenario de ejemplo", list(PRESETS.keys()),
                 key="preset", on_change=aplicar_preset)
    st.slider("Temperatura promedio T (°C)", 0.0, 30.0, step=0.5,
              key="t", on_change=marcar_personalizado)
    st.slider("Humedad relativa HR (%)", 0.0, 100.0, step=1.0,
              key="hr", on_change=marcar_personalizado)
    st.slider("Severidad sintomática S (%)", 0.0, 100.0, step=1.0,
              key="s", on_change=marcar_personalizado,
              help="Porcentaje de área foliar con lesiones (escala visual de campo).")
    st.divider()
    st.caption("Curso: Sistemas Inteligentes · UNMSM · Grupo N.º 10")

t, hr, s = st.session_state.t, st.session_state.hr, st.session_state.s

# ----------------------------------------------------------------------
# INFERENCIA
# ----------------------------------------------------------------------
st.title("🥔 Alerta temprana de rancha en papa")
st.caption("Sistema de Inferencia Difusa Mamdani en cascada (FIS-1 → FIS-2) · "
           "Phytophthora infestans · XAI / ODS 15")

try:
    r = evaluar(temperatura=t, humedad=hr, severidad=s)
    reglas = reglas_activas(t, hr, s)
except Exception as e:  # p. ej. ninguna regla activa
    st.error(f"No se pudo calcular la inferencia: {e}")
    st.stop()

nombre_nivel, color = NIVELES[r["nivel"]]

# ----------------------------------------------------------------------
# PESTAÑAS
# ----------------------------------------------------------------------
tab_res, tab_agg, tab_mf, tab_sup, tab_info = st.tabs(
    ["📊 Resultado", "🧮 Agregación Mamdani", "📈 Membresías", "🗺️ Superficies", "ℹ️ Acerca de"])

# ---- Resultado -------------------------------------------------------
with tab_res:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("FIS-1 · Riesgo climático RC*", f"{r['RC']:.2f} %")
        st.progress(min(max(r["RC"] / 100, 0.0), 1.0))
    with c2:
        st.metric("FIS-2 · Riesgo fitosanitario RF*", f"{r['RF']:.2f} %")
        st.progress(min(max(r["RF"] / 100, 0.0), 1.0))
    with c3:
        st.markdown("**Nivel de alerta**")
        st.markdown(
            f"<div style='background:{color};color:white;padding:10px 16px;"
            f"border-radius:8px;font-size:1.4rem;font-weight:700;text-align:center'>"
            f"{nombre_nivel}</div>", unsafe_allow_html=True)

    st.subheader("Recomendación MIP")
    st.info(r["MIP"])
    st.caption("Recomendaciones orientativas: deben validarse con un especialista y "
               "con la normativa vigente de SENASA.")

    st.subheader("Reglas activas (explicación XAI)")
    filas = [{"Regla": cod,
              "Módulo": "FIS-1" if int(cod[1:]) <= 9 else "FIS-2",
              "Condición → Conclusión": texto,
              "Fuerza α": alpha} for cod, texto, alpha in reglas]
    st.dataframe(
        pd.DataFrame(filas), hide_index=True, width="stretch",
        column_config={"Fuerza α": st.column_config.ProgressColumn(
            "Fuerza α", min_value=0.0, max_value=1.0, format="%.3f")})

# ---- Agregación ------------------------------------------------------
with tab_agg:
    agg1 = agregada(FAM_1, T, HR, U_T, U_HR, U_RC, RC_out, t, hr)
    agg2 = agregada(FAM_2, RC_in, S, U_RC, U_S, U_RF, RF, r["RC"], s)
    a, b = st.columns(2)
    with a:
        st.pyplot(fig_agregacion(RC_out, U_RC, agg1, r["RC"],
                                 "FIS-1: agregación y centroide", "RC (%)"))
    with b:
        st.pyplot(fig_agregacion(RF, U_RF, agg2, r["RF"],
                                 "FIS-2: agregación y centroide", "RF (%)"))
    st.caption("Línea punteada: conjuntos de salida · Área roja: conjunto agregado "
               "(implicación = min, agregación = max) · Línea negra: centroide.")

# ---- Membresías ------------------------------------------------------
with tab_mf:
    st.caption("La línea punteada marca el valor actual de cada variable.")
    fila1 = st.columns(3)
    fila2 = st.columns(2)
    fila1[0].pyplot(fig_membresias(T, "Temperatura T (°C)", t))
    fila1[1].pyplot(fig_membresias(HR, "Humedad relativa HR (%)", hr))
    fila1[2].pyplot(fig_membresias(RC_out, "Riesgo climático RC (%)", r["RC"]))
    fila2[0].pyplot(fig_membresias(S, "Severidad sintomática S (%)", s))
    fila2[1].pyplot(fig_membresias(RF, "Riesgo fitosanitario RF (%)", r["RF"]))

# ---- Superficies -----------------------------------------------------
with tab_sup:
    if st.toggle("Calcular y mostrar superficies de control 3D",
                 help="La primera vez tarda unos segundos; luego queda en caché."):
        (xs1, ys1, Z1), (xs2, ys2, Z2) = calcular_superficies()
        a, b = st.columns(2)
        with a:
            st.pyplot(fig_superficie(xs1, ys1, Z1, "Temperatura T (°C)",
                                     "Humedad relativa HR (%)", "RC* (%)",
                                     (t, hr, r["RC"])))
            st.caption("FIS-1: RC* en función de T y HR.")
        with b:
            st.pyplot(fig_superficie(xs2, ys2, Z2, "Riesgo climático RC (%)",
                                     "Severidad S (%)", "RF* (%)",
                                     (r["RC"], s, r["RF"])))
            st.caption("FIS-2: RF* en función de RC y S.")
        st.caption("El punto rojo es la entrada actual. Las superficies son continuas: "
                   "no hay efecto acantilado.")
    else:
        st.info("Activa el interruptor para generar las superficies.")

# ---- Acerca de -------------------------------------------------------
with tab_info:
    st.markdown("""
**Arquitectura en cascada**

- **FIS-1:** (T, HR) → RC\\* — riesgo climático de infección (9 reglas).
- **FIS-2:** (RC\\*, S) → RF\\* — riesgo fitosanitario final y recomendación MIP (9 reglas).

**Operadores:** AND = mínimo · implicación = mínimo · agregación = máximo ·
defuzzificación por centroide.

**Interpretabilidad (XAI):** cada decisión se traza hasta reglas en lenguaje natural,
auditables por un agrónomo.

Los valores de los escenarios de ejemplo son ilustrativos; para un diagnóstico real
se deben usar lecturas de estación meteorológica y estimación visual de S en campo.
""")
    st.caption("Alumnos: Carrasco Alcantara, Maria Alejandra · Contreras Peña, Rodrigo Eduardo · "
               "Rojas Suarez, Ana Cristina")