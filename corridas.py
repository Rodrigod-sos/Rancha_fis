# -*- coding: utf-8 -*-
"""
Corridas del programa: 4 casos de estudio para el Sistema de Inferencia Difusa
de alerta temprana de rancha en papa (modulo rancha_fis.py).

Para cada caso se muestra: entradas, RC*, RF*, nivel, recomendacion MIP,
reglas activas y un grafico de la agregacion de Mamdani con su centroide.
"""

import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz

from rancha_fis import (evaluar, reglas_activas, FAM_1, FAM_2, T, HR, S, RC_out,
                        RC_in, RF, U_T, U_HR, U_RC, U_S, U_RF)

# ----------------------------------------------------------------------
# 1. CASOS DE ESTUDIO (T en °C; HR y S en %)
# ----------------------------------------------------------------------
CASOS = [
    {"n": 1, "nombre": "Época seca sin síntomas",
     "T": 22, "HR": 35, "S": 2},
    {"n": 2, "nombre": "Condiciones intermedias con membresías solapadas",
     "T": 19, "HR": 70, "S": 15},
    {"n": 3, "nombre": "Campaña húmeda con lesiones avanzadas",
     "T": 17, "HR": 93, "S": 55},
    {"n": 4, "nombre": "Caso atípico: síntomas severos con clima desfavorable",
     "T": 4, "HR": 45, "S": 60},
]


# ----------------------------------------------------------------------
# 2. AGREGACION DE MAMDANI (implicacion = min, agregacion = max)
# ----------------------------------------------------------------------
def agregada(fam, ant1, ant2, u1, u2, u_sal, sal, x1, x2):
    """Devuelve el conjunto agregado y los recortes de las reglas activas."""
    agg = np.zeros_like(u_sal, dtype=float)
    recortes = []
    for (e1, e2), cons in fam.items():
        alfa = min(fuzz.interp_membership(u1, ant1[e1].mf, x1),
                   fuzz.interp_membership(u2, ant2[e2].mf, x2))
        if alfa > 0:
            recorte = np.fmin(alfa, sal[cons].mf)
            agg = np.fmax(agg, recorte)
            recortes.append((cons, alfa))
    return agg, recortes


# ----------------------------------------------------------------------
# 3. GRAFICO DE UNA CORRIDA
# ----------------------------------------------------------------------
def graficar_corrida(caso, res, ruta):
    agg1, _ = agregada(FAM_1, T, HR, U_T, U_HR, U_RC, RC_out, caso["T"], caso["HR"])
    agg2, _ = agregada(FAM_2, RC_in, S, U_RC, U_S, U_RF, RF, res["RC"], caso["S"])
    c1 = fuzz.defuzz(U_RC, agg1, "centroid")
    c2 = fuzz.defuzz(U_RF, agg2, "centroid")

    fig, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, var, u, agg, c, tit, ej in [
        (ejes[0], RC_out, U_RC, agg1, res["RC"], "FIS-1: agregación y centroide", "RC (%)"),
        (ejes[1], RF, U_RF, agg2, res["RF"], "FIS-2: agregación y centroide", "RF (%)")]:
        for et, term in var.terms.items():
            ax.plot(u, term.mf, "--", linewidth=1, alpha=0.6, label=et)
        ax.fill_between(u, agg, color="tab:red", alpha=0.35, label="Agregado")
        ax.axvline(c, color="black", linewidth=1.8)
        ax.text(c, 1.1, f" {c:.2f} %", fontsize=10, fontweight="bold")
        ax.set_title(tit, fontsize=11)
        ax.set_xlabel(ej)
        ax.set_ylabel("Membresía")
        ax.set_ylim(0, 1.5)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper center", ncol=5)
    fig.suptitle(f"Corrida {caso['n']}: T = {caso['T']} °C, HR = {caso['HR']} %, "
                 f"S = {caso['S']} %", fontsize=12)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return c1, c2


# ----------------------------------------------------------------------
# 4. EJECUCION DE UNA CORRIDA
# ----------------------------------------------------------------------
def ejecutar_corrida(caso):
    res = evaluar(caso["T"], caso["HR"], caso["S"])
    print(f"=== Corrida {caso['n']}: {caso['nombre']} ===")
    print(f"Entradas : T = {caso['T']} °C | HR = {caso['HR']} % | S = {caso['S']} %")
    print(f"FIS-1    : RC* = {res['RC']:.2f} %")
    print(f"FIS-2    : RF* = {res['RF']:.2f} %  ->  {res['nivel']}")
    print(f"MIP      : {res['MIP']}")
    print("Reglas activas (fuerza de disparo):")
    for cod, texto, alfa in reglas_activas(caso["T"], caso["HR"], caso["S"]):
        print(f"  {cod}: {texto}  (alfa = {alfa:.3f})")
    c1, c2 = graficar_corrida(caso, res, f"corrida_{caso['n']}.png")
    print(f"Control  : centroide manual RC = {c1:.3f} | RF = {c2:.3f}\n")
    return res


# ----------------------------------------------------------------------
# 5. PROGRAMA PRINCIPAL: tabla resumen de las 4 corridas
# ----------------------------------------------------------------------
if __name__ == "__main__":
    resultados = [ejecutar_corrida(c) for c in CASOS]
    print("Resumen")
    print(f"{'N':<3}{'T':>4}{'HR':>5}{'S':>4}{'RC*':>8}{'RF*':>8}  Nivel")
    for c, r in zip(CASOS, resultados):
        print(f"{c['n']:<3}{c['T']:>4}{c['HR']:>5}{c['S']:>4}"
              f"{r['RC']:>8.2f}{r['RF']:>8.2f}  {r['nivel']}")
