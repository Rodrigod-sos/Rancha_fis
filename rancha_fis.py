# -*- coding: utf-8 -*-
"""
Sistema de Inferencia Difusa (Mamdani) en cascada para la alerta temprana
de rancha (Phytophthora infestans) en el cultivo de papa.

    FIS-1:  (T, HR)  ->  RC*     Riesgo climatico de infeccion
    FIS-2:  (RC*, S) ->  RF*     Riesgo fitosanitario final (recomendacion MIP)

Curso: Sistemas Inteligentes - UNMSM - Grupo N.º 10
Dependencias: numpy, matplotlib, scikit-fuzzy  (pip install scikit-fuzzy)
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")  # permite guardar figuras sin ventana grafica
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# ----------------------------------------------------------------------
# 1. UNIVERSOS DE DISCURSO (seccion 2.1.1 y 2.2.1)
# ----------------------------------------------------------------------
U_T = np.arange(0, 30.5, 0.5)    # Temperatura promedio  [0, 30] °C
U_HR = np.arange(0, 100.5, 0.5)  # Humedad relativa      [0, 100] %
U_RC = np.arange(0, 100.5, 0.5)  # Riesgo climatico      [0, 100] %
U_S = np.arange(0, 100.5, 0.5)   # Severidad sintomatica [0, 100] %
U_RF = np.arange(0, 100.5, 0.5)  # Riesgo fitosanitario  [0, 100] %

# ----------------------------------------------------------------------
# 2. PARAMETROS DE LAS FUNCIONES DE MEMBRESIA (tabla de la seccion 2.1.2)
#    ("trap" = trapmf [a, b, c, d]; "tri" = trimf [a, b, c])
# ----------------------------------------------------------------------
MF_T = {"Fria": ("trap", [0, 0, 8, 12]),
        "Optima": ("trap", [8, 12, 18, 22]),
        "Calida": ("trap", [18, 22, 30, 30])}

MF_HR = {"Baja": ("trap", [0, 0, 40, 60]),
         "Media": ("tri", [40, 60, 80]),
         "Alta": ("trap", [60, 80, 100, 100])}

MF_RC = {"Bajo": ("trap", [0, 0, 20, 45]),
         "Medio": ("tri", [25, 50, 75]),
         "Alto": ("trap", [55, 80, 100, 100])}

# "Nula" corresponde a la etiqueta "Nula/Incipiente" del informe
MF_S = {"Nula": ("trap", [0, 0, 10, 30]),
        "Moderada": ("tri", [10, 35, 60]),
        "Severa": ("trap", [40, 60, 100, 100])}

MF_RF = {"SinRiesgo": ("trap", [0, 0, 10, 25]),
         "Leve": ("tri", [10, 30, 50]),
         "Moderado": ("tri", [40, 60, 80]),
         "Critico": ("trap", [65, 85, 100, 100])}


def asignar_mf(variable, definiciones):
    """Asigna a una variable linguistica sus funciones de membresia."""
    for etiqueta, (tipo, params) in definiciones.items():
        if tipo == "trap":
            variable[etiqueta] = fuzz.trapmf(variable.universe, params)
        else:
            variable[etiqueta] = fuzz.trimf(variable.universe, params)


# ----------------------------------------------------------------------
# 3. VARIABLES LINGUISTICAS
# ----------------------------------------------------------------------
T = ctrl.Antecedent(U_T, "T")
HR = ctrl.Antecedent(U_HR, "HR")
RC_out = ctrl.Consequent(U_RC, "RC")      # salida de FIS-1 (defuzzificacion: centroide)
RC_in = ctrl.Antecedent(U_RC, "RC_in")    # entrada de FIS-2 (mismos conjuntos que RC)
S = ctrl.Antecedent(U_S, "S")
RF = ctrl.Consequent(U_RF, "RF")          # salida de FIS-2 (defuzzificacion: centroide)

for var, mfs in [(T, MF_T), (HR, MF_HR), (RC_out, MF_RC),
                 (RC_in, MF_RC), (S, MF_S), (RF, MF_RF)]:
    asignar_mf(var, mfs)

# ----------------------------------------------------------------------
# 4. BASES DE REGLAS (secciones 3.1 y 3.2) - todas con peso 1 y AND = min
# ----------------------------------------------------------------------
# FIS-1: matriz FAM  T x HR -> RC
FAM_1 = {("Fria", "Baja"): "Bajo",   ("Fria", "Media"): "Bajo",   ("Fria", "Alta"): "Medio",
         ("Optima", "Baja"): "Bajo", ("Optima", "Media"): "Medio", ("Optima", "Alta"): "Alto",
         ("Calida", "Baja"): "Bajo", ("Calida", "Media"): "Bajo", ("Calida", "Alta"): "Medio"}

# FIS-2: matriz FAM  RC x S -> RF
FAM_2 = {("Bajo", "Nula"): "SinRiesgo",  ("Bajo", "Moderada"): "Leve",
         ("Bajo", "Severa"): "Moderado",
         ("Medio", "Nula"): "Leve",      ("Medio", "Moderada"): "Moderado",
         ("Medio", "Severa"): "Critico",
         ("Alto", "Nula"): "Moderado",   ("Alto", "Moderada"): "Critico",
         ("Alto", "Severa"): "Critico"}

reglas_1 = [ctrl.Rule(T[t] & HR[h], RC_out[rc]) for (t, h), rc in FAM_1.items()]
reglas_2 = [ctrl.Rule(RC_in[rc] & S[s], RF[rf]) for (rc, s), rf in FAM_2.items()]

# ----------------------------------------------------------------------
# 5. SISTEMAS DE CONTROL (Mamdani: implicacion = min, agregacion = max,
#    defuzzificacion = centroide, que es el valor por defecto de skfuzzy)
# ----------------------------------------------------------------------
sistema_1 = ctrl.ControlSystem(reglas_1)
sistema_2 = ctrl.ControlSystem(reglas_2)

# ----------------------------------------------------------------------
# 6. RECOMENDACIONES MIP ORIENTATIVAS (seccion 3.3)
# ----------------------------------------------------------------------
MIP = {
    "SinRiesgo": "Monitoreo rutinario del cultivo.",
    "Leve": "Vigilancia intensificada, eliminacion de plantas voluntarias y focos, "
            "verificacion del sintoma (enves de la hoja, humedad de la manana).",
    "Moderado": "Aplicacion preventiva de fungicida protectante segun etiqueta y "
                "asesoria tecnica, y reevaluacion en 48-72 h.",
    "Critico": "Intervencion inmediata con rotacion de modos de accion, remocion de "
               "focos, evaluacion de cosecha anticipada y reporte a la autoridad "
               "fitosanitaria.",
}


def nivel_linguistico(rf):
    """Etiqueta de RF con mayor grado de pertenencia para el valor defuzzificado."""
    grados = {et: fuzz.interp_membership(U_RF, RF[et].mf, rf) for et in MF_RF}
    return max(grados, key=grados.get)


# ----------------------------------------------------------------------
# 7. FUNCION DE INFERENCIA EN CASCADA   RC* = f1(T, HR)  ->  RF* = f2(RC*, S)
# ----------------------------------------------------------------------
def evaluar(temperatura, humedad, severidad):
    """Ejecuta FIS-1 y FIS-2 en cascada y devuelve un diccionario de resultados."""
    # --- FIS-1: alerta temprana agroclimatica ---
    sim1 = ctrl.ControlSystemSimulation(sistema_1)
    sim1.input["T"] = temperatura
    sim1.input["HR"] = humedad
    sim1.compute()
    rc = float(sim1.output["RC"])

    # --- FIS-2: RC* se vuelve a fuzzificar como entrada ---
    sim2 = ctrl.ControlSystemSimulation(sistema_2)
    sim2.input["RC_in"] = rc
    sim2.input["S"] = severidad
    sim2.compute()
    rf = float(sim2.output["RF"])

    nivel = nivel_linguistico(rf)
    return {"T": temperatura, "HR": humedad, "S": severidad,
            "RC": rc, "RF": rf, "nivel": nivel, "MIP": MIP[nivel]}


def reglas_activas(temperatura, humedad, severidad):
    """Fuerza de disparo (min) de las 18 reglas: explica la decision (XAI)."""
    rc = evaluar(temperatura, humedad, severidad)["RC"]
    activas = []
    for k, ((t, h), rc_et) in enumerate(FAM_1.items(), start=1):
        a = min(fuzz.interp_membership(U_T, T[t].mf, temperatura),
                fuzz.interp_membership(U_HR, HR[h].mf, humedad))
        if a > 0:
            activas.append((f"R{k}", f"T={t} Y HR={h} -> RC={rc_et}", round(a, 3)))
    for k, ((r, s), rf_et) in enumerate(FAM_2.items(), start=10):
        a = min(fuzz.interp_membership(U_RC, RC_in[r].mf, rc),
                fuzz.interp_membership(U_S, S[s].mf, severidad))
        if a > 0:
            activas.append((f"R{k}", f"RC={r} Y S={s} -> RF={rf_et}", round(a, 3)))
    return activas


# ----------------------------------------------------------------------
# 8. GRAFICOS: funciones de membresia y superficies de control
# ----------------------------------------------------------------------
def graficar_membresias(ruta="membresias.png"):
    variables = [(T, "Temperatura T (°C)"), (HR, "Humedad relativa HR (%)"),
                 (RC_out, "Riesgo climatico RC (%)"), (S, "Severidad sintomatica S (%)"),
                 (RF, "Riesgo fitosanitario RF (%)")]
    fig, ejes = plt.subplots(3, 2, figsize=(11, 10))
    ejes = ejes.ravel()
    for ax, (var, titulo) in zip(ejes, variables):
        for et, term in var.terms.items():
            ax.plot(var.universe, term.mf, linewidth=2, label=et)
        ax.set_title(titulo, fontsize=11)
        ax.set_ylim(-0.03, 1.3)
        ax.set_ylabel("Membresia")
        ax.legend(fontsize=8, loc="upper center", ncol=len(var.terms))
        ax.grid(alpha=0.3)
    ejes[-1].axis("off")
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)


def superficie(sistema, ent_x, ent_y, salida, ux, uy, paso=2):
    """Malla de salida del sistema sobre dos entradas."""
    xs, ys = np.arange(ux[0], ux[-1] + 0.1, paso), np.arange(uy[0], uy[-1] + 0.1, paso)
    Z = np.zeros((len(ys), len(xs)))
    sim = ctrl.ControlSystemSimulation(sistema)
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            sim.input[ent_x] = x
            sim.input[ent_y] = y
            sim.compute()
            Z[i, j] = sim.output[salida]
    return xs, ys, Z


def graficar_superficies(ruta1="superficie_fis1.png", ruta2="superficie_fis2.png"):
    for ruta, (sist, ex, ey, sal, ux, uy, lx, ly, lz) in {
        ruta1: (sistema_1, "T", "HR", "RC", U_T, U_HR,
                "Temperatura T (°C)", "Humedad relativa HR (%)", "Riesgo climatico RC* (%)"),
        ruta2: (sistema_2, "RC_in", "S", "RF", U_RC, U_S,
                "Riesgo climatico RC (%)", "Severidad S (%)", "Riesgo fitosanitario RF* (%)"),
    }.items():
        xs, ys, Z = superficie(sist, ex, ey, sal, ux, uy)
        X, Y = np.meshgrid(xs, ys)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")
        sup = ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.95)
        ax.set_xlabel(lx)
        ax.set_ylabel(ly)
        ax.set_zlabel(lz)
        ax.view_init(elev=30, azim=-130)
        fig.colorbar(sup, shrink=0.6, aspect=12)
        fig.tight_layout()
        fig.savefig(ruta, dpi=150)
        plt.close(fig)


# ----------------------------------------------------------------------
# 9. PROGRAMA PRINCIPAL: interactivo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("--- SISTEMA DE ALERTA DE RANCHA ---")
    
    # El programa te pedirá los datos por consola
    t = float(input("Ingresa la temperatura promedio (°C): "))
    h = float(input("Ingresa la humedad relativa (%): "))
    s = float(input("Ingresa la severidad observada en hojas (%): "))
    
    # Evalúa los datos que acabas de ingresar
    r = evaluar(temperatura=t, humedad=h, severidad=s)
    
    print(f"\nEntradas : T = {r['T']} °C | HR = {r['HR']} % | S = {r['S']} %")
    print(f"FIS-1    : RC* = {r['RC']:.2f} %")
    print(f"FIS-2    : RF* = {r['RF']:.2f} %  ->  {r['nivel']}")
    print(f"MIP      : {r['MIP']}")
    
    print("\nReglas activas (fuerza de disparo):")
    for cod, texto, alpha in reglas_activas(t, h, s):
        print(f"  {cod}: {texto}  (alfa = {alpha})")

    # Mantenemos esto si quieres que siga generando las fotos de los gráficos
    graficar_membresias()
    graficar_superficies()
