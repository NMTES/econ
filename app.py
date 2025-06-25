import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm

st.set_page_config(page_title="Trabajo Final - Macroeconomía II", layout="wide")

# --- Cargar datasets ---
df_turismo = pd.read_csv("streamlit_data/turismo_tcr.csv")
df_indices = pd.read_csv("streamlit_data/indices_piezas_consumo.csv")
df_emae = pd.read_csv("streamlit_data/emae_mensual.csv")

# --- TITULO PRINCIPAL ---
st.title("Trabajo Final - Macroeconomía II")
st.markdown("---")

# --- Sección: TCR y saldo turístico ---
st.header("1. Tipo de cambio real y saldo turístico")

# Limpiar datos turismo
df_turismo.dropna(subset=["TCR_indice", "Saldo"], inplace=True)
X = df_turismo[["TCR_indice"]].values
y = df_turismo["Saldo"].values

model_turismo = LinearRegression().fit(X, y)
y_pred = model_turismo.predict(X)

coef = model_turismo.coef_[0]
intercepto = model_turismo.intercept_
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.scatter(X, y, label="Datos", alpha=0.6)
ax1.plot(X, y_pred, color="black", label="Regresión")
ax1.set_xlabel("TCR Bilateral (Indice)")
ax1.set_ylabel("Saldo turístico (miles de personas)")
ax1.set_title("Regresión: Saldo turístico vs TCR bilateral")
ax1.grid(True)
st.pyplot(fig1)

st.markdown(f"""
**Regresión lineal:**  
Coeficiente: {coef:.2f}  
Intercepto: {intercepto:.2f}  
$R^2$: {r2:.3f}  
RMSE: {rmse:.2f}
""")

# --- Sección: Piezas y consumo vs EMAE ---
st.header("2. Componentes de importación y actividad económica")

# Merge piezas-consumo y emae
merged = pd.merge(df_indices, df_emae, on="Fecha", how="inner")

# Correlaciones
corr_piezas = merged[["Var_Piezas_Desest", "Var_EMAE"]].corr().iloc[0, 1]
corr_consumo = merged[["Var_Consumo_Desest", "Var_EMAE"]].corr().iloc[0, 1]

fig2, ax2 = plt.subplots(figsize=(12, 5))
ax2.plot(merged["Fecha"], merged["Var_EMAE"], label="EMAE", color="black")
ax2.plot(merged["Fecha"], merged["Var_Piezas_Desest"], label="Piezas", alpha=0.8)
ax2.plot(merged["Fecha"], merged["Var_Consumo_Desest"], label="Consumo", alpha=0.8)
ax2.axhline(0, color="gray", linestyle="--")
ax2.set_ylabel("Variación mensual (%)")
ax2.set_title("Variaciones desestacionalizadas")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

st.markdown(f"""
**Correlaciones mensuales:**  
- Piezas vs EMAE: {corr_piezas:.3f}  
- Consumo vs EMAE: {corr_consumo:.3f}
""")

# --- Sección: Regresiones OLS ---
st.header("3. Regresiones lineales: EMAE y componentes de importación")

# OLS con statsmodels
clean = merged.dropna(subset=["Var_EMAE", "Var_Piezas_Desest", "Var_Consumo_Desest"])
X_sm = sm.add_constant(clean["Var_EMAE"])

model_pz = sm.OLS(clean["Var_Piezas_Desest"], X_sm).fit()
model_cs = sm.OLS(clean["Var_Consumo_Desest"], X_sm).fit()

st.subheader("Regresión: Piezas ~ EMAE")
st.text(model_pz.summary())

st.subheader("Regresión: Consumo ~ EMAE")
st.text(model_cs.summary())

fig3, ax3 = plt.subplots(figsize=(6, 6))
ax3.scatter(clean["Var_EMAE"], clean["Var_Piezas_Desest"], alpha=0.6, label="Datos")
x_vals = np.linspace(clean["Var_EMAE"].min(), clean["Var_EMAE"].max(), 100)
y_vals = model_pz.params["const"] + model_pz.params["Var_EMAE"] * x_vals
ax3.plot(x_vals, y_vals, color="black", label="Recta de regresión")
ax3.set_xlabel("EMAE Δ%")
ax3.set_ylabel("Piezas Δ%")
ax3.grid(True)
ax3.set_title("Piezas vs EMAE")
st.pyplot(fig3)

# Footer
st.markdown("---")
st.caption("Fuente: Elaboración propia en base a datos de INDEC y BCRA")
