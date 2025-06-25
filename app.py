import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="TCR y Turismo", layout="wide")

st.title("📊 Tipo de Cambio Real Bilateral y Saldo Turístico")

st.markdown("""
Esta app visualiza la evolución del **tipo de cambio real bilateral Argentina-Brasil** y su posible relación con el **saldo turístico**.
Los datos provienen de fuentes oficiales como INDEC, IBGE, entre otros.
""")

# --- URLs desde GitHub ---
BASE_URL = "https://raw.githubusercontent.com/NMTES/econ/main/streamlit_data/"

url_br = BASE_URL + "brasil.csv"
url_ar = BASE_URL + "serie_ipc_divisiones.csv"
url_blue = BASE_URL + "usd_ars_blue.csv"
url_brl = BASE_URL + "usd-brl.csv"
url_turismo = BASE_URL + "serie_turismo_receptivo_emisivo.xlsx"
url_importaciones = BASE_URL + "serie_mensual_indices_impo_ue.xls"
url_emae = BASE_URL + "sh_emae_mensual_base2004.xls"

try:
    # Inflación Brasil
    df_br = pd.read_csv(url_br, skiprows=3)
    df_brasil = df_br[df_br.iloc[:, 0] == "Brasil"]
    fechas_str = df_brasil.columns[1:]
    valores = df_brasil.values[0][1:]
    ipca_br = pd.DataFrame({"Fecha": fechas_str, "Infl_BR": valores})
    ipca_br['Infl_BR'] = ipca_br['Infl_BR'].astype(str).str.replace(',', '.').astype(float)
    meses = {'janeiro': '01', 'fevereiro': '02', 'março': '03', 'abril': '04',
             'maio': '05', 'junho': '06', 'julho': '07', 'agosto': '08',
             'setembro': '09', 'outubro': '10', 'novembro': '11', 'dezembro': '12'}
    ipca_br[['Mes', 'Año']] = ipca_br['Fecha'].str.extract(r'(\w+)\s+(\d{4})')
    ipca_br['Mes'] = ipca_br['Mes'].map(meses)
    ipca_br['Fecha'] = pd.to_datetime(ipca_br['Año'] + '-' + ipca_br['Mes'] + '-01')
    ipca_br = ipca_br[ipca_br['Fecha'] >= '2019-01-01'][['Fecha', 'Infl_BR']]

    # Inflación Argentina
    df_ar = pd.read_csv(url_ar, sep=';', encoding='latin1')
    ipc_ar = df_ar[(df_ar['Descripcion'] == 'NIVEL GENERAL') & (df_ar['Region'] == 'Nacional')].copy()
    ipc_ar['Fecha'] = pd.to_datetime(ipc_ar['Periodo'], format='%Y%m')
    ipc_ar['Infl_AR'] = ipc_ar['v_m_IPC'].str.replace(',', '.').astype(float)
    ipc_ar = ipc_ar[['Fecha', 'Infl_AR']]

    # Merge inflaciones
    df_comb = pd.merge(ipca_br, ipc_ar, on='Fecha', how='inner')

    # Blue
    df_ars = pd.read_csv(url_blue)
    df_ars['Fecha'] = pd.to_datetime(df_ars['category'])
    df_ars['valor'] = df_ars['valor'].astype(float)
    df_ars['Mes'] = df_ars['Fecha'].dt.to_period('M')
    blue_mensual = df_ars.sort_values('Fecha').groupby('Mes').tail(1).copy()
    blue_mensual['Fecha'] = blue_mensual['Mes'].dt.to_timestamp()
    blue_mensual = blue_mensual[['Fecha', 'valor']].rename(columns={'valor': 'ARS_USD_Blue'})

    # BRL
    df_brl = pd.read_csv(url_brl)
    df_brl['Fecha'] = pd.to_datetime(df_brl['Fecha'], format='%d.%m.%Y')
    df_brl['USD_BRL'] = df_brl['Último'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    df_brl = df_brl[['Fecha', 'USD_BRL']].sort_values('Fecha')

    # Tipo de cambio real
    df_tcr = df_comb.merge(blue_mensual, on='Fecha', how='inner').merge(df_brl, on='Fecha', how='inner')
    df_tcr['TC_nominal_ARS_BRL'] = df_tcr['ARS_USD_Blue'] / df_tcr['USD_BRL']
    df_tcr['TCR_bruto'] = df_tcr['TC_nominal_ARS_BRL'] * df_tcr['Infl_BR'] / df_tcr['Infl_AR']
    valor_base = df_tcr.loc[df_tcr['Fecha'] == '2019-01-01', 'TCR_bruto'].values[0]
    df_tcr['TCR_indice'] = df_tcr['TCR_bruto'] / valor_base * 100
    df_tcr['IPC_AR_idx'] = (1 + df_tcr['Infl_AR'] / 100).cumprod() * 100
    df_tcr['IPC_BR_idx'] = (1 + df_tcr['Infl_BR'] / 100).cumprod() * 100
    df_tcr['TCR'] = df_tcr['TC_nominal_ARS_BRL'] * df_tcr['IPC_BR_idx'] / df_tcr['IPC_AR_idx']
    tcr_base = df_tcr.loc[df_tcr['Fecha'] == '2019-01-01', 'TCR'].values[0]
    df_tcr['TCR_indice'] = df_tcr['TCR'] / tcr_base * 100

    # --- GRAFICO TCR ---
    st.subheader("📈 Índice del Tipo de Cambio Real Bilateral")
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df_tcr['Fecha'], df_tcr['TCR_indice'], label="TCR Bilateral", color="blue")
    ax1.set_title("Índice del Tipo de Cambio Real Bilateral (Argentina - Brasil)")
    ax1.set_ylabel("Índice (Base Ene-2019 = 100)")
    ax1.grid(True)
    fig1.tight_layout()
    st.pyplot(fig1)

    # --- TURISMO ---
    df = pd.read_excel(url_turismo, header=2, engine="openpyxl")
    df.columns = ["Año", "Fecha", "Receptivo", "Emisivo", "Saldo"]
    df = df[df["Fecha"].notna()].copy()
    df = df[df["Fecha"] >= "2019-01-01"]
    df["Fecha"] = df["Fecha"].dt.to_period("M").dt.to_timestamp()
    df["Saldo"] = pd.to_numeric(df["Saldo"], errors="coerce")
    df_tcr["Fecha"] = pd.to_datetime(df_tcr["Fecha"]).dt.to_period("M").dt.to_timestamp()

    df_completo = pd.merge(df_tcr[["Fecha", "TCR_indice"]], df[["Fecha", "Saldo"]], on="Fecha", how="inner").dropna()
    df_completo["FechaNum"] = mdates.date2num(df_completo["Fecha"])

    # Regresiones
    X = df_completo["FechaNum"].values.reshape(-1, 1)
    y_tcr = df_completo["TCR_indice"].values
    y_saldo = df_completo["Saldo"].values
    reg_tcr = LinearRegression().fit(X, y_tcr)
    reg_saldo = LinearRegression().fit(X, y_saldo)

    # GRAFICO TCR vs Turismo
    st.subheader("🌍 TCR vs Saldo Turístico")
    fig2, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_completo["Fecha"], df_completo["TCR_indice"], label="TCR Bilateral", color="blue")
    ax2 = ax.twinx()
    ax2.plot(df_completo["Fecha"], df_completo["Saldo"], label="Saldo Turístico", color="red", linestyle="--")
    ax.set_title("TCR Bilateral Argentina-Brasil vs Saldo Turístico")
    ax.set_ylabel("TCR (Índice Ene-2019 = 100)", color="blue")
    ax2.set_ylabel("Saldo Turístico (miles de personas)", color="red")
    ax.grid(True)
    fig2.tight_layout()
    st.pyplot(fig2)

    # Regresión entre TCR y saldo turístico
    X_tcr = df_completo[["TCR_indice"]].values
    y = df_completo["Saldo"].values
    model = LinearRegression()
    model.fit(X_tcr, y)
    y_pred = model.predict(X_tcr)
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))

    st.markdown("### 📉 Regresión: Saldo turístico vs TCR")
    st.write(f"**Coeficiente:** {model.coef_[0]:.4f}")
    st.write(f"**Intercepto:** {model.intercept_:.4f}")
    st.write(f"**R²:** {r2:.3f}")
    st.write(f"**RMSE:** {rmse:.2f}")

    fig3, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(X_tcr, y, label="Datos", alpha=0.6)
    ax.plot(X_tcr, y_pred, color="black", label="Regresión")
    ax.set_xlabel("TCR Bilateral (Índice)")
    ax.set_ylabel("Saldo turístico")
    ax.set_title("Regresión lineal")
    ax.grid(True)
    st.pyplot(fig3)

        # --- Leer archivos ---
    df_raw = pd.read_excel(url_importaciones, header=1, engine="xlrd")

    categorias = [
        "Nivel general", "Bienes de capital", "Bienes intermedios",
        "Combustibles y lubricantes", "Piezas y accesorios para bienes de capital",
        "Bienes de consumo", "Vehículos automotores de pasajeros"
    ]

    columnas = ["Año", "Mes"]
    for categoria in categorias:
        columnas += [f"{categoria} - Valor", f"{categoria} - Precio", f"{categoria} - Cantidad", f"{categoria} - Extra"]
    df_raw.columns = columnas[:len(df_raw.columns)]

    meses_validos = [
        "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
        "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
    ]
    df_raw["Mes"] = df_raw["Mes"].astype(str).str.strip().str.capitalize()
    df_raw = df_raw[df_raw["Mes"].isin(meses_validos)].copy()

    año_actual = None
    años = []
    for val in df_raw["Año"]:
        val_str = str(val).strip()
        posible_año = ''.join(filter(str.isdigit, val_str))
        if posible_año.isdigit():
            año_actual = int(posible_año)
        años.append(año_actual)
    df_raw["Año"] = años

    df_raw["Mes_num"] = df_raw["Mes"].apply(lambda x: meses_validos.index(x) + 1)
    df_raw["Fecha"] = pd.to_datetime(dict(year=df_raw["Año"], month=df_raw["Mes_num"], day=1))

    # --- Extraccion y desestacionalización ---
    df = pd.DataFrame()
    df["Fecha"] = df_raw["Fecha"]
    df["Piezas_Cantidad"] = pd.to_numeric(df_raw["Piezas y accesorios para bienes de capital - Cantidad"], errors="coerce")
    df["Consumo_Cantidad"] = pd.to_numeric(df_raw["Bienes de consumo - Cantidad"], errors="coerce")
    df["Año"] = df["Fecha"].dt.year
    df["Mes"] = df["Fecha"].dt.month

    prom_anual = df.groupby("Año")[["Piezas_Cantidad", "Consumo_Cantidad"]].transform("mean")
    df["Piezas_Proxy"] = df["Piezas_Cantidad"] / prom_anual["Piezas_Cantidad"]
    df["Consumo_Proxy"] = df["Consumo_Cantidad"] / prom_anual["Consumo_Cantidad"]

    coef = df.groupby("Mes")[["Piezas_Proxy", "Consumo_Proxy"]].mean().reset_index()
    df = df.merge(coef, on="Mes", suffixes=("", "_coef"))
    df["Piezas_Desest"] = df["Piezas_Cantidad"] / df["Piezas_Proxy_coef"]
    df["Consumo_Desest"] = df["Consumo_Cantidad"] / df["Consumo_Proxy_coef"]
    df["Var_Piezas_Desest"] = df["Piezas_Desest"].pct_change(fill_method=None) * 100
    df["Var_Consumo_Desest"] = df["Consumo_Desest"].pct_change(fill_method=None) * 100

    # --- Gráfico desestacionalizado ---
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(df["Fecha"], df["Piezas_Desest"], label="Piezas Desestacionalizadas")
    ax1.plot(df["Fecha"], df["Consumo_Desest"], label="Consumo Desestacionalizado")
    ax1.set_title("Índices Desestacionalizados (2004-2025)")
    ax1.grid(True)
    st.pyplot(fig1)

    # --- EMAE ---
    df_emae = pd.read_excel(url_emae, skiprows=3, header=None, engine="xlrd")
    df_emae.columns = ["Año", "Mes", "Serie_Original", "Var_anual", "EMAE_Desest", "Var_mensual_desest", "Tendencia_Ciclo", "Var_mensual_tendencia"]
    meses_dict = {m: i+1 for i, m in enumerate(meses_validos)}
    df_emae["Mes"] = df_emae["Mes"].astype(str).str.strip().str.capitalize()
    df_emae = df_emae[df_emae["Mes"].isin(meses_dict.keys())].copy()

    año_actual = None
    años = []
    for val in df_emae["Año"]:
        if pd.notna(val) and str(val).strip().isdigit():
            año_actual = int(val)
        años.append(año_actual)
    df_emae["Año"] = años
    df_emae["Mes_num"] = df_emae["Mes"].map(meses_dict)
    df_emae["Fecha"] = pd.to_datetime(dict(year=df_emae["Año"], month=df_emae["Mes_num"], day=1), errors='coerce')
    df_emae = df_emae[df_emae["Fecha"].notna()].sort_values("Fecha").reset_index(drop=True)
    df_merged = pd.merge(df, df_emae[["Fecha", "Var_mensual_desest"]], on="Fecha", how="inner")
    df_merged.rename(columns={"Var_mensual_desest": "Var_EMAE"}, inplace=True)
    corr_piezas = df_merged[["Var_Piezas_Desest", "Var_EMAE"]].corr().iloc[0, 1]
    corr_consumo = df_merged[["Var_Consumo_Desest", "Var_EMAE"]].corr().iloc[0, 1]
    
    # --- Gráfico de variaciones ---
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(df_merged["Fecha"], df_merged["Var_EMAE"], label="EMAE Δ%", color="black")
    ax2.plot(df_merged["Fecha"], df_merged["Var_Piezas_Desest"], label="Piezas Δ%")
    ax2.plot(df_merged["Fecha"], df_merged["Var_Consumo_Desest"], label="Consumo Δ%")
    ax2.set_title("Variaciones mensuales desestacionalizadas")
    ax2.axhline(0, color="gray")
    ax2.grid(True)
    st.pyplot(fig2)
    
    # --- Indicadores ---
    st.write(f"📈 Correlación mensual (Piezas vs EMAE): {corr_piezas:.3f}")
    st.write(f"📈 Correlación mensual (Consumo vs EMAE): {corr_consumo:.3f}")

    df_cleaned = df_merged.dropna(subset=["Var_EMAE", "Var_Piezas_Desest", "Var_Consumo_Desest"]).copy()

    # Regresión para Piezas
    X_piezas = df_cleaned[["Var_EMAE"]].values
    y_piezas = df_cleaned["Var_Piezas_Desest"].values
    modelo_piezas = LinearRegression().fit(X_piezas, y_piezas)

    coef_piezas = modelo_piezas.coef_[0]
    intercepto_piezas = modelo_piezas.intercept_

    # Regresión para Consumo
    X_consumo = df_cleaned[["Var_EMAE"]].values
    y_consumo = df_cleaned["Var_Consumo_Desest"].values
    modelo_consumo = LinearRegression().fit(X_consumo, y_consumo)

    coef_consumo = modelo_consumo.coef_[0]
    intercepto_consumo = modelo_consumo.intercept_

    # --- Gráfico de regresión (solo Piezas vs EMAE) ---
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    ax3.scatter(X_piezas, y_piezas, alpha=0.7, label="Datos")
    x_vals = np.linspace(X_piezas.min(), X_piezas.max(), 100)
    y_pred = coef_piezas * x_vals + intercepto_piezas
    ax3.plot(x_vals, y_pred, color="black", label="Recta de regresión")
    ax3.set_title("Regresión: Piezas Δ% vs EMAE Δ%")
    ax3.set_xlabel("EMAE Δ%")
    ax3.set_ylabel("Piezas Δ%")
    ax3.grid(True)
    st.pyplot(fig3)

    # --- Gráfico anual ---
    df_merged["Año"] = df_merged["Fecha"].dt.year
    df_anual = df_merged.groupby("Año")[["Var_EMAE", "Var_Piezas_Desest", "Var_Consumo_Desest"]].mean().reset_index()
    corr_piezas_anual = df_anual[["Var_Piezas_Desest", "Var_EMAE"]].corr().iloc[0, 1]
    corr_consumo_anual = df_anual[["Var_Consumo_Desest", "Var_EMAE"]].corr().iloc[0, 1]
    fig4, ax4 = plt.subplots(figsize=(8, 4))
    ax4.plot(df_anual["Año"], df_anual["Var_EMAE"], label="EMAE Δ% anual", color="black")
    ax4.plot(df_anual["Año"], df_anual["Var_Piezas_Desest"], label="Piezas Δ% anual")
    ax4.plot(df_anual["Año"], df_anual["Var_Consumo_Desest"], label="Consumo Δ% anual")
    ax4.axhline(0, color="gray")
    ax4.grid(True)
    ax4.set_title("Variaciones mensuales desestacionalizadas - promedio anual")
    st.pyplot(fig4)
    
    # --- Correlación anual ---
    st.write(f"📊 Correlación ANUAL (Piezas vs EMAE): {corr_piezas_anual:.3f}")
    st.write(f"📊 Correlación ANUAL (Consumo vs EMAE): {corr_consumo_anual:.3f}")

except Exception as e:
    st.error(f"Ocurrió un error al cargar los datos: {e}")
    st.info("📌 Verificá que los archivos estén disponibles en la carpeta `streamlit_data` del repositorio.")
