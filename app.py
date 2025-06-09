# Streamlit App para análisis de inflación, TCR y EMAE

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression

# Asegurar que seaborn esté instalado
try:
    import seaborn as sns
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns

sns.set(style="whitegrid")  # Mejora visual

st.set_page_config(page_title="Análisis Macroeconómico", layout="wide")
st.title("📊 Análisis de inflación, tipo de cambio y EMAE")

# --- Subida de archivos ---
st.sidebar.header("Subí tus archivos CSV")
file_brasil = st.sidebar.file_uploader("Inflación Brasil (IPCA)", type="csv")
file_arg = st.sidebar.file_uploader("Inflación Argentina (IPC)", type="csv")
file_ars = st.sidebar.file_uploader("Dólar Blue ARS/USD", type="csv")
file_brl = st.sidebar.file_uploader("USD/BRL", type="csv")
file_import = st.sidebar.file_uploader("Importaciones por uso económico", type="csv")
file_emae = st.sidebar.file_uploader("EMAE desestacionalizado", type="csv")

if all([file_brasil, file_arg, file_ars, file_brl, file_import, file_emae]):
    # --- Procesar IPCA Brasil ---
    df_br = pd.read_csv(file_brasil, skiprows=3)
    df_brasil = df_br[df_br.iloc[:, 0] == "Brasil"]
    fechas_str = df_brasil.columns[1:]
    valores = df_brasil.values[0][1:]
    ipca_br = pd.DataFrame({"Fecha": fechas_str, "Infl_BR": valores})
    ipca_br['Infl_BR'] = ipca_br['Infl_BR'].astype(str).str.replace(',', '.').astype(float)
    meses_pt = {'janeiro':'01','fevereiro':'02','março':'03','abril':'04','maio':'05','junho':'06','julho':'07','agosto':'08','setembro':'09','outubro':'10','novembro':'11','dezembro':'12'}
    ipca_br[['Mes', 'Año']] = ipca_br['Fecha'].str.extract(r'(\w+)\s+(\d{4})')
    ipca_br['Mes'] = ipca_br['Mes'].map(meses_pt)
    ipca_br['Fecha'] = pd.to_datetime(ipca_br['Año'] + '-' + ipca_br['Mes'] + '-01')
    ipca_br = ipca_br[ipca_br['Fecha'] >= '2019-01-01'][['Fecha', 'Infl_BR']]

    # --- IPC Argentina ---
    df_ar = pd.read_csv(file_arg, sep=';', encoding='latin1')
    ipc_ar = df_ar[(df_ar['Descripcion'] == 'NIVEL GENERAL') & (df_ar['Region'] == 'Nacional')].copy()
    ipc_ar['Fecha'] = pd.to_datetime(ipc_ar['Periodo'], format='%Y%m')
    ipc_ar['Infl_AR'] = ipc_ar['v_m_IPC'].str.replace(',', '.').astype(float)
    ipc_ar = ipc_ar[['Fecha', 'Infl_AR']]

    # --- Unir inflaciones ---
    df_comb = pd.merge(ipca_br, ipc_ar, on='Fecha')
    df_comb = df_comb.sort_values('Fecha')

    # --- Plot Inflaciones ---
    st.subheader("Inflación Mensual: Brasil vs Argentina")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_comb['Fecha'], df_comb['Infl_BR'], label='Brasil', marker='o')
    ax.plot(df_comb['Fecha'], df_comb['Infl_AR'], label='Argentina', marker='o')
    ax.set_ylabel('% mensual')
    ax.set_xlabel('Fecha')
    ax.set_title('Inflación Mensual')
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # --- Dólar Blue ARS/USD ---
    df_ars = pd.read_csv(file_ars)
    df_ars['Fecha'] = pd.to_datetime(df_ars['category'])
    df_ars['valor'] = df_ars['valor'].astype(float)
    df_ars['Mes'] = df_ars['Fecha'].dt.to_period('M')
    blue_mensual = df_ars.sort_values('Fecha').groupby('Mes').tail(1).copy()
    blue_mensual['Fecha'] = blue_mensual['Mes'].dt.to_timestamp()
    blue_mensual = blue_mensual[['Fecha', 'valor']].rename(columns={'valor': 'ARS_USD_Blue'})

    # --- USD/BRL ---
    df_brl = pd.read_csv(file_brl)
    df_brl['Fecha'] = pd.to_datetime(df_brl['Fecha'], format='%d.%m.%Y')
    df_brl['USD_BRL'] = df_brl['Último'].str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)
    df_brl = df_brl[['Fecha', 'USD_BRL']].sort_values('Fecha')

    # --- Importaciones ---
    df_raw = pd.read_csv(file_import, sep=';', encoding='utf-8', skiprows=5, decimal=',')
    df_raw.columns = df_raw.columns.str.strip()
    meses_validos = ["Enero","Febrero","Marzo","Abril","Mayo","Junio","Julio","Agosto","Septiembre","Octubre","Noviembre","Diciembre"]
    df_raw["Mes"] = df_raw.iloc[:,1].astype(str).str.strip().str.capitalize()
    df_raw = df_raw[df_raw["Mes"].isin(meses_validos)].copy()
    df_raw["Mes_num"] = df_raw["Mes"].apply(lambda x: meses_validos.index(x) + 1)
    año_actual = 2004
    años = []
    for i, row in df_raw.iterrows():
        val = str(row.iloc[0]).strip()
        val = ''.join(filter(str.isdigit, val))
        if val.isdigit():
            año_actual = int(val)
        años.append(año_actual)
    df_raw["Año"] = años
    df_raw["Fecha"] = pd.to_datetime(dict(year=df_raw["Año"], month=df_raw["Mes_num"], day=1))
    pot = df_raw.select_dtypes(include="number").columns
    conteo = df_raw[pot].notna().sum().sort_values(ascending=False)
    piezas_col = conteo.index[0]
    consumo_col = conteo.index[1]
    df = pd.DataFrame({
        "Fecha": df_raw["Fecha"],
        "Piezas_Cantidad": pd.to_numeric(df_raw[piezas_col], errors="coerce"),
        "Consumo_Cantidad": pd.to_numeric(df_raw[consumo_col], errors="coerce")
    })
    df["Año"] = df["Fecha"].dt.year
    df["Mes"] = df["Fecha"].dt.month
    prom_anual = df.groupby("Año")[["Piezas_Cantidad","Consumo_Cantidad"]].transform("mean")
    df["Piezas_Proxy"] = df["Piezas_Cantidad"] / prom_anual["Piezas_Cantidad"]
    df["Consumo_Proxy"] = df["Consumo_Cantidad"] / prom_anual["Consumo_Cantidad"]
    coef = df.groupby("Mes")[["Piezas_Proxy", "Consumo_Proxy"]].mean().reset_index()
    df = df.merge(coef, on="Mes", suffixes=("", "_coef"))
    df["Piezas_Desest"] = df["Piezas_Cantidad"] / df["Piezas_Proxy_coef"]
    df["Consumo_Desest"] = df["Consumo_Cantidad"] / df["Consumo_Proxy_coef"]
    df["Var_Piezas_Desest"] = df["Piezas_Desest"].pct_change(fill_method=None) * 100
    df["Var_Consumo_Desest"] = df["Consumo_Desest"].pct_change(fill_method=None) * 100
    df_final = df[["Fecha", "Piezas_Desest", "Var_Piezas_Desest", "Consumo_Desest", "Var_Consumo_Desest"]]

    # --- EMAE ---
    df_emae = pd.read_csv(file_emae, sep=";", decimal=",", encoding="utf-8", skiprows=4, header=None)
    df_emae.columns = ["Año", "Mes", "Serie_Original", "Var_anual", "EMAE_Desest", "Var_mensual_desest", "Tendencia_Ciclo", "Var_mensual_tendencia"]
    df_emae = df_emae[df_emae["EMAE_Desest"].notna()].copy()
    meses_map = {"Enero":1,"Febrero":2,"Marzo":3,"Abril":4,"Mayo":5,"Junio":6,"Julio":7,"Agosto":8,"Septiembre":9,"Octubre":10,"Noviembre":11,"Diciembre":12}
    a_actual = None
    años = []
    for val in df_emae["Año"]:
        if pd.notna(val):
            a_actual = int(val)
        años.append(a_actual)
    df_emae["Año"] = años
    df_emae["Mes_num"] = df_emae["Mes"].str.strip().map(meses_map)
    df_emae["Fecha"] = pd.to_datetime(dict(year=df_emae["Año"], month=df_emae["Mes_num"], day=1))
    df_emae = df_emae.sort_values("Fecha")
    df_emae["Var_EMAE"] = df_emae["EMAE_Desest"].pct_change(fill_method=None) * 100
    df_merged = pd.merge(df_final, df_emae[["Fecha", "Var_EMAE"]], on="Fecha", how="inner")
    corr_piezas = df_merged["Var_Piezas_Desest"].corr(df_merged["Var_EMAE"])
    corr_consumo = df_merged["Var_Consumo_Desest"].corr(df_merged["Var_EMAE"])

    # --- Gráficos finales ---
    st.subheader("Correlaciones Importaciones - EMAE")
    st.markdown(f"- Piezas y Accesorios vs EMAE: **{corr_piezas:.2f}**\n- Bienes de Consumo vs EMAE: **{corr_consumo:.2f}**")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.regplot(x="Var_Piezas_Desest", y="Var_EMAE", data=df_merged, ax=ax[0])
    ax[0].set_title(f"Piezas vs EMAE (r={corr_piezas:.2f})")
    sns.regplot(x="Var_Consumo_Desest", y="Var_EMAE", data=df_merged, ax=ax[1])
    ax[1].set_title(f"Consumo vs EMAE (r={corr_consumo:.2f})")
    st.pyplot(fig)

else:
    st.warning("Subí todos los archivos requeridos para visualizar el análisis.")
