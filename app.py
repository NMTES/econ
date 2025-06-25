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

# --- Carga de archivos ---
st.sidebar.header("📂 Carga de archivos")
file_br = st.sidebar.file_uploader("Inflación Brasil (CSV)", type="csv")
file_ar = st.sidebar.file_uploader("Inflación Argentina (CSV)", type="csv")
file_blue = st.sidebar.file_uploader("USD/ARS Blue (CSV)", type="csv")
file_brl = st.sidebar.file_uploader("USD/BRL (CSV)", type="csv")
file_turismo = st.sidebar.file_uploader("Turismo receptivo/emisivo (XLSX)", type="xlsx")

if all([file_br, file_ar, file_blue, file_brl, file_turismo]):
    # Inflación Brasil
    df_br = pd.read_csv(file_br, skiprows=3)
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
    df_ar = pd.read_csv(file_ar, sep=';', encoding='latin1')
    ipc_ar = df_ar[(df_ar['Descripcion'] == 'NIVEL GENERAL') & (df_ar['Region'] == 'Nacional')].copy()
    ipc_ar['Fecha'] = pd.to_datetime(ipc_ar['Periodo'], format='%Y%m')
    ipc_ar['Infl_AR'] = ipc_ar['v_m_IPC'].str.replace(',', '.').astype(float)
    ipc_ar = ipc_ar[['Fecha', 'Infl_AR']]

    # Merge inflaciones
    df_comb = pd.merge(ipca_br, ipc_ar, on='Fecha', how='inner')

    # Blue
    df_ars = pd.read_csv(file_blue)
    df_ars['Fecha'] = pd.to_datetime(df_ars['category'])
    df_ars['valor'] = df_ars['valor'].astype(float)
    df_ars['Mes'] = df_ars['Fecha'].dt.to_period('M')
    blue_mensual = df_ars.sort_values('Fecha').groupby('Mes').tail(1).copy()
    blue_mensual['Fecha'] = blue_mensual['Mes'].dt.to_timestamp()
    blue_mensual = blue_mensual[['Fecha', 'valor']].rename(columns={'valor': 'ARS_USD_Blue'})

    # BRL
    df_brl = pd.read_csv(file_brl)
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
    ax1.legend()
    fig1.tight_layout()
    st.pyplot(fig1)

    # --- TURISMO ---
    df = pd.read_excel(file_turismo, header=2, engine="openpyxl")
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

else:
    st.info("📌 Por favor, cargá todos los archivos desde la barra lateral para continuar.")
