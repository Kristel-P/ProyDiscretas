import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.set_page_config(page_title="Estratega de Inversión - ESPOL", layout="wide")

st.title("Sistema de Decisión de Inversión (ML)")
st.markdown("Proyecto de Matemáticas Discretas - Análisis de Activos mediante Árboles de Decisión")

st.sidebar.header("Configuración")
activos = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Apple (AAPL)": "AAPL",
    "Nvidia (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA",
    "S&P 500 (Índice USA)": "^GSPC",
    "Amazon (AMZN)": "AMZN",
    "Microsoft (MSFT)": "MSFT"
}
seleccion = st.sidebar.selectbox("Selecciona un Activo", list(activos.keys()))
ticker = activos[seleccion]

if st.sidebar.button("Analizar e Invertir"):
    with st.spinner('Descargando datos y entrenando modelo...'):
       
        df = yf.download(ticker, period="5y", interval="1d")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean() / df['Close'].pct_change().rolling(14).std()))
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        df = df.dropna()

        X = df[['Close', 'RSI']]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = DecisionTreeClassifier(max_depth=3)
        clf.fit(X_train, y_train)

        pred = clf.predict(X.tail(1))[0]
    
        col1, col2 = st.columns(2)
        with col1:
            if pred == 1:
                st.success(f"###  RECOMENDACIÓN: INVERTIR EN {ticker}")
            else:
                st.error(f"### ⚠️ RECOMENDACIÓN: NO INVERTIR EN {ticker}")
        
        st.divider()
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.subheader("Estructura del Árbol (Grafo)")
            fig_tree, ax_tree = plt.subplots()
            plot_tree(clf, feature_names=['Precio', 'RSI'], class_names=['Baja', 'Sube'], filled=True, ax=ax_tree)
            st.pyplot(fig_tree)
            
        with col_b:
            st.subheader("Matriz de Confusión (Fiabilidad)")
            y_pred = clf.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Baja', 'Sube'])
            disp.plot(ax=ax_cm, cmap='Blues')

            st.pyplot(fig_cm)
