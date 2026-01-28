import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="Estratega de Inversión Pro - ESPOL", layout="wide")

st.title(" Sistema de Decisión de Inversión")
st.markdown("### Proyecto de Matemáticas Discretas: Optimización de Grafos de Decisión")

st.sidebar.header("Configuración de Activos")
activos = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Apple (AAPL)": "AAPL",
    "Nvidia (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA",
    "S&P 500 (^GSPC)": "^GSPC"
}
seleccion = st.sidebar.selectbox("Selecciona un Activo", list(activos.keys()))
ticker = activos[seleccion]

if st.sidebar.button("Ejecutar Análisis Optimizado"):
    with st.spinner('Procesando datos con indicadores avanzados...'):
        # 1. Obtención de datos (5 años para mejor entrenamiento)
        df = yf.download(ticker, period="5y", interval="1d")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

       
        df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean() / df['Close'].pct_change().rolling(14).std()))
       
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        df['Volatilidad'] = df['Close'].pct_change().rolling(window=20).std()
        
        df['Target'] = np.where(df['Close'].shift(-1) > (df['Close'] * 1.005), 1, 0)
        df = df.dropna()

        features = ['Close', 'RSI', 'SMA_50', 'Volatilidad']
        X = df[features]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  
        clf = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        pred_hoy = clf.predict(X.tail(1))[0]

        col1, col2 = st.columns(2)
        with col1:
            if pred_hoy == 1:
                st.success(f"###  SUGERENCIA: COMPRAR {seleccion}")
            else:
                st.error(f"### ⚠️ SUGERENCIA: NO INVERTIR EN {seleccion}")
            st.metric("Precisión del Modelo (Accuracy)", f"{acc*100:.2f}%")
        st.divider()
        c1, c2 = st.columns([1.5, 1])
        
        with c1:
            st.subheader("Estructura Lógica del Árbol")
            fig_tree, ax_tree = plt.subplots(figsize=(12,8))
            plot_tree(clf, feature_names=features, class_names=['Baja', 'Sube'], filled=True, ax=ax_tree, fontsize=7)
            st.pyplot(fig_tree)
            
        with c2:
            st.subheader("Matriz de Confusión")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Baja', 'Sube'])
            disp.plot(ax=ax_cm, cmap='Greens')
            st.pyplot(fig_cm)

