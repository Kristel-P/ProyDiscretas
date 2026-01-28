import tkinter as tk
from tkinter import ttk, messagebox
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

ACTIVOS_DISPONIBLES = {
    "Bitcoin (BTC-USD)": "BTC-USD",
    "Ethereum (ETH-USD)": "ETH-USD",
    "Apple (AAPL)": "AAPL",
    "Nvidia (NVDA)": "NVDA",
    "Tesla (TSLA)": "TSLA",
    "S&P 500 (Índice USA)": "^GSPC"
}

def procesar_inversion():
    seleccion = combo.get()
    if seleccion == "Elige una opción...":
        messagebox.showwarning("Atención", "Por favor, elige un activo.")
        return
    
    ticker = ACTIVOS_DISPONIBLES[seleccion]
    
    try:
        lbl_estado.config(text=f"⏳ Analizando {ticker}...", foreground="blue")
        root.update()
        
        df = yf.download(ticker, period="5y", interval="1d")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean() / df['Close'].pct_change().rolling(14).std()))
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
        df = df.dropna()

        features = ['Close', 'RSI', 'SMA_20']
        X = df[features]
        y = df['Target']

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        clf = DecisionTreeClassifier(max_depth=3, random_state=42)
        clf.fit(X_train, y_train)

        pred_hoy = clf.predict(X.tail(1))[0]
        y_pred_test = clf.predict(X_test) 
        if pred_hoy == 1:
            lbl_resultado.config(text="  RESULTADO: ¡SÍ, INVIERTE!", foreground="#2ecc71")
        else:
            lbl_resultado.config(text="⚠️ RESULTADO: NO INVIERTAS", foreground="#e74c3c")

        fig, ax = plt.subplots(1, 2, figsize=(16, 6))

        plot_tree(clf, feature_names=features, class_names=['BAJA', 'SUBE'], filled=True, ax=ax[0])
        ax[0].set_title("Estructura Lógica (Árbol de Decisión)")

        # Lado derecho: Matriz de Confusión
        cm = confusion_matrix(y_test, y_pred_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Baja', 'Sube'])
        disp.plot(ax=ax[1], cmap='Blues')
        ax[1].set_title("Fiabilidad (Matriz de Confusión)")

        lbl_estado.config(text="✅ Análisis y evaluación listos", foreground="green")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        messagebox.showerror("Error", f"Fallo técnico: {str(e)}")

root = tk.Tk()
root.title("Estratega de Inversión ESPOL - Análisis de Fiabilidad")
root.geometry("500x450")

tk.Label(root, text="SISTEMA DE DECISIÓN CON MATRIZ DE CONFUSIÓN", font=("Arial", 12, "bold")).pack(pady=20)
combo = ttk.Combobox(root, values=list(ACTIVOS_DISPONIBLES.keys()), state="readonly", width=35)
combo.pack(pady=10)
combo.set("Elige una opción...")

btn_accion = tk.Button(root, text="ANALIZAR FIABILIDAD", command=procesar_inversion, 
                       bg="#34495e", fg="white", font=("Arial", 12, "bold"), padx=20)
btn_accion.pack(pady=20)

lbl_resultado = tk.Label(root, text="", font=("Arial", 13, "bold"))
lbl_resultado.pack(pady=10)

lbl_estado = tk.Label(root, text="Listo para evaluar", font=("Arial", 10, "italic"))
lbl_estado.pack(side="bottom", pady=10)

root.mainloop()