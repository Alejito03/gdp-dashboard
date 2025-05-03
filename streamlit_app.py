import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Cargar el modelo pre-entrenado
model = load_model('ecg_model.h5')

# Cargar los datos (simulado)
@st.cache
def load_data():
    df = pd.read_csv('ecg_data.csv')
    return df

# Función para hacer predicción
def predict_ecg(signal):
    signal = np.reshape(signal, (1, len(signal), 1))  # Ajuste a la forma del modelo
    pred = model.predict(signal)
    return np.argmax(pred), np.max(pred)

# Subir el archivo CSV
st.title('Clasificador de ECG')
st.write("Sube un archivo CSV con la señal ECG")

uploaded_file = st.file_uploader("Elige un archivo CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write(df.head())

    # Visualización de la señal ECG
    st.write("Visualización de la señal ECG")
    plt.figure(figsize=(10, 5))
    plt.plot(df['Tiempo(ms)'], df['ECG'])
    plt.title('Señal ECG')
    plt.xlabel('Tiempo (ms)')
    plt.ylabel('ECG')
    st.pyplot()

    # Predicción
    pred_label, pred_conf = predict_ecg(df['ECG'].values)
    st.write(f"Predicción: {pred_label} con una confianza de {pred_conf:.2f}")

    # Mostrar métricas si se tiene
    st.write("Métricas de desempeño")
    st.write("Precisión, recall, F1-score, etc.")
