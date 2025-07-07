import streamlit as st
import joblib
import os

# Cargar modelo entrenado (mock para demo)
MODEL_PATH = os.getenv("MODEL_PATH", "modelo_fraude.pkl")
model = joblib.load(MODEL_PATH)

st.title("🛡️ Clasificador de Publicaciones Sospechosas - Mercado Libre")

# Inputs
titulo = st.text_input("📝 Título de la publicación")
descripcion = st.text_area("📄 Descripción")
precio = st.number_input("💰 Precio", min_value=0.0, step=10.0)

# Feature engineering simple (mock)
if st.button("Predecir"):
    features = [[len(titulo), len(descripcion), precio]]
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    st.subheader("Resultado:")
    st.write(f"📦 Sospechosa: **{'Sí' if pred == 1 else 'No'}**")
    st.write(f"Probabilidad: {proba:.2%}")
