import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ======================================================
# CONFIG PAGE
# ======================================================
st.set_page_config(
    page_title="Prediksi Kategori Prosesor AMD Ryzen",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# LOAD MODEL
# ======================================================
model = joblib.load("model_rf_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.image("logo_kampus.png", width=150)
    st.markdown("## 🎛️ Mode Input")

    mode = st.radio(
        "Pilih Mode",
        ["Basic", "Advanced"]
    )

    dark_mode = st.toggle("🌗 Dark Mode")

# ======================================================
# DARK MODE STYLE
# ======================================================
if dark_mode:
    st.markdown("""
        <style>
        body { background-color: #0e1117; color: white; }
        .stApp { background-color: #0e1117; }
        </style>
    """, unsafe_allow_html=True)

# ======================================================
# TITLE
# ======================================================
st.markdown("""
<h1 style='text-align:center;'>🔮 Sistem Prediksi Kategori Prosesor AMD Ryzen</h1>
<p style='text-align:center;'>Berbasis Machine Learning (Random Forest)</p>
""", unsafe_allow_html=True)

st.divider()

# ======================================================
# INPUT FORM
# ======================================================
st.subheader("🧮 Spesifikasi Prosesor")

col1, col2, col3 = st.columns(3)

with col1:
    core = st.number_input("Core Count", 2, 64, 8)
    base_clock = st.number_input("Base Clock (GHz)", 2.0, 5.0, 3.5)

with col2:
    thread = st.number_input("Thread Count", 4, 128, 16)
    boost_clock = st.number_input("Boost Clock (GHz)", 2.0, 6.0, 5.0)

with col3:
    cache = st.number_input("Cache (MB)", 8, 256, 32)
    tdp = st.number_input("TDP (Watt)", 35, 200, 120)
    price = st.number_input("Price ($)", 100, 2000, 500)

# ======================================================
# ADVANCED MODE
# ======================================================
if mode == "Advanced":
    st.info("🔍 Advanced mode menampilkan confidence probabilitas tiap kelas")

# ======================================================
# PREDICTION
# ======================================================
if st.button("🚀 Prediksi Kategori"):
    input_data = np.array([[core, thread, base_clock, boost_clock, cache, tdp, price]])

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0]

    label = label_encoder.inverse_transform(prediction)[0]

    st.success(f"🎯 **Kategori Prosesor: {label}**")

    # ==================================================
    # PROBABILITY VISUALIZATION
    # ==================================================
    if mode == "Advanced":
        st.subheader("📊 Confidence Probabilitas")

        fig, ax = plt.subplots()
        ax.barh(label_encoder.classes_, probability)
        ax.set_xlabel("Probabilitas")
        ax.set_xlim(0, 1)

        st.pyplot(fig)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<hr>
<p style='text-align:center; font-size:14px;'>
© 2025 | Sistem Prediksi Prosesor AMD Ryzen<br>
Teknik Informatika
</p>
""", unsafe_allow_html=True)