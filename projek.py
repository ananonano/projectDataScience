import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

st.set_page_config(
    page_title="Coffee Shop Analytics",
    page_icon="☕",
    layout="wide"
)

# FUNGSI LOAD ASET (MODEL & DATA) 
@st.cache_resource
def load_assets():
    try:
        # Load Model & Tools dari file .pkl
        model = joblib.load('model_final.pkl')
        scaler = joblib.load('scaler.pkl')
        le = joblib.load('le_day.pkl')
        
        # Load Data Bersih dari CSV
        df = pd.read_csv('clean_data_agg.csv')
        return model, scaler, le, df
    except FileNotFoundError:
        return None, None, None, None

# Panggil Fungsi Load
model, scaler, le, df = load_assets()

# SIDEBAR 
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/924/924514.png", width=100)
st.sidebar.title("Navigasi Proyek")
st.sidebar.info("**Anggota Kelompok:**\n1. Akmal Danendra Maulana\n2. Danang Adiwibowo\n3. Hafiz Alaudin Rasendriya")

if df is None:
    st.error("File aset tidak ditemukan! Pastikan `model_final.pkl`, `scaler.pkl`, `le_day.pkl`, dan `clean_data_agg.csv` ada di folder yang sama.")
    st.stop()

menu = st.sidebar.radio(
    "Menu:", 
    ["Dashboard Insight", "Prediksi Peak Hour"]
)

# HALAMAN 1: DASHBOARD INSIGHT
if menu == "Dashboard Insight":
    st.title("Dashboard Analisis Coffee Shop")
    st.markdown("Analisis pola transaksi berdasarkan data historis yang sudah dibersihkan.")

    # KPI Utama
    col1, col2, col3 = st.columns(3)
    total_trx = df['order_count'].sum()
    avg_trx = df['order_count'].mean()
    busiest_day = df.groupby('day_name')['order_count'].sum().idxmax()

    col1.metric("Total Transaksi", f"{total_trx:,}")
    col2.metric("Rata-rata Order/Jam", f"{avg_trx:.1f}")
    col3.metric("Hari Teramai", busiest_day)

    st.divider()

    # 1. Heatmap
    st.subheader("Kepadatan Pesanan (Hari vs Jam)")
    
    # Pivot table untuk heatmap
    pivot_heatmap = df.pivot_table(
        index='day_name', 
        columns='hour_of_day', 
        values='order_count', 
        aggfunc='sum'
    ).fillna(0)
    
    # Urutkan hari
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_heatmap = pivot_heatmap.reindex(days_order)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(pivot_heatmap, cmap='YlGnBu', annot=True, fmt='g', ax=ax)
    ax.set_xlabel("Jam Operasional")
    ax.set_ylabel("Hari")
    st.pyplot(fig)

    # 2. Tren Harian
    st.subheader("Tren Transaksi Berdasarkan Jam")
    hourly_trend = df.groupby('hour_of_day')['order_count'].mean().reset_index()
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.lineplot(data=hourly_trend, x='hour_of_day', y='order_count', marker='o', color='green', ax=ax2)
    ax2.set_xlabel("Jam")
    ax2.set_ylabel("Rata-rata Order")
    ax2.set_title("Rata-rata Kepadatan per Jam")
    ax2.grid(True, linestyle='--')
    st.pyplot(fig2)


# HALAMAN 2: PREDIKSI (DEPLOYMENT)
elif menu == "Prediksi Peak Hour":
    st.title("Prediksi Kepadatan Toko")
    st.markdown("Masukkan waktu operasional untuk memprediksi apakah toko akan **Ramai (Peak Hour)** atau **Normal**.")

    with st.form("prediksi_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Revisi: Jangan pakai slider
            jam_ops = list(range(6, 24)) 
            input_jam = st.selectbox("Pilih Jam Operasional:", jam_ops, index=6) 
        
        with col2:
            hari_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            input_hari = st.selectbox("Pilih Hari:", hari_list)
        
        with col3:
            # Revisi: Jangan pakai slider
            bulan_map = {
                1: 'Januari', 2: 'Februari', 3: 'Maret', 4: 'April', 
                5: 'Mei', 6: 'Juni', 7: 'Juli', 8: 'Agustus', 
                9: 'September', 10: 'Oktober', 11: 'November', 12: 'Desember'
            }
            input_bulan_nama = st.selectbox("Pilih Bulan:", list(bulan_map.values()))
            # Konversi balik nama bulan ke angka
            input_bulan = [k for k, v in bulan_map.items() if v == input_bulan_nama][0]

        submit_btn = st.form_submit_button("Prediksi Sekarang")

    if submit_btn:
        # 1. Preprocessing Input
        try:
            # Encode Hari
            hari_encoded = le.transform([input_hari])[0]
            
            # Buat DataFrame Input (Sesuai urutan training: hour, day_code, month)
            data_input = pd.DataFrame([[input_jam, hari_encoded, input_bulan]], 
                                      columns=['hour_of_day', 'day_code', 'month'])
            
            # Scaling Data
            data_scaled = scaler.transform(data_input)
            
            # 2. Prediksi
            hasil_prediksi = model.predict(data_scaled)[0]
            probabilitas = model.predict_proba(data_scaled)[0]

            # 3. Tampilkan Hasil
            st.divider()
            if hasil_prediksi == 1:
                st.error(f"###  HASIL: PEAK HOUR (RAMAI)")
                st.write(f"Tingkat Keyakinan Model: **{probabilitas[1]*100:.1f}%**")
                st.info(" **Saran:** Tambah personel barista dan pastikan stok susu/kopi aman.")
            else:
                st.success(f"### HASIL: NORMAL HOUR (SEPI)")
                st.write(f"Tingkat Keyakinan Model: **{probabilitas[0]*100:.1f}%**")
                st.info(" **Saran:** Bisa alokasikan staf untuk *cleaning* atau persiapan bahan.")
                
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")