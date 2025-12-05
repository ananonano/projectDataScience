import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler # Import StandardScaler

GLOBAL_RANDOM_STATE = 42

st.set_page_config(
    page_title="UAS Sains Data - Coffee Shop Analytics",
    page_icon="☕",
    layout="wide"
)

@st.cache_data
def load_data():
    """Membaca file CSV yang diunggah."""
    try:
        df = pd.read_csv('Coffe_sales.csv') 
        return df
    except FileNotFoundError:
        st.error("File 'Coffe_sales.csv' tidak ditemukan. Pastikan file ada di folder yang sama.")
        return None

df_raw = load_data()

def run_data_preparation(df_raw):
    """Menerapkan logika agregasi, feature engineering, dan encoding dari Jupyter Notebook."""
    df = df_raw.copy()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    else:
        st.error("Kolom 'Date' tidak ditemukan.")
        return None, None, None

    # 1. Agregasi Data
    df_agg = df.groupby(['Date', 'hour_of_day']).size().reset_index(name='order_count')
    
    # 2. Menentukan Threshold Peak Hour (Quantil 75%)
    threshold = df_agg['order_count'].quantile(0.75)
    
    # 3. Membuat Label Binary (Target Y)
    df_agg['is_peak_hour'] = (df_agg['order_count'] > threshold).astype(int)
    
    # 4. Ekstrak fitur tambahan
    df_agg['day_name'] = df_agg['Date'].dt.day_name()
    df_agg['month'] = df_agg['Date'].dt.month
    
    # 5. Encoding Day
    le = LabelEncoder()
    df_agg['day_code'] = le.fit_transform(df_agg['day_name']) 
    
    return df_agg, le, threshold

if df_raw is not None:
    df_agg, le, threshold = run_data_preparation(df_raw)
    st.session_state['data_ready'] = df_agg
    st.session_state['encoder'] = le
    st.session_state['threshold'] = threshold

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("Navigasi Proyek UAS")
st.sidebar.info("Nama Kelompok:\n1. Akmal Danendra Maulana\n2. Danang Adiwibibowo\n3. Hafiz Alaudin Rasendriya")
menu = st.sidebar.radio(
    "Pilih Tahapan CRISP-DM:", 
    ["1. Business & Data Understanding", "2. EDA (Exploratory Data Analysis)", "3. Data Preparation", "4. Modelling & Evaluation (Random Forest)", "5. Deployment (Prediksi)"]
)

if menu == "1. Business & Data Understanding":
    st.title("☕ Analisis & Prediksi Peak Hours Kedai Kopi")
    
    st.header("Business Understanding")
    st.write("""
    **Tujuan Bisnis:** Memprediksi apakah suatu jam akan menjadi **Peak Hour (Jam Sibuk)** atau **Normal Hour** untuk optimasi staf, stok, dan alokasi sumber daya.
    """)

    st.header("Data Understanding")
    if df_raw is not None:
        st.write("Dataset Awal (`Coffe_sales.csv`):")
        st.dataframe(df_raw.head())
        st.write(f"**Dimensi Data:** {df_raw.shape[0]} Baris, {df_raw.shape[1]} Kolom")
        
        st.subheader("Fitur Utama yang Digunakan Model")
        st.markdown("""
        * **Fitur X (Prediktor):** `hour_of_day`, `day_code`, `month`
        * **Fitur Y (Target):** `is_peak_hour` (0=Normal, 1=Peak)
        """)
    else:
        st.error("Data tidak ditemukan.")

elif menu == "2. EDA (Exploratory Data Analysis)":
    st.title("🔍 Exploratory Data Analysis (EDA)")
    
    if df_raw is not None:
        
        # 1. Distribusi Transaksi per Jam
        st.subheader("1. Distribusi Jumlah Transaksi Berdasarkan Jam Operasional")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=df_raw, x='hour_of_day', bins=16, kde=True, color='skyblue', ax=ax)
        ax.set_xlabel('Jam Transaksi (Hour of Day)')
        ax.set_ylabel('Jumlah Transaksi')
        ax.grid(axis='y', linestyle='--')
        st.pyplot(fig)

        # 2. Top 10 Menu Kopi
        st.subheader("2. Top 10 Jenis Kopi dengan Transaksi Terbanyak")
        top_products = df_raw['coffee_name'].value_counts().head(10)
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=top_products.values, y=top_products.index, palette='viridis', ax=ax2)
        ax2.set_xlabel('Jumlah Transaksi')
        ax2.set_ylabel('Nama Produk')
        st.pyplot(fig2)

        # 3. Heatmap Harian vs Jam
        st.subheader("3. Heatmap Kepadatan Transaksi (Hari vs Jam)")
        
        # Menggunakan day_name dari data agregasi
        df_heatmap_hourly = st.session_state['data_ready'].groupby(['day_name', 'hour_of_day']).size().unstack(fill_value=0)
        
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        sns.heatmap(df_heatmap_hourly, cmap='YlGnBu', annot=True, fmt='d', linewidths=.5, linecolor='black', ax=ax3)
        ax3.set_title('Kepadatan Transaksi Berdasarkan Hari dan Jam')
        ax3.set_ylabel('Hari')
        ax3.set_xlabel('Jam')
        st.pyplot(fig3)

elif menu == "3. Data Preparation":
    st.title("⚙️ Data Preparation")
    
    if 'data_ready' in st.session_state:
        df = st.session_state['data_ready']
        
        st.subheader("1. Transformasi Data dan Pembuatan Target")
        st.write("Data transaksi diubah menjadi data teragregasi per-jam untuk menentukan keramaian.")
        
        st.info(f"**Threshold Peak Hour:** Lebih dari **{int(st.session_state['threshold'])}** order/jam dikategorikan 'Peak Hour' (1).")
        
        st.dataframe(df[['Date', 'hour_of_day', 'order_count', 'is_peak_hour', 'day_name', 'day_code']].head())

        st.subheader("2. Korelasi Fitur (Siapa yang paling ngaruh ke Peak Hour?)")
        # Visualisasi Korelasi
        fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[['hour_of_day', 'month', 'day_code', 'is_peak_hour']].corr(), 
                    annot=True, 
                    cmap='coolwarm', 
                    fmt=".2f",
                    ax=ax_corr)
        ax_corr.set_title('Korelasi Antar Variabel')
        st.pyplot(fig_corr)
        st.markdown("**Insight:** Variabel dengan nilai absolut (tanpa minus) tertinggi terhadap `is_peak_hour` adalah fitur yang paling penting bagi model.")
    else:
        st.error("Data mentah tidak tersedia.")

elif menu == "4. Modelling & Evaluation (Random Forest)":
    st.title("🤖 Modelling & Evaluation: Random Forest Classifier")

    if 'data_ready' in st.session_state:
        df = st.session_state['data_ready']
        
        # Definisi Fitur (X) dan Target (y)
        X = df[['hour_of_day', 'day_code', 'month']]
        y = df['is_peak_hour']
        
        # Split Data (Menggunakan slider Data Training dan RANDOM_STATE=42)
        st.subheader("1. Splitting Data (Stratified Split)")
        split_train_percent = st.slider("Rasio Data Training (%)", 60, 90, 80)
        test_size_ratio = (100 - split_train_percent) / 100
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size_ratio, 
            random_state=GLOBAL_RANDOM_STATE, 
            stratify=y
        )
        st.info(f"Data Training: {len(X_train)} ({split_train_percent}%) | Data Testing: {len(X_test)} ({(100-split_train_percent)}%)")

        # 2. SCALING DATA (StandardScaler) - Ditambahkan untuk konsistensi dengan IPYNB
        st.subheader("2. Scaling Data (StandardScaler)")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Konversi kembali ke DataFrame untuk menjaga nama kolom
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
        
        st.write("Data fitur sudah di-scale.")

        # 3. Model Training (Random Forest)
        st.subheader("3. Model Training: Random Forest Classifier")
        
        n_estimators = st.slider("Jumlah Pohon (n_estimators)", 10, 200, 100)
        
        # class_weight='balanced' dan RANDOM_STATE=42 konsisten
        model = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=GLOBAL_RANDOM_STATE, 
            class_weight='balanced'
        )
        
        # Model dilatih pada data yang sudah di-scale
        model.fit(X_train_scaled_df, y_train)
        
        # Simpan model
        st.session_state['model'] = model

        # Evaluation
        st.subheader("4. Evaluation Metrics")
        
        # Prediksi menggunakan data yang sudah di-scale
        y_pred = model.predict(X_test_scaled_df) 
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        st.metric("Akurasi Model", f"{acc:.2%}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.text("Classification Report:")
            report = classification_report(y_test, y_pred, output_dict=True)
            st.dataframe(pd.DataFrame(report).transpose()) 
        
        with col2:
            st.text("Confusion Matrix:")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            # CONFUSION MATRIX DISAMAKAN
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
            ax.set_title('Confusion Matrix - Random Forest')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
        # 5. Feature Importance (DISAMAKAN PERSIS)
        st.subheader("5. Feature Importance")
        
        importances = model.feature_importances_
        feature_names = X.columns 

        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance Score': importances
        })

        feature_importance_df = feature_importance_df.sort_values(
            by='Importance Score', ascending=False
        ).reset_index(drop=True)

        st.dataframe(feature_importance_df)

        fig_feat, ax_feat = plt.subplots(figsize=(8, 4))
        sns.barplot(x='Importance Score', y='Feature', data=feature_importance_df, palette='viridis', ax=ax_feat)
        ax_feat.set_title('Kontribusi Fitur ke Peak Hour')
        st.pyplot(fig_feat)

    else:
        st.warning("Data belum diproses. Kembali ke menu '3. Data Preparation'!")


elif menu == "5. Deployment (Prediksi)":
    st.title("🚀 Deployment: Prediksi Peak Hour")

    if 'model' in st.session_state and 'encoder' in st.session_state:
        model = st.session_state['model']
        le = st.session_state['encoder']
        
        with st.form("prediction_form"):
            st.subheader("Masukkan Parameter Waktu (Input untuk Prediksi)")
            
            col1, col2 = st.columns(2)
            with col1:
                input_hour = st.slider("Pilih Jam Operasional:", 6, 22, 12)
            with col2:
                days_list = le.classes_
                input_day = st.selectbox("Pilih Hari:", days_list)
            
            input_month = st.selectbox("Bulan:", range(1, 13), index=2) 
            
            submit = st.form_submit_button("Prediksi Keramaian")
            
            if submit:
                # Preprocess input
                day_encoded = le.transform([input_day])[0]
                input_data_unscaled = pd.DataFrame({
                    'hour_of_day': [input_hour],
                    'day_code': [day_encoded],
                    'month': [input_month]
                })
                
                # Input harus di-scale sebelum diprediksi (sesuai pipeline training)
                # Ambil scaler dari training (asumsi scaler disimpan saat training)
                if 'scaler' not in st.session_state:
                    # Latih scaler dengan data train yang sudah ada
                    X_dummy = st.session_state['data_ready'][['hour_of_day', 'day_code', 'month']]
                    X_train_dummy, _, _, _ = train_test_split(X_dummy, st.session_state['data_ready']['is_peak_hour'], test_size=0.2, random_state=GLOBAL_RANDOM_STATE, stratify=st.session_state['data_ready']['is_peak_hour'])
                    scaler = StandardScaler()
                    scaler.fit(X_train_dummy)
                    st.session_state['scaler'] = scaler # Simpan scaler
                else:
                    scaler = st.session_state['scaler']

                input_data_scaled = scaler.transform(input_data_unscaled)
                
                # Prediksi
                prediction = model.predict(input_data_scaled)[0]
                prob = model.predict_proba(input_data_scaled)[0][1] # Probabilitas Peak Hour (Kelas 1)
                
                st.divider()
                if prediction == 1:
                    st.error(f"🔥 **HASIL PREDIKSI: PEAK HOUR (RAMAI)!**")
                    st.write(f"Tingkat Keyakinan Ramai: **{prob:.2%}**")
                    st.markdown("⚠️ **REKOMENDASI:** Tambah staf, siapkan stok kopi lebih banyak.")
                else:
                    st.success(f"✅ **HASIL PREDIKSI: NORMAL HOUR (SEPI)**")
                    st.write(f"Tingkat Keyakinan Ramai: **{prob:.2%}**")
                    st.markdown("👍 **REKOMENDASI:** Aman, bisa gunakan waktu untuk *cleaning* atau *training* staf.")
    else:
        st.warning("Model belum dilatih. Silakan ke menu '4. Modelling & Evaluation' terlebih dahulu.")