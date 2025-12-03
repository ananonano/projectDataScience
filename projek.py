import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Coffee Shop Analytics Dashboard",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk tampilan lebih profesional
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #ffffff;
        border-radius: 4px 4px 0px 0px;
        box-shadow: 0px 1px 2px rgba(0,0,0,0.1);
    }
    .stTabs [aria-selected="true"] {
        background-color: #e6f3ff;
        border-bottom: 2px solid #0068c9;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING & PROCESSING (CACHED)
# ==========================================
@st.cache_data
def load_data():
    try:
        file_path = 'Coffe_sales.csv'
        df = pd.read_csv(file_path)
        df.drop_duplicates(inplace=True)

        # Robust Date Parsing
        raw_datetime = df['Date'] + ' ' + df['Time']
        df['datetime'] = pd.to_datetime(raw_datetime, format="%Y-%m-%d %H:%M:%S.%f", errors='coerce')
        mask_nat = df['datetime'].isna()
        df.loc[mask_nat, 'datetime'] = pd.to_datetime(raw_datetime[mask_nat], format="%Y-%m-%d %H:%M:%S", errors='coerce')
        
        df.drop(columns=['Date', 'Time'], inplace=True)
        df['period'] = df['datetime'].dt.to_period('M')
        return df
    except FileNotFoundError:
        return None

df = load_data()

# ==========================================
# 3. SIDEBAR & NAVIGASI
# ==========================================
with st.sidebar:
    st.title("☕ Business Intelligence")
    st.info("**Client:** Coffee Shop Executive\n\n**Goal:** Ops & Marketing Optimization")
    
    st.markdown("---")
    st.write("Filter Global:")
    
    if df is not None:
        # Filter Bulan (Opsional, untuk demo kita gunakan semua data)
        selected_payment = st.multiselect(
            "Metode Pembayaran", 
            options=df['cash_type'].unique(),
            default=df['cash_type'].unique()
        )
        
        # Terapkan filter ke dataframe utama
        df_filtered = df[df['cash_type'].isin(selected_payment)]
    else:
        st.error("File 'Coffe_sales.csv' tidak ditemukan.")
        st.stop()

# ==========================================
# 4. MAIN DASHBOARD CONTENT
# ==========================================
st.title("📊 Laporan Kinerja & Strategi Bisnis")
st.markdown("Dashboard ini memberikan wawasan berbasis data untuk **Optimalisasi Inventaris** dan **Segmentasi Pelanggan**.")

# Tabs untuk memisahkan topik
tab1, tab2, tab3 = st.tabs(["📈 Executive Overview", "📦 Inventory Intelligence", "🎯 Marketing Strategy"])

# --- TAB 1: EXECUTIVE OVERVIEW ---
with tab1:
    st.subheader("Ringkasan Kinerja Bisnis")
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    total_sales = df_filtered['money'].sum()
    total_trx = len(df_filtered)
    avg_ticket = df_filtered['money'].mean()
    top_product = df_filtered['coffee_name'].mode()[0]
    
    col1.metric("Total Revenue", f"${total_sales:,.2f}")
    col2.metric("Total Transaksi", f"{total_trx:,}")
    col3.metric("Rata-rata Order", f"${avg_ticket:.2f}")
    col4.metric("Produk Terlaris", top_product)
    
    st.markdown("---")
    
    # Chart: Tren Penjualan & Top Produk
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("**Tren Penjualan Harian (Volume)**")
        daily_trend = df_filtered.groupby(df_filtered['datetime'].dt.date).size()
        st.line_chart(daily_trend)
        
    with c2:
        st.markdown("**Top 5 Produk (Berdasarkan Pendapatan)**")
        top_products = df_filtered.groupby('coffee_name')['money'].sum().nlargest(5).sort_values()
        fig, ax = plt.subplots(figsize=(5,3))
        top_products.plot(kind='barh', ax=ax, color='#4e342e')
        ax.set_ylabel("")
        st.pyplot(fig)

# --- TAB 2: INVENTORY INTELLIGENCE (Forecasting) ---
with tab2:
    st.header("Optimalisasi Inventaris Cerdas")
    st.markdown("Gunakan modul ini untuk memprediksi kebutuhan stok bulan depan dan menghindari *stockout*.")
    
    # User Input
    col_input, col_res = st.columns([1, 3])
    
    with col_input:
        selected_product = st.selectbox("Pilih Produk untuk Analisis:", df['coffee_name'].unique(), index=0)
        st.caption("Analisis dilakukan menggunakan model SMA (Moving Average) dan Linear Trend.")
    
    # --- Backend Logic untuk Forecasting ---
    product_data = df[df['coffee_name'] == selected_product].copy()
    monthly_sales = product_data.groupby('period').size().reset_index(name='actual_sales')
    monthly_sales['period_str'] = monthly_sales['period'].astype(str)
    monthly_sales['time_idx'] = np.arange(len(monthly_sales))
    
    # Train/Test Split
    train_size = max(len(monthly_sales) - 3, 1)
    train = monthly_sales.iloc[:train_size]
    test = monthly_sales.iloc[train_size:]
    
    # Modeling
    # 1. SMA
    monthly_sales['sma_forecast'] = monthly_sales['actual_sales'].rolling(window=3).mean().shift(1)
    
    # 2. Linear Regression
    if len(train) > 1:
        lr = LinearRegression()
        lr.fit(train[['time_idx']], train['actual_sales'])
        monthly_sales['trend_forecast'] = lr.predict(monthly_sales[['time_idx']])
    else:
        monthly_sales['trend_forecast'] = np.nan
        lr = None

    # Evaluasi & Rekomendasi
    if len(test) > 0:
        rmse_sma = np.sqrt(mean_squared_error(test['actual_sales'], monthly_sales.loc[test.index, 'sma_forecast'].fillna(0)))
        rmse_trend = np.sqrt(mean_squared_error(test['actual_sales'], monthly_sales.loc[test.index, 'trend_forecast'].fillna(0))) if lr else 999
        
        best_model = "SMA (Moving Average)" if rmse_sma < rmse_trend else "Linear Trend"
        best_rmse = min(rmse_sma, rmse_trend)
        
        # Forecast Bulan Depan
        if best_model.startswith("SMA"):
            next_forecast = monthly_sales['actual_sales'].iloc[-3:].mean()
        else:
            next_forecast = lr.predict([[monthly_sales['time_idx'].max() + 1]])[0]
            
        safety_stock = 1.65 * best_rmse
        total_rec = next_forecast + safety_stock
    else:
        # Fallback jika data terlalu sedikit
        next_forecast = monthly_sales['actual_sales'].mean()
        safety_stock = 0
        total_rec = next_forecast
        best_model = "Average (Insufficient Data)"
        best_rmse = 0

    # --- Display Result ---
    with col_res:
        # Kartu Rekomendasi (Highlight)
        st.success(f"### 📦 Rekomendasi Restock: **{int(total_rec)} Cup**")
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Prediksi Dasar", f"{int(next_forecast)} Cup")
        m2.metric("Safety Stock (Buffer)", f"{int(safety_stock)} Cup", help="Cadangan untuk antisipasi lonjakan permintaan (95% Service Level)")
        m3.metric("Model Akurasi Terbaik", best_model, delta=f"Error +/- {best_rmse:.1f}")

        # Visualisasi
        st.subheader("Grafik Peramalan & Realisasi")
        fig_forecast, ax_f = plt.subplots(figsize=(10, 4))
        ax_f.plot(monthly_sales['period_str'], monthly_sales['actual_sales'], marker='o', label='Aktual', linewidth=2)
        ax_f.plot(monthly_sales['period_str'], monthly_sales['sma_forecast'], '--', label='Prediksi SMA', color='orange')
        if lr:
            ax_f.plot(monthly_sales['period_str'], monthly_sales['trend_forecast'], ':', label='Prediksi Trend', color='green')
        
        ax_f.axvline(x=train_size - 0.5, color='gray', linestyle='--', alpha=0.5, label='Batas Data Training')
        ax_f.legend()
        ax_f.set_xlabel("Bulan")
        ax_f.set_ylabel("Volume Penjualan")
        plt.xticks(rotation=45)
        st.pyplot(fig_forecast)

# --- TAB 3: MARKETING STRATEGY (Segmentation) ---
with tab3:
    st.header("Strategi Promosi & Segmentasi Produk")
    st.markdown("Analisis ini mengelompokkan produk berdasarkan **Harga vs Popularitas** untuk menentukan strategi promosi yang tepat.")
    
    # 1. Clustering Data Prep
    prod_features = df.groupby('coffee_name').agg({
        'money': 'mean',
        'datetime': 'count'
    }).reset_index().rename(columns={'money': 'Avg_Price', 'datetime': 'Total_Transactions'})
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(prod_features[['Avg_Price', 'Total_Transactions']])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    prod_features['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Logic Penamaan Label Otomatis
    avg_trans = prod_features['Total_Transactions'].mean()
    avg_price = prod_features['Avg_Price'].mean()
    
    def get_label(row):
        if row['Total_Transactions'] > avg_trans:
            return "Star Product (Laris)" if row['Avg_Price'] > avg_price else "Cash Cow (Volume Maker)"
        else:
            return "Premium Niche (Mahal)" if row['Avg_Price'] > avg_price else "Slow Mover (Perlu Promo)"
            
    prod_features['Segment'] = prod_features.apply(get_label, axis=1)

    col_strat1, col_strat2 = st.columns([2, 1])

    with col_strat1:
        st.subheader("Peta Segmentasi Produk")
        fig_seg, ax_s = plt.subplots(figsize=(8, 5))
        sns.scatterplot(data=prod_features, x='Total_Transactions', y='Avg_Price', hue='Segment', s=200, style='Segment', palette='viridis', ax=ax_s)
        
        # Anotasi
        for i in range(prod_features.shape[0]):
            ax_s.text(prod_features.Total_Transactions[i]+2, prod_features.Avg_Price[i], 
                      prod_features.coffee_name[i], fontsize=8)
            
        ax_s.set_xlabel("Popularitas (Jumlah Transaksi)")
        ax_s.set_ylabel("Harga Rata-rata ($)")
        st.pyplot(fig_seg)
        
    with col_strat2:
        st.subheader("Rekomendasi Strategi")
        st.info("**Star Products:** Pertahankan stok, hindari diskon.")
        st.success("**Cash Cows:** Buat bundling dengan Slow Movers.")
        st.warning("**Slow Movers:** Berikan diskon agresif / Flash Sale.")

    st.markdown("---")
    st.subheader("🕒 Kapan Waktu Terbaik untuk Promo?")
    
    # Heatmap
    heatmap_data = pd.crosstab(df['coffee_name'], df['Time_of_Day'])
    heatmap_norm = heatmap_data.div(heatmap_data.sum(axis=1), axis=0)
    
    fig_heat, ax_h = plt.subplots(figsize=(10, 4))
    sns.heatmap(heatmap_norm, annot=True, fmt=".0%", cmap="YlGnBu", ax=ax_h)
    ax_h.set_title("Proporsi Penjualan per Waktu (Heatmap)")
    st.pyplot(fig_heat)
    st.caption("*Angka menunjukkan persentase penjualan produk tersebut terjadi pada waktu tertentu.*")