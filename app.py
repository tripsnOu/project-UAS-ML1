import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(
    page_title="Prediksi Kepadatan Penduduk",
    page_icon="📈",
    layout="wide"
)

# Judul aplikasi
st.title("📈 Prediksi Kepadatan Penduduk")
st.markdown("Regresi Berganda dengan StandardScaler - Menggunakan Semua Data Tahun")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data_final.csv')
    return df

df = load_data()

# Fitur dan target
features = ['Jumlah Penduduk (Ribu)', 'Persentase Penduduk', 
            'Laju Pertumbuhan Penduduk per Tahun',
            'Rasio Jenis Kelamin Penduduk', 'Luas Wilayah (Km2)']
target = 'Kepadatan Penduduk per km persegi (Km2)'

# Sidebar
st.sidebar.header("⚙️ Pengaturan Model")
test_size = st.sidebar.slider("Ukuran Data Testing (%)", 10, 40, 20)

# Gunakan SEMUA data tahun
st.info(f"📊 Menggunakan **SEMUA DATA** dari tahun {df['Tahun'].min()} hingga {df['Tahun'].max()}")
st.info(f"Total data: **{len(df)}** observasi")

# Split data dengan SEMUA tahun
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size/100, random_state=42
)

# Scaling dengan StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Prediksi
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Main content - 3 bagian
tab1, tab2, tab3 = st.tabs(["📊 Evaluasi Model", "🔍 Koefisien Regresi", "🎯 Prediksi Manual"])

with tab1:
    st.header("📊 Evaluasi Model")
    
    # Metrik dalam 4 kolom
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        mae = mean_absolute_error(y_test, y_pred_test)
        st.metric("MAE", f"{mae:.2f}", 
                 help="Mean Absolute Error - Rata-rata kesalahan absolut")
    
    with col2:
        mse = mean_squared_error(y_test, y_pred_test)
        st.metric("MSE", f"{mse:.2f}", 
                 help="Mean Squared Error - Rata-rata kuadrat kesalahan")
    
    with col3:
        rmse = np.sqrt(mse)
        st.metric("RMSE", f"{rmse:.2f}", 
                 help="Root Mean Squared Error - Akar dari MSE")
    
    with col4:
        r2 = r2_score(y_test, y_pred_test)
        st.metric("R² Score", f"{r2:.4f}", 
                 help="Koefisien determinasi - Seberapa baik model menjelaskan variasi data")
    
    # Interpretasi R² Score
    st.subheader("📈 Interpretasi R² Score")
    
    if r2 >= 0.9:
        st.success(f"✅ **R² Score = {r2:.4f}** - Sangat Baik! Model menjelaskan {r2*100:.1f}% variasi data")
    elif r2 >= 0.7:
        st.info(f"ℹ️ **R² Score = {r2:.4f}** - Baik! Model menjelaskan {r2*100:.1f}% variasi data")
    elif r2 >= 0.5:
        st.warning(f"⚠️ **R² Score = {r2:.4f}** - Cukup! Model menjelaskan {r2*100:.1f}% variasi data")
    else:
        st.error(f"❌ **R² Score = {r2:.4f}** - Kurang Baik! Model hanya menjelaskan {r2*100:.1f}% variasi data")
    
    # Visualisasi prediksi vs aktual
    st.subheader("📉 Prediksi vs Aktual")
    
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Scatter plot
    scatter = ax1.scatter(y_test, y_pred_test, alpha=0.7, s=100, 
                         c=df.iloc[y_test.index]['Tahun'], cmap='viridis', 
                         edgecolors='black')
    
    # Garis ideal
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 
            'r--', linewidth=2, label='Garis Ideal (y=x)')
    
    ax1.set_xlabel('Nilai Aktual')
    ax1.set_ylabel('Nilai Prediksi')
    ax1.set_title('Perbandingan Prediksi vs Aktual (Semua Tahun)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Tambahkan colorbar untuk tahun
    plt.colorbar(scatter, ax=ax1, label='Tahun')
    
    st.pyplot(fig1)
    
    # Tabel hasil prediksi
    st.subheader("📋 Detail Prediksi")
    
    results_df = pd.DataFrame({
        'Kecamatan': df.iloc[X_test.index]['Kecamatan'].values,
        'Tahun': df.iloc[X_test.index]['Tahun'].values,
        'Aktual': y_test.values,
        'Prediksi': y_pred_test,
        'Selisih': y_pred_test - y_test.values,
        'Error (%)': (abs(y_pred_test - y_test.values) / y_test.values) * 100
    }).sort_values('Error (%)')
    
    # Rata-rata error per tahun
    avg_error_by_year = results_df.groupby('Tahun')['Error (%)'].mean().round(2)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Rata-rata Error per Tahun:**")
        st.dataframe(avg_error_by_year, use_container_width=True)
    
    with col2:
        st.write("**Statistik Error:**")
        error_stats = pd.DataFrame({
            'Metrik': ['Error Terkecil', 'Error Terbesar', 'Error Rata-rata'],
            'Nilai': [
                f"{results_df['Error (%)'].min():.1f}%",
                f"{results_df['Error (%)'].max():.1f}%",
                f"{results_df['Error (%)'].mean():.1f}%"
            ]
        })
        st.dataframe(error_stats, use_container_width=True)
    
    st.dataframe(results_df.style.format({
        'Aktual': '{:.0f}',
        'Prediksi': '{:.0f}',
        'Selisih': '{:.0f}',
        'Error (%)': '{:.1f}%'
    }).background_gradient(subset=['Error (%)'], cmap='RdYlGn_r'), 
    use_container_width=True)

with tab2:
    st.header("🔍 Koefisien Regresi")
    
    # Tampilkan koefisien
    coefficients = pd.DataFrame({
        'Fitur': features,
        'Koefisien': model.coef_,
        'Pengaruh': ['Meningkatkan' if c > 0 else 'Menurunkan' for c in model.coef_],
        'Besaran': np.abs(model.coef_)
    }).sort_values('Besaran', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Koefisien Model")
        st.dataframe(coefficients[['Fitur', 'Koefisien', 'Pengaruh']].style.format({
            'Koefisien': '{:.4f}'
        }).bar(subset=['Koefisien'], align='mid', color=['#ff6b6b', '#51cf66']),
        use_container_width=True)
    
    with col2:
        st.subheader("📈 Info Model")
        st.metric("Intercept", f"{model.intercept_:.2f}")
        st.metric("Total Data", len(df))
        st.metric("Training Set", len(X_train))
        st.metric("Testing Set", len(X_test))
        st.metric("R² Score", f"{r2:.4f}")
    
    # Visualisasi koefisien
    st.subheader("📉 Visualisasi Pengaruh Fitur")
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    colors = ['green' if c > 0 else 'red' for c in coefficients['Koefisien']]
    bars = ax2.barh(coefficients['Fitur'], coefficients['Koefisien'], color=colors)
    
    ax2.set_xlabel('Nilai Koefisien')
    ax2.set_title('Pengaruh Fitur terhadap Kepadatan Penduduk')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Tambahkan nilai pada bar
    for bar in bars:
        width = bar.get_width()
        ax2.text(width, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', 
                ha='left' if width > 0 else 'right', 
                va='center', fontweight='bold')
    
    st.pyplot(fig2)
    
    # Interpretasi koefisien
    st.subheader("📝 Interpretasi Koefisien")
    
    top_feature = coefficients.iloc[0]
    st.info(f"""
    **Fitur Paling Berpengaruh:** **{top_feature['Fitur']}**
    
    **Koefisien:** {top_feature['Koefisien']:.4f}
    
    **Artinya:**
    - Setiap peningkatan 1 standar deviasi pada **{top_feature['Fitur']}**
    - Akan **{top_feature['Pengaruh'].lower()}** kepadatan penduduk sebesar **{abs(top_feature['Koefisien']):.2f}** orang/km²
    """)

with tab3:
    st.header("🎯 Prediksi Manual")
    
    st.info("Masukkan nilai untuk 5 fitur berikut:")
    
    # Hitung statistik untuk SEMUA data
    stats_df = pd.DataFrame({
        'Fitur': features,
        'Min': [df[f].min() for f in features],
        'Max': [df[f].max() for f in features],
        'Rata-rata': [df[f].mean() for f in features],
        'Std Dev': [df[f].std() for f in features]
    })
    
    # Input dalam 3 kolom
    col1, col2 = st.columns(2)
    
    input_values = {}
    
    with col1:
        st.subheader("Input Fitur")
        for i, feature in enumerate(features[:3]):
            stats = stats_df[stats_df['Fitur'] == feature].iloc[0]
            
            input_values[feature] = st.number_input(
                f"{feature}",
                min_value=float(stats['Min']),
                max_value=float(stats['Max'] * 1.5),
                value=float(stats['Rata-rata']),
                step=float(stats['Std Dev'] / 10),
                help=f"Min: {stats['Min']:.1f}, Max: {stats['Max']:.1f}, Avg: {stats['Rata-rata']:.1f}"
            )
    
    with col2:
        st.subheader("")  # Spacer untuk alignment
        for i, feature in enumerate(features[3:]):
            stats = stats_df[stats_df['Fitur'] == feature].iloc[0]
            
            input_values[feature] = st.number_input(
                f"{feature}",
                min_value=float(stats['Min']),
                max_value=float(stats['Max'] * 1.5),
                value=float(stats['Rata-rata']),
                step=float(stats['Std Dev'] / 10),
                help=f"Min: {stats['Min']:.1f}, Max: {stats['Max']:.1f}, Avg: {stats['Rata-rata']:.1f}"
            )
    
    # Tampilkan statistik
    with st.expander("📊 Lihat Statistik Data"):
        st.dataframe(stats_df.style.format({
            'Min': '{:.2f}',
            'Max': '{:.2f}',
            'Rata-rata': '{:.2f}',
            'Std Dev': '{:.2f}'
        }), use_container_width=True)
    
    # Tombol prediksi
    if st.button("🚀 Prediksi Sekarang", type="primary", use_container_width=True):
        # Konversi ke DataFrame dan scale
        input_df = pd.DataFrame([input_values])
        input_scaled = scaler.transform(input_df)
        
        # Prediksi
        prediction = model.predict(input_scaled)[0]
        
        # Tampilkan hasil
        st.success("## 🎯 Hasil Prediksi")
        
        # Metrik dalam kartu
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Kepadatan Penduduk Prediksi", 
                f"{prediction:,.0f}",
                "orang/km²",
                delta_color="normal"
            )
        
        with col2:
            avg_density = df[target].mean()
            diff = prediction - avg_density
            diff_percent = (diff / avg_density) * 100
            st.metric(
                "vs Rata-rata Data", 
                f"{diff_percent:+.1f}%",
                f"{diff:+,.0f} orang/km²"
            )
        
        with col3:
            percentile = (df[target] < prediction).mean() * 100
            st.metric(
                "Percentile", 
                f"{percentile:.1f}%",
                f"lebih tinggi dari data lain"
            )
        
        # Visualisasi
        st.subheader("📊 Posisi Prediksi dalam Distribusi")
        
        fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram semua data
        ax1.hist(df[target], bins=30, alpha=0.6, color='skyblue', 
                edgecolor='black', density=True, label='Distribusi Data')
        
        # Kurva density
        from scipy.stats import gaussian_kde
        kde = gaussian_kde(df[target])
        x_range = np.linspace(df[target].min(), df[target].max(), 100)
        ax1.plot(x_range, kde(x_range), 'b-', linewidth=2)
        
        # Garis prediksi
        ax1.axvline(prediction, color='red', linestyle='--', 
                   linewidth=3, label=f'Prediksi: {prediction:,.0f}')
        
        # Garis rata-rata
        ax1.axvline(avg_density, color='green', linestyle=':', 
                   linewidth=2, label=f'Rata-rata: {avg_density:,.0f}')
        
        ax1.set_xlabel('Kepadatan Penduduk (orang/km²)')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribusi Kepadatan Penduduk (Semua Tahun)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot per tahun
        yearly_data = [df[df['Tahun'] == year][target] for year in sorted(df['Tahun'].unique())]
        box = ax2.boxplot(yearly_data, labels=sorted(df['Tahun'].unique()),
                         patch_artist=True)
        
        # Warna box plot
        colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
        for patch, color in zip(box['boxes'], colors):
            patch.set_facecolor(color)
        
        # Garis prediksi
        ax2.axhline(prediction, color='red', linestyle='--', 
                   linewidth=2, label=f'Prediksi: {prediction:,.0f}')
        
        ax2.set_xlabel('Tahun')
        ax2.set_ylabel('Kepadatan Penduduk')
        ax2.set_title('Distribusi per Tahun')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig3)
        
        # Ringkasan input
        st.subheader("📋 Ringkasan Input dan Kontribusi")
        
        # Hitung kontribusi setiap fitur
        contributions = model.coef_ * input_scaled[0]
        
        summary_df = pd.DataFrame({
            'Fitur': features,
            'Nilai Input': [input_values[f] for f in features],
            'Nilai Scaled': input_scaled[0],
            'Koefisien': model.coef_,
            'Kontribusi': contributions
        })
        
        # Hitung total kontribusi
        total_contribution = contributions.sum()
        intercept = model.intercept_
        
        st.write(f"**Rumus Prediksi:** `{intercept:.2f} + Σ(Koefisien × Nilai Scaled)`")
        st.write(f"**Intercept:** {intercept:.2f}")
        st.write(f"**Total Kontribusi Fitur:** {total_contribution:.2f}")
        st.write(f"**Prediksi Akhir:** {intercept:.2f} + {total_contribution:.2f} = **{prediction:.2f}**")
        
        st.dataframe(summary_df.style.format({
            'Nilai Input': '{:.2f}',
            'Nilai Scaled': '{:.4f}',
            'Koefisien': '{:.4f}',
            'Kontribusi': '{:.2f}'
        }).bar(subset=['Kontribusi'], align='mid', color=['#ff6b6b', '#51cf66']),
        use_container_width=True)

# Informasi dataset di sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("**📁 Info Dataset**")
st.sidebar.markdown(f"Total Data: {len(df)} observasi")
st.sidebar.markdown(f"Rentang Tahun: {df['Tahun'].min()} - {df['Tahun'].max()}")
st.sidebar.markdown(f"Jumlah Kecamatan: {df['Kecamatan'].nunique()}")
st.sidebar.markdown(f"Training Set: {len(X_train)} data ({100-test_size}%)")
st.sidebar.markdown(f"Testing Set: {len(X_test)} data ({test_size}%)")

st.sidebar.markdown("---")
st.sidebar.markdown("**📈 Performa Model**")
st.sidebar.markdown(f"R² Score: **{r2:.4f}**")
st.sidebar.markdown(f"RMSE: **{rmse:.1f}**")
st.sidebar.markdown(f"MAE: **{mae:.1f}**")
st.sidebar.markdown(f"MSE: **{mse:.1f}**")

# Footer
st.markdown("---")
st.markdown("**📊 Model Regresi Berganda** | Menggunakan SEMUA DATA dari tahun 2019-2022")
st.markdown(f"**R² Score:** {r2:.4f} | **RMSE:** {rmse:.1f} | **Jumlah Data:** {len(df)} observasi")