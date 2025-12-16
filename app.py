import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config dengan tema gagah
st.set_page_config(
    page_title="Prediksi Kepadatan Penduduk",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state for theme
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# Toggle theme function
def toggle_theme():
    if st.session_state.theme == 'dark':
        st.session_state.theme = 'light'
    else:
        st.session_state.theme = 'dark'

# TEMA CUSTOM STYLE - DARK MODE & LIGHT MODE
dark_theme = """
<style>
    /* Warna Utama: Navy Blue, Dark Grey, Gold Accents */
    :root {
        --primary: #0f172a;
        --secondary: #1e293b;
        --accent: #f59e0b;
        --accent2: #dc2626;
        --text: #e2e8f0;
        --text-light: #94a3b8;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(15, 23, 42, 0.95) !important;
        border-right: 2px solid #f59e0b;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #fbbf24 !important;
        font-weight: 800 !important;
        border-bottom: 2px solid #f59e0b;
        padding-bottom: 5px;
        margin-bottom: 15px !important;
    }
    
    /* Metrics Cards */
    div[data-testid="stMetricValue"] {
        color: #fbbf24 !important;
        font-size: 24px !important;
        font-weight: 800 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMetricDelta"] {
        font-weight: 700 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #1e293b;
        padding: 5px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #334155;
        color: #cbd5e1;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #f59e0b !important;
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #dc2626 0%, #f59e0b 100%) !important;
        color: white !important;
        font-weight: 800 !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 6px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(220, 38, 38, 0.4) !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid #475569 !important;
        border-radius: 8px !important;
    }
    
    /* Input Fields */
    .stNumberInput input, .stSelectbox select, .stSlider {
        background-color: #1e293b !important;
        color: white !important;
        border: 1px solid #475569 !important;
        border-radius: 6px !important;
    }
    
    /* Success/Info/Warning/Error Messages */
    .stAlert {
        background-color: rgba(30, 41, 59, 0.9) !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 0 8px 8px 0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1e293b !important;
        color: #fbbf24 !important;
        font-weight: 700 !important;
        border: 1px solid #475569 !important;
        border-radius: 6px !important;
    }
    
    /* Plot Container */
    .stPlotlyChart, .stPyplot {
        background-color: #1e293b !important;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #475569;
    }
    
    /* Cards */
    .css-1r6slb0 {
        background-color: rgba(30, 41, 59, 0.7) !important;
        border: 1px solid #475569 !important;
        border-radius: 10px !important;
        padding: 20px !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #dc2626 0%, #f59e0b 100%) !important;
    }
    
    /* Markdown Text */
    .stMarkdown {
        color: #e2e8f0 !important;
    }
    
    /* Custom Class untuk Highlight */
    .highlight-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
        margin: 10px 0;
    }
    
    /* Custom Badge */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        background: linear-gradient(90deg, #dc2626 0%, #f59e0b 100%);
        color: white;
        font-weight: 700;
        border-radius: 4px;
        font-size: 0.8em;
        margin-right: 5px;
    }
    
    /* Theme Toggle Button */
    .theme-toggle {
        background: linear-gradient(90deg, #dc2626 0%, #f59e0b 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 10px auto;
    }
    
    .theme-toggle:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(220, 38, 38, 0.4);
    }
</style>
"""

light_theme = """
<style>
    /* Warna Utama: Light Blue, White, Gold Accents */
    :root {
        --primary: #f8fafc;
        --secondary: #e2e8f0;
        --accent: #f59e0b;
        --accent2: #dc2626;
        --text: #0f172a;
        --text-light: #475569;
    }
    
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: rgba(248, 250, 252, 0.95) !important;
        border-right: 2px solid #f59e0b;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #0f172a !important;
        font-weight: 800 !important;
        border-bottom: 2px solid #f59e0b;
        padding-bottom: 5px;
        margin-bottom: 15px !important;
    }
    
    /* Metrics Cards */
    div[data-testid="stMetricValue"] {
        color: #dc2626 !important;
        font-size: 24px !important;
        font-weight: 800 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #475569 !important;
        font-weight: 600 !important;
    }
    
    div[data-testid="stMetricDelta"] {
        font-weight: 700 !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #e2e8f0;
        padding: 5px;
        border-radius: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #cbd5e1;
        color: #475569;
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #f59e0b !important;
        color: #0f172a !important;
        font-weight: 800 !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(90deg, #dc2626 0%, #f59e0b 100%) !important;
        color: white !important;
        font-weight: 800 !important;
        border: none !important;
        padding: 10px 24px !important;
        border-radius: 6px !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(220, 38, 38, 0.4) !important;
    }
    
    /* Dataframes */
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
    }
    
    /* Input Fields */
    .stNumberInput input, .stSelectbox select, .stSlider {
        background-color: white !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
    }
    
    /* Success/Info/Warning/Error Messages */
    .stAlert {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 0 8px 8px 0 !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background-color: white !important;
        color: #dc2626 !important;
        font-weight: 700 !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 6px !important;
    }
    
    /* Plot Container */
    .stPlotlyChart, .stPyplot {
        background-color: white !important;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #cbd5e1;
    }
    
    /* Cards */
    .css-1r6slb0 {
        background-color: rgba(255, 255, 255, 0.9) !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 20px !important;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #dc2626 0%, #f59e0b 100%) !important;
    }
    
    /* Markdown Text */
    .stMarkdown {
        color: #0f172a !important;
    }
    
    /* Custom Class untuk Highlight */
    .highlight-box {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(220, 38, 38, 0.1) 100%);
        padding: 20px;
        border-radius: 10px;
        border-left: 4px solid #f59e0b;
        margin: 10px 0;
    }
    
    /* Custom Badge */
    .badge {
        display: inline-block;
        padding: 3px 10px;
        background: linear-gradient(90deg, #dc2626 0%, #f59e0b 100%);
        color: white;
        font-weight: 700;
        border-radius: 4px;
        font-size: 0.8em;
        margin-right: 5px;
    }
    
    /* Theme Toggle Button */
    .theme-toggle {
        background: linear-gradient(90deg, #dc2626 0%, #f59e0b 100%);
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: 700;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 10px auto;
    }
    
    .theme-toggle:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(220, 38, 38, 0.4);
    }
</style>
"""

# Apply theme based on session state
if st.session_state.theme == 'dark':
    st.markdown(dark_theme, unsafe_allow_html=True)
else:
    st.markdown(light_theme, unsafe_allow_html=True)

# Header dengan toggle theme
col1, col2, col3 = st.columns([3, 1, 1])

with col1:
    if st.session_state.theme == 'dark':
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 42px; margin-bottom: 10px; background: linear-gradient(90deg, #fbbf24 0%, #dc2626 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                ⚡ PREDIKSI KEPADATAN PENDUDUK
            </h1>
            <h3 style="color: #94a3b8; font-weight: 600;">Regresi Berganda dengan StandardScaler - Analisis Data Komprehensif</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="font-size: 42px; margin-bottom: 10px; background: linear-gradient(90deg, #dc2626 0%, #f59e0b 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);">
                ⚡ PREDIKSI KEPADATAN PENDUDUK
            </h1>
            <h3 style="color: #475569; font-weight: 600;">Regresi Berganda dengan StandardScaler - Analisis Data Komprehensif</h3>
        </div>
        """, unsafe_allow_html=True)

with col3:
    # Theme toggle button
    theme_icon = "🌙" if st.session_state.theme == 'dark' else "☀️"
    theme_text = "Light Mode" if st.session_state.theme == 'dark' else "Dark Mode"
    
    if st.button(f"{theme_icon} {theme_text}", key="theme_toggle", on_click=toggle_theme):
        pass

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

# Sidebar dengan style sesuai tema
with st.sidebar:
    # Theme indicator
    if st.session_state.theme == 'dark':
        st.markdown("""
        <div style="text-align: center; padding: 10px; background: rgba(245, 158, 11, 0.1); border-radius: 10px; border: 1px solid #f59e0b; margin-bottom: 20px;">
            <h3 style="color: #fbbf24; margin: 0;">⚙️ KONTROL MODEL</h3>
            <p style="color: #94a3b8; margin: 5px 0 0 0; font-size: 12px;">Mode: <strong>Dark</strong> 🌙</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align: center; padding: 10px; background: rgba(245, 158, 11, 0.1); border-radius: 10px; border: 1px solid #f59e0b; margin-bottom: 20px;">
            <h3 style="color: #0f172a; margin: 0;">⚙️ KONTROL MODEL</h3>
            <p style="color: #475569; margin: 5px 0 0 0; font-size: 12px;">Mode: <strong>Light</strong> ☀️</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### **🔧 PENGATURAN**")
    test_size = st.slider("**Ukuran Data Testing (%)**", 10, 40, 20, help="Persentase data yang digunakan untuk testing")
    
    st.markdown("---")
    
    # Statistik dataset
    st.markdown("### **📊 STATISTIK DATASET**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("📁 Total Data", f"{len(df):,}")
        st.metric("📍 Kecamatan", df['Kecamatan'].nunique())
    with col2:
        st.metric("📅 Rentang Tahun", f"{df['Tahun'].min()}-{df['Tahun'].max()}")
        st.metric("🎯 Fitur", len(features))

# Informasi utama dengan tema adaptif
if st.session_state.theme == 'dark':
    st.markdown(f"""
    <div class="highlight-box">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 24px;">📊</span>
            <div>
                <h4 style="color: #fbbf24; margin: 0;">ANALISIS DATA KOMPREHENSIF</h4>
                <p style="margin: 5px 0 0 0; color: #e2e8f0;">
                    Menggunakan <span class="badge">SEMUA DATA</span> dari tahun <strong>{df['Tahun'].min()}</strong> hingga <strong>{df['Tahun'].max()}</strong>
                    | Total observasi: <strong>{len(df):,}</strong>
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="highlight-box">
        <div style="display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 24px;">📊</span>
            <div>
                <h4 style="color: #0f172a; margin: 0;">ANALISIS DATA KOMPREHENSIF</h4>
                <p style="margin: 5px 0 0 0; color: #0f172a;">
                    Menggunakan <span class="badge">SEMUA DATA</span> dari tahun <strong>{df['Tahun'].min()}</strong> hingga <strong>{df['Tahun'].max()}</strong>
                    | Total observasi: <strong>{len(df):,}</strong>
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Split data
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

# Calculate metrics
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

# Main content - 3 tabs dengan style sesuai tema
tab1, tab2, tab3 = st.tabs(["📊 **EVALUASI MODEL**", "🔍 **KOEFISIEN REGRESI**", "🎯 **PREDIKSI MANUAL**"])

with tab1:
    # Header sesuai tema
    if st.session_state.theme == 'dark':
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: #fbbf24; margin: 0;">📈 METRIK PERFORMANSI MODEL</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: #0f172a; margin: 0;">📈 METRIK PERFORMANSI MODEL</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Metrik dalam 4 kolom dengan style sesuai tema
    col1, col2, col3, col4 = st.columns(4)
    
    # Warna text berdasarkan tema
    text_color = "#fbbf24" if st.session_state.theme == 'dark' else "#dc2626"
    bg_color = "rgba(30, 41, 59, 0.8)" if st.session_state.theme == 'dark' else "rgba(255, 255, 255, 0.9)"
    border_color = "#475569" if st.session_state.theme == 'dark' else "#cbd5e1"
    
    with col1:
        st.markdown(f"""
        <div style="background: {bg_color}; padding: 15px; border-radius: 10px; border-left: 4px solid #dc2626; border: 1px solid {border_color};">
            <h4 style="color: #94a3b8; margin: 0 0 10px 0;">MAE</h4>
            <h2 style="color: {text_color}; margin: 0; font-size: 28px;">{mae:.2f}</h2>
            <p style="color: #64748b; font-size: 12px; margin: 5px 0 0 0;">Mean Absolute Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div style="background: {bg_color}; padding: 15px; border-radius: 10px; border-left: 4px solid #f59e0b; border: 1px solid {border_color};">
            <h4 style="color: #94a3b8; margin: 0 0 10px 0;">RMSE</h4>
            <h2 style="color: {text_color}; margin: 0; font-size: 28px;">{rmse:.2f}</h2>
            <p style="color: #64748b; font-size: 12px; margin: 5px 0 0 0;">Root Mean Squared Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div style="background: {bg_color}; padding: 15px; border-radius: 10px; border-left: 4px solid #10b981; border: 1px solid {border_color};">
            <h4 style="color: #94a3b8; margin: 0 0 10px 0;">R² SCORE</h4>
            <h2 style="color: {text_color}; margin: 0; font-size: 28px;">{r2:.4f}</h2>
            <p style="color: #64748b; font-size: 12px; margin: 5px 0 0 0;">Koefisien Determinasi</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div style="background: {bg_color}; padding: 15px; border-radius: 10px; border-left: 4px solid #8b5cf6; border: 1px solid {border_color};">
            <h4 style="color: #94a3b8; margin: 0 0 10px 0;">MSE</h4>
            <h2 style="color: {text_color}; margin: 0; font-size: 28px;">{mse:.2f}</h2>
            <p style="color: #64748b; font-size: 12px; margin: 5px 0 0 0;">Mean Squared Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Interpretasi R² Score dengan style visual
    st.markdown("### 📊 INTERPRETASI MODEL")
    
    # Progress bar untuk R² Score
    r2_percent = r2 * 100
    if r2 >= 0.9:
        color = "#10b981"
        status = "SANGAT BAIK"
        icon = "✅"
    elif r2 >= 0.7:
        color = "#3b82f6"
        status = "BAIK"
        icon = "📈"
    elif r2 >= 0.5:
        color = "#f59e0b"
        status = "CUKUP"
        icon = "⚠️"
    else:
        color = "#dc2626"
        status = "PERLU PERBAIKAN"
        icon = "🔧"
    
    st.markdown(f"""
    <div style="background: {bg_color}; padding: 20px; border-radius: 10px; border: 1px solid {color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h4 style="color: {text_color}; margin: 0;">{icon} <span style="color: {color};">{status}</span></h4>
            <span style="color: {text_color}; font-weight: 800; font-size: 18px;">R² = {r2:.4f}</span>
        </div>
        <div style="background: #334155; height: 20px; border-radius: 10px; overflow: hidden; margin: 10px 0;">
            <div style="background: linear-gradient(90deg, {color} 0%, {color}80 100%); width: {r2_percent}%; height: 100%; border-radius: 10px;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; color: #94a3b8; font-size: 12px;">
            <span>0%</span>
            <span>Model menjelaskan {r2_percent:.1f}% variasi data</span>
            <span>100%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualisasi prediksi vs aktual dengan tema sesuai mode
    st.markdown("### 📉 VISUALISASI HASIL PREDIKSI")
    
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    # Set background color berdasarkan tema
    if st.session_state.theme == 'dark':
        fig1.patch.set_facecolor('#0f172a')
        ax1.set_facecolor('#1e293b')
        text_color_plot = '#e2e8f0'
        grid_color = '#475569'
        spine_color = '#475569'
    else:
        fig1.patch.set_facecolor('#f8fafc')
        ax1.set_facecolor('#ffffff')
        text_color_plot = '#0f172a'
        grid_color = '#cbd5e1'
        spine_color = '#cbd5e1'
    
    # Scatter plot dengan color map yang gagah
    scatter = ax1.scatter(y_test, y_pred_test, alpha=0.8, s=80, 
                         c=df.iloc[y_test.index]['Tahun'], 
                         cmap='YlOrRd', edgecolors='white' if st.session_state.theme == 'dark' else 'black', 
                         linewidth=0.5)
    
    # Garis ideal
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 
            color='#f59e0b', linestyle='--', linewidth=3, 
            label='Garis Ideal (y=x)', alpha=0.8)
    
    # Styling
    ax1.set_xlabel('Nilai Aktual', color=text_color_plot, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Nilai Prediksi', color=text_color_plot, fontsize=12, fontweight='bold')
    ax1.set_title('Perbandingan Prediksi vs Aktual (Semua Tahun)', 
                 color='#f59e0b', fontsize=14, fontweight='bold', pad=20)
    
    ax1.tick_params(colors=text_color_plot)
    ax1.grid(True, alpha=0.2, color=grid_color)
    ax1.spines['bottom'].set_color(spine_color)
    ax1.spines['top'].set_color(spine_color)
    ax1.spines['left'].set_color(spine_color)
    ax1.spines['right'].set_color(spine_color)
    
    ax1.legend(facecolor=bg_color, edgecolor=spine_color, 
               labelcolor=text_color_plot, fontsize=10)
    
    # Colorbar dengan styling
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Tahun', color=text_color_plot, fontsize=10, fontweight='bold')
    cbar.ax.yaxis.set_tick_params(color=text_color_plot)
    cbar.outline.set_edgecolor(spine_color)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=text_color_plot)
    
    st.pyplot(fig1, use_container_width=True)

with tab2:
    # Header sesuai tema
    if st.session_state.theme == 'dark':
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: #fbbf24; margin: 0;">🔍 ANALISIS KOEFISIEN REGRESI</h3>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: rgba(245, 158, 11, 0.05); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: #0f172a; margin: 0;">🔍 ANALISIS KOEFISIEN REGRESI</h3>
        </div>
        """, unsafe_allow_html=True)
    
    # Koefisien dalam cards
    coefficients = pd.DataFrame({
        'Fitur': features,
        'Koefisien': model.coef_,
        'Pengaruh': ['Meningkatkan' if c > 0 else 'Menurunkan' for c in model.coef_],
        'Besaran': np.abs(model.coef_)
    }).sort_values('Besaran', ascending=False)
    
    # Tampilkan koefisien dalam grid cards
    st.markdown("### 📊 PENGARUH FITUR TERHADAP KEPADATAN")
    
    cols = st.columns(len(features))
    for idx, (_, row) in enumerate(coefficients.iterrows()):
        with cols[idx]:
            color = "#10b981" if row['Koefisien'] > 0 else "#dc2626"
            icon = "📈" if row['Koefisien'] > 0 else "📉"
            
            st.markdown(f"""
            <div style="background: {bg_color}; padding: 15px; border-radius: 10px; border-top: 4px solid {color}; height: 150px; border: 1px solid {border_color};">
                <div style="font-size: 24px; margin-bottom: 10px;">{icon}</div>
                <h4 style="color: {text_color}; margin: 0 0 5px 0; font-size: 14px;">{row['Fitur'].split(' ')[0]}</h4>
                <p style="color: {color}; font-weight: 800; font-size: 18px; margin: 5px 0;">{row['Koefisien']:.3f}</p>
                <p style="color: #94a3b8; font-size: 12px; margin: 0;">{row['Pengaruh']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Visualisasi koefisien dengan tema sesuai mode
    st.markdown("### 📉 VISUALISASI KOEFISIEN")
    
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    
    # Set background color berdasarkan tema
    if st.session_state.theme == 'dark':
        fig2.patch.set_facecolor('#0f172a')
        ax2.set_facecolor('#1e293b')
        text_color_plot = '#e2e8f0'
        grid_color = '#475569'
        spine_color = '#475569'
    else:
        fig2.patch.set_facecolor('#f8fafc')
        ax2.set_facecolor('#ffffff')
        text_color_plot = '#0f172a'
        grid_color = '#cbd5e1'
        spine_color = '#cbd5e1'
    
    # Bar plot dengan warna berdasarkan nilai
    colors = ['#10b981' if c > 0 else '#dc2626' for c in coefficients['Koefisien']]
    bars = ax2.barh(coefficients['Fitur'], coefficients['Koefisien'], color=colors, height=0.6)
    
    # Styling
    ax2.set_xlabel('Nilai Koefisien', color=text_color_plot, fontsize=12, fontweight='bold')
    ax2.set_title('Pengaruh Fitur terhadap Kepadatan Penduduk', 
                 color='#f59e0b', fontsize=14, fontweight='bold', pad=20)
    
    ax2.tick_params(colors=text_color_plot)
    ax2.grid(True, alpha=0.2, color=grid_color, axis='x')
    ax2.axvline(x=0, color='#f59e0b', linestyle='-', linewidth=2, alpha=0.5)
    
    # Remove spines
    for spine in ax2.spines.values():
        spine.set_edgecolor(spine_color)
    
    # Tambahkan nilai pada bar
    for bar in bars:
        width = bar.get_width()
        ha = 'left' if width > 0 else 'right'
        x_pos = width + (0.01 if width > 0 else -0.01)
        bar_color = '#10b981' if width > 0 else '#dc2626'
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', 
                ha=ha, va='center', 
                color=bar_color, fontweight='bold', fontsize=10)
    
    st.pyplot(fig2, use_container_width=True)
    
    # Interpretasi fitur paling berpengaruh
    top_feature = coefficients.iloc[0]
    
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, rgba(245, 158, 11, 0.1) 0%, rgba(30, 41, 59, 0.8) 100%); 
                padding: 20px; border-radius: 10px; border-left: 4px solid #f59e0b; margin-top: 20px;">
        <h4 style="color: {text_color}; margin: 0 0 10px 0;">🏆 FITUR PALING BERPENGARUH</h4>
        <div style="display: flex; align-items: center; gap: 15px;">
            <span style="font-size: 48px; color: #f59e0b;">📊</span>
            <div>
                <h3 style="color: {text_color}; margin: 0;">{top_feature['Fitur']}</h3>
                <p style="color: #94a3b8; margin: 5px 0;">
                    Koefisien: <span style="color: #f59e0b; font-weight: 800; font-size: 18px;">{top_feature['Koefisien']:.4f}</span>
                </p>
                <p style="color: {text_color}; margin: 0;">
                    Setiap peningkatan 1 standar deviasi pada fitur ini akan 
                    <span style="color: {'#10b981' if top_feature['Koefisien'] > 0 else '#dc2626'}; font-weight: 800;">
                    {top_feature['Pengaruh'].lower()}
                    </span> kepadatan penduduk sebesar <strong>{abs(top_feature['Koefisien']):.2f}</strong> orang/km²
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

