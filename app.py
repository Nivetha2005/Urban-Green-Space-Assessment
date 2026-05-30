# === FILE: app.py ===
import os
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

# Load environment variables
load_dotenv()

# Page configuration MUST be the first Streamlit command
st.set_page_config(
    layout="wide",
    page_title="Urban GVI Assessment System",
    page_icon="🌿",
    initial_sidebar_state="expanded"
)

# Load custom CSS
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found: {file_name}")

local_css('assets/style.css')

# Initialize session state keys
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}
if 'aerial_image' not in st.session_state:
    st.session_state.aerial_image = None
if 'street_image' not in st.session_state:
    st.session_state.street_image = None
if 'location_mode' not in st.session_state:
    st.session_state.location_mode = False
if 'location_name' not in st.session_state:
    st.session_state.location_name = ''
if 'assistant_history' not in st.session_state:
    st.session_state.assistant_history = []

# Model loading functions with caching
@st.cache_resource
def get_unet_model():
    import torch
    from utils.inference import load_unet
    
    model_path = os.path.join('models', 'unet_ISPRS_30epochs_complete.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model = load_unet(model_path, device)
        return model, device
    except Exception as e:
        st.error(f"Failed to load UNet model: {e}")
        return None, device

@st.cache_resource
def get_yolo_model():
    from utils.inference import load_yolo
    
    model_path = os.path.join('models', 'yolo_street.pt')
    
    try:
        model = load_yolo(model_path)
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        return None

@st.cache_resource
def get_sr_model():
    from utils.superres import load_sr_model
    return load_sr_model()

# Load models
unet_model, device = get_unet_model()
yolo_model = get_yolo_model()
sr_model = get_sr_model()

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    
    st.image("https://img.icons8.com/color/96/000000/forest.png", width=80)
    st.markdown("## 🌿 Urban GVI")
    st.markdown("### Green Space Assessment System")
    st.markdown("---")
    
    st.markdown("#### 📊 Model Performance")
    st.metric("UNet Accuracy", "69.6%", "±2.1%")
    st.metric("mIoU", "38.5%", "±1.8%")
    st.metric("GVI Correlation", "0.954", "Strong")
    
    st.markdown("---")
    st.markdown("#### 🎯 Methodology")
    st.info(
        "• Dual-View GVI Fusion\n"
        "• MC Dropout Uncertainty\n"
        "• Discrepancy Mapping\n"
        "• Gini Equity Analysis\n"
        "• Urban Cooling Potential\n"
        "• Grad-CAM Explainability"
    )
    
    st.markdown("---")
    st.markdown("#### 🔌 API Status")
    
    google_key = os.getenv('GOOGLE_MAPS_API_KEY', '')
    if google_key:
        st.success("✅ Google Maps API")
    else:
        st.warning("⚠️ Google Maps API (set env var)")
    
    st.markdown("---")
    st.markdown("#### 📱 Navigation")
    st.markdown("Use the tabs above to navigate between pages:")
    st.markdown("1. **Analyze** - Main analysis")
    st.markdown("2. **Compare** - City comparison")
    st.markdown("3. **Assistant** - AI chat")
    st.markdown("4. **About** - Documentation")
    
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; font-size: 0.8rem;'>Device: {device.upper()}</p>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main content area with hero banner
st.markdown("""
<div class="hero-banner">
    <h1>🌍 Urban Green Space Assessment System</h1>
    <p>AI-powered analysis of urban vegetation from satellite and street-level perspectives</p>
    <p style="font-size: 0.9rem;">🚀 Dual-View GVI | 🎯 Uncertainty Quantification | 📊 Equity Analysis | 🌡️ Cooling Potential</p>
</div>
""", unsafe_allow_html=True)

# Navigation using tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Analyze", "🔄 Compare Cities", "🤖 Assistant", "📖 About"])

with tab1:
    from pages.page_analyze import show as analyze_show
    analyze_show(unet_model, yolo_model, sr_model, device)

with tab2:
    from pages.page_compare import show as compare_show
    compare_show(unet_model, yolo_model, sr_model, device)

with tab3:
    from pages.page_assistant import show as assistant_show
    assistant_show()

with tab4:
    from pages.page_about import show as about_show
    about_show()

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <p>🌿 Urban Green Space Assessment System | Powered by UNet + YOLO + Real-ESRGAN</p>
    <p>© 2026 | For research and urban planning purposes</p>
</div>
""", unsafe_allow_html=True)