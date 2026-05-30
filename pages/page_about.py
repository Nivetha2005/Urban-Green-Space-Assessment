# === FILE: pages/page_about.py ===
import streamlit as st

def show():
    """Main about page function"""
    
    st.markdown("## 📖 About Urban Green Space Assessment System")
    
    # Hero section
    st.markdown("""
    <div style="background: #e8f5e9; border-left: 4px solid #2E8B57; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
        <h3 style="margin: 0; color: #2E8B57;">🌍 AI-Powered Urban Greenery Assessment</h3>
        <p style="margin-top: 0.5rem;">Combining satellite and street-level imagery for comprehensive green visibility analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Project Overview
    with st.expander("🎯 Project Overview", expanded=True):
        st.markdown("""
        The **Urban Green Space Assessment System** is a production-ready Streamlit application that provides 
        comprehensive urban vegetation analysis by integrating aerial (satellite) and street-level perspectives.
        
        **Key Capabilities:**
        - **Dual-View GVI Framework:** Combines UNet semantic segmentation (6 classes) with YOLO street scene analysis
        - **Uncertainty Quantification:** Monte Carlo Dropout with 20 stochastic passes for confidence intervals
        - **Discrepancy Mapping:** Quantifies hidden canopy vs vertical greenery dominance
        - **Equity Analysis:** Gini coefficient across 4x4 spatial patches
        - **Cooling Potential:** Estimates urban heat mitigation (Bowler et al. 2010)
        - **Explainable AI:** Grad-CAM visualizations for vegetation attention
        """)
    
    # The 6 Novelties
    with st.expander("✨ The 6 Key Innovations", expanded=True):
        st.markdown("""
        | # | Innovation | Description |
        |---|------------|-------------|
        | 1 | **Dual-View GVI Framework** | Aerial UNet + Street YOLO in unified pipeline |
        | 2 | **Discrepancy Mapping** | Quantifies hidden canopy vs vertical greenery |
        | 3 | **MC Dropout Uncertainty** | 20 stochastic passes, entropy maps, confidence intervals |
        | 4 | **Uncertainty-Weighted Fusion** | Blends views based on prediction entropy |
        | 5 | **Hotspot Detection** | 4x4 grid, top 25% scores for intervention |
        | 6 | **Supporting Analyses** | Equity Gini, Grad-CAM, seasonal simulation, cooling |
        """)
    
    # Dataset Information
    with st.expander("🗂️ Dataset Information", expanded=False):
        st.markdown("""
        **ISPRS Potsdam + Vaihingen Benchmarks**
        
        - **Images:** 71 true orthophotos (6000×6000 pixels)
        - **Resolution:** 5 cm per pixel
        - **Classes:** 6 semantic categories
        - **Tiling:** 256×256 patches for training
        
        **Class Definitions:**
        - **0: Impervious surfaces** - Roads, paved areas (RGB: 255,255,255)
        - **1: Building** - Residential, commercial structures (RGB: 0,0,255)
        - **2: Low vegetation** - Grass, shrubs (RGB: 0,255,255)
        - **3: Tree** - All tree canopy (RGB: 0,255,0)
        - **4: Car** - Vehicles (RGB: 255,255,0)
        - **5: Clutter** - Other objects (RGB: 255,0,0)
        """)
    
    # Model Architecture
    with st.expander("🧠 Model Architecture", expanded=False):
        st.markdown("""
        **UNet with Dropout (31M parameters)**
        
        ```
        Encoder:
        - DoubleConv(3→64) → MaxPool
        - DoubleConv(64→128) → MaxPool
        - DoubleConv(128→256) → MaxPool
        - DoubleConv(256→512) → MaxPool
        
        Bottleneck:
        - DoubleConv(512→1024) → Dropout2d(0.3)
        
        Decoder with Skip Connections:
        - UpConv(1024→512) → Concatenate → DoubleConv(1024→512)
        - UpConv(512→256) → Concatenate → DoubleConv(512→256)
        - UpConv(256→128) → Concatenate → DoubleConv(256→128)
        - UpConv(128→64) → Concatenate → DoubleConv(128→64)
        
        Output: Conv2d(64→6)
        ```
        
        **YOLOv11 (Cityscapes)**
        - Vegetation class (index 8)
        - Terrain class (index 9)
        - Segmentation masks + bounding boxes
        """)
    
    # Methodology
    with st.expander("📐 Methodology", expanded=False):
        st.markdown("""
        **Green Visibility Index (GVI)**
        ```
        GVI = (Class 2 pixels + Class 3 pixels) / Total pixels × 100
        ```
        
        **Discrepancy Thresholds**
        - **> 15%:** Hidden canopy (aerial sees more)
        - **< -15%:** Vertical greenery dominance (street sees more)
        - **-15% to 15%:** Consistent visibility
        
        **Gini Coefficient (Equity)**
        ```
        G = (2 × Σ(rank_i × gvi_i)) / (n × Σ(gvi_i)) - (n+1)/n
        ```
        Where 0 = perfect equality, 1 = perfect inequality
        
        **Cooling Potential (Bowler et al. 2010)**
        ```
        ΔT = (Patch GVI / 10) × 0.3°C
        ```
        
        **Uncertainty-Weighted Fusion**
        ```
        Normalized_Entropy = (Entropy - min) / (max - min)
        Street_Weight = mean(Normalized_Entropy)
        Aerial_Weight = 1 - Street_Weight
        Fused_GVI = Aerial_Weight × Aerial_GVI + Street_Weight × Street_GVI
        ```
        """)
    
    # Performance Metrics
    with st.expander("📊 Performance Metrics", expanded=False):
        st.markdown("""
        | Metric | Value | Description |
        |--------|-------|-------------|
        | **Validation Accuracy** | 69.6% | Overall pixel classification |
        | **mIoU** | 38.5% | Mean Intersection over Union |
        | **GVI Correlation** | 0.954 | Correlation with ground truth GVI |
        | **MC Dropout Passes** | 20 | For uncertainty estimation |
        | **Inference Time** | ~0.5s | Per image (GPU) |
        
        **Class-wise IoU:**
        - Impervious: 0.82
        - Building: 0.89
        - Low Vegetation: 0.71
        - Tree: 0.78
        - Car: 0.52
        - Clutter: 0.38
        """)
    
    # API Integrations
    with st.expander("🔌 API Integrations", expanded=False):
        st.markdown("""
        **Google Maps API** (Primary)
        - Geocoding: Convert location names to coordinates
        - Static Maps: Satellite imagery (zoom 18, 640×640)
        - Street View: Ground-level photography
        
        **OpenStreetMap/Nominatim** (Fallback)
        - Free geocoding without API key
        - Rate-limited but reliable
        
        **Esri World Imagery** (Satellite Fallback)
        - XYZ tile service
        - No API key required
        
        **Mapillary** (Street View Fallback)
        - Community-sourced street imagery
        - Requires free token
        
        **Hugging Face Inference API** (Assistant)
        - Free model inference (DialoGPT, etc.)
        - Optional API key for higher rate limits
        """)
    
    # Limitations
    with st.expander("⚠️ Limitations & Future Work", expanded=False):
        st.markdown("""
        **Current Limitations:**
        - Model trained on European datasets (ISPRS) - may not generalize globally
        - Seasonal simulation is heuristic (not physics-based)
        - Street view availability depends on API coverage
        - Resolution mismatch between training (5cm) and API images (varies)
        - Cooling coefficient is averaged - local factors may differ
        
        **Future Improvements:**
        - [ ] Multi-temporal analysis (track changes over time)
        - [ ] Integration with local climate data
        - [ ] 3D vegetation structure from stereo imagery
        - [ ] Mobile app for field validation
        - [ ] Community reporting integration
        - [ ] Real-time API for live monitoring
        """)
    
    # References
    with st.expander("📚 References", expanded=False):
        st.markdown("""
        **Academic References:**
        
        1. Bowler, D.E., Buyung-Ali, L., Knight, T.M., & Pullin, A.S. (2010). 
           *Urban greening to cool towns and cities: A systematic review of the empirical evidence.*
           Landscape and Urban Planning, 97(3), 147-155.
        
        2. Rottensteiner, F., Sohn, G., Gerke, M., & Wegner, J.D. (2014).
           *ISPRS Semantic Labeling Contest.*
           ISPRS Journal of Photogrammetry and Remote Sensing.
        
        3. Cordts, M., Omran, M., Ramos, S., et al. (2016).
           *The Cityscapes Dataset for Semantic Urban Scene Understanding.*
           CVPR 2016.
        
        4. Gal, Y., & Ghahramani, Z. (2016).
           *Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.*
           ICML 2016.
        
        5. Selvaraju, R.R., Cogswell, M., Das, A., et al. (2017).
           *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.*
           ICCV 2017.
        
        **Software & Models:**
        - Real-ESRGAN: Xintao Wang et al. (2021)
        - Ultralytics YOLO: Jocher et al. (2023)
        - Streamlit: Streamlit Open Source (2024)
        """)
    
    # Technical Stack
    with st.expander("🛠️ Technical Stack", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Deep Learning**")
            st.code("""
- PyTorch 2.0.1
- TorchVision 0.15.2
- Ultralytics 8.4.33
- Real-ESRGAN 0.3.0
            """)
        with col2:
            st.markdown("**Application**")
            st.code("""
- Streamlit 1.56.0
- OpenCV 4.8.1
- Pillow 12.2.0
- Matplotlib 3.10.8
- FPDF2 2.8.7
            """)
    
    # License
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        <p>© 2024 Urban Green Space Assessment System</p>
        <p>For research and urban planning purposes | MIT License</p>
        <p>Built with ❤️ for greener, more equitable cities</p>
    </div>
    """, unsafe_allow_html=True)

# For direct execution
if __name__ == "__main__":
    show()