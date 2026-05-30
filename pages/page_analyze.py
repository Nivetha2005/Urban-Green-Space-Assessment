# === FILE: pages/page_analyze.py ===
import os
import streamlit as st
import numpy as np
import torch
from PIL import Image
import time
import cv2

# Import utilities
from utils.maps import fetch_images_for_location, fetch_images_for_location_osm_fallback
from utils.superres import apply_sr_if_needed
from utils.inference import run_unet, run_yolo
from utils.analysis import (
    run_mc_dropout,
    compute_discrepancy,
    compute_uncertainty_weighted_fusion,
    detect_hotspots,
    compute_equity_index,
    compute_gradcam,
    simulate_winter,
    compute_cooling
)
from utils.visualizations import (
    plot_segmentation_overlay,
    plot_uncertainty_heatmap,
    plot_equity_map,
    plot_gradcam,
    plot_seasonal,
    plot_cooling_map,
    plot_discrepancy,
    plot_hotspots,
    plot_fusion
)
from models.unet_model import CLASS_COLORS, CLASS_NAMES
from utils.pdf_report import generate_pdf_report

def show(unet_model, yolo_model, sr_model, device):
    """Main analysis page function"""
    
    st.markdown("## 📊 Green Space Analysis")
    st.markdown("Analyze urban green space from satellite and street-level imagery")
    
    # Initialize session state for this page if not exists
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
    
    # Input section with three columns
    st.markdown("### 📥 Input Images")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🛰️ Satellite/Aerial")
        sat_upload = st.file_uploader(
            "Upload satellite image (JPG/PNG)", 
            type=['jpg', 'jpeg', 'png'], 
            key='sat_upload'
        )
        if sat_upload is not None:
            st.session_state.aerial_image = Image.open(sat_upload).convert('RGB')
            st.session_state.location_mode = False
            st.success("✅ Satellite image uploaded")
    
    with col2:
        st.markdown("#### 📸 Street View")
        street_upload = st.file_uploader(
            "Upload street image (JPG/PNG)", 
            type=['jpg', 'jpeg', 'png'], 
            key='street_upload'
        )
        if street_upload is not None:
            st.session_state.street_image = Image.open(street_upload).convert('RGB')
            st.session_state.location_mode = False
            st.success("✅ Street image uploaded")
    
    with col3:
        st.markdown("#### 🌍 Or Enter Location")
        location_input = st.text_input(
            "Location name", 
            value=st.session_state.get('location_name', ''),
            placeholder="e.g., Central Park, New York"
        )
        
        col3_1, col3_2 = st.columns(2)
        with col3_1:
            fetch_google = st.button("🔍 Fetch (Google)", use_container_width=True)
        with col3_2:
            fetch_osm = st.button("🗺️ Fetch (OSM)", use_container_width=True)
        
        if fetch_google and location_input.strip():
            st.session_state.location_name = location_input.strip()
            google_api_key = os.getenv('GOOGLE_MAPS_API_KEY', '')
            
            if not google_api_key:
                st.error("⚠️ Google Maps API key not found. Please set GOOGLE_MAPS_API_KEY in .env file")
            else:
                try:
                    with st.spinner('🌍 Fetching satellite and street images from Google...'):
                        fetched = fetch_images_for_location(
                            location_input.strip(), 
                            google_api_key,
                            os.getenv('MAPILLARY_ACCESS_TOKEN', '')
                        )
                    st.session_state.aerial_image = fetched['satellite']
                    st.session_state.street_image = fetched['street']
                    st.session_state.location_mode = True
                    st.session_state.location_name = fetched['address'].split(',')[0]
                    st.success(f"✅ Fetched images for: {fetched['address']}")
                    
                    if fetched['street'] is None:
                        st.warning("⚠️ No street view available for this location")
                        
                except Exception as e:
                    st.error(f"❌ Could not fetch location data: {e}")
        
        if fetch_osm and location_input.strip():
            st.session_state.location_name = location_input.strip()
            try:
                with st.spinner('🌍 Fetching images from OpenStreetMap/Esri...'):
                    fetched = fetch_images_for_location_osm_fallback(
                        location_input.strip(),
                        os.getenv('MAPILLARY_ACCESS_TOKEN', '')
                    )
                st.session_state.aerial_image = fetched['satellite']
                st.session_state.street_image = fetched['street']
                st.session_state.location_mode = True
                st.session_state.location_name = fetched['address'].split(',')[0]
                st.success(f"✅ Fetched images for: {fetched['address']}")
                
                if fetched['street'] is None:
                    st.warning("⚠️ No street view available for this location")
                    
            except Exception as e:
                st.error(f"❌ Could not fetch location data: {e}")
    
    # Display disclaimer for auto-fetched images
    if st.session_state.location_mode and st.session_state.aerial_image is not None:
        st.info("ℹ️ Auto-fetched images may differ in resolution from training data. Results are indicative.")
    
    # Display uploaded/fetched images
    if st.session_state.aerial_image is not None or st.session_state.street_image is not None:
        st.markdown("### 🖼️ Input Images Preview")
        display_cols = st.columns(2)
        
        with display_cols[0]:
            if st.session_state.aerial_image is not None:
                st.image(st.session_state.aerial_image, caption='🛰️ Satellite/Aerial Image', use_container_width=True)
            else:
                st.info("No satellite image loaded")
        
        with display_cols[1]:
            if st.session_state.street_image is not None:
                st.image(st.session_state.street_image, caption='📸 Street View Image', use_container_width=True)
            else:
                st.info("No street image loaded (optional for analysis)")
    
    # Analysis button
    st.markdown("---")
    analyze_pressed = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    if analyze_pressed:
        try:
            if unet_model is None:
                st.error("❌ UNet model is unavailable")
                st.stop()
            
            if st.session_state.aerial_image is None:
                st.error("❌ Satellite/Aerial image is required for analysis")
                st.stop()
            
            aerial_img = st.session_state.aerial_image
            street_img = st.session_state.street_image
            
            with st.spinner("🔄 Processing images..."):
                try:
                    # Apply super-resolution if needed
                    progress_bar = st.progress(0)
                    st.info("🔧 Applying super-resolution enhancement...")
                    
                    sr_applied = False
                    if sr_model is not None:
                        aerial_img, sr_applied = apply_sr_if_needed(aerial_img, sr_model)
                        if street_img is not None:
                            street_img, _ = apply_sr_if_needed(street_img, sr_model)
                    
                    if sr_applied:
                        st.success("✨ Super-resolution applied to low-resolution images")
                    
                    progress_bar.progress(20)
                    # ... rest of the code until the exception ...
                    
                except Exception as e:
                    st.error(f"❌ Pipeline failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                            
                # UNet inference
                st.info("🖼️ Running semantic segmentation on satellite image...")
                pred_mask, pred_rgb = run_unet(unet_model, aerial_img, device)
                progress_bar.progress(40)
                
                # MC Dropout uncertainty
                st.info("📊 Calculating prediction uncertainty (MC Dropout)...")
                resized_image = aerial_img.resize((256, 256), Image.BILINEAR)
                arr = np.array(resized_image).astype(np.float32) / 255.0
                image_tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
                
                mc_results = run_mc_dropout(unet_model, image_tensor, device, n_passes=20)
                mean_pred = mc_results['mean_pred']
                entropy_map = mc_results['entropy_map']
                gvi_mean = mc_results['gvi_mean']
                gvi_ci_lower = mc_results['gvi_ci_lower']
                gvi_ci_upper = mc_results['gvi_ci_upper']
                progress_bar.progress(60)
                
                # Resize to original dimensions
                orig_np = np.array(aerial_img)
                
                mean_pred_full = cv2.resize(
                    mean_pred.astype(np.uint8),
                    (orig_np.shape[1], orig_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                
                entropy_map_full = cv2.resize(
                    entropy_map.astype(np.float32),
                    (orig_np.shape[1], orig_np.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
                
                aerial_gvi = ((mean_pred == 2).sum() + (mean_pred == 3).sum()) / (mean_pred.size + 1e-8) * 100
                
                # YOLO street GVI
                street_gvi = 0.0
                street_annot = None
                if yolo_model is not None and street_img is not None:
                    st.info("🚗 Running street-level vegetation detection...")
                    street_gvi, street_annot = run_yolo(yolo_model, street_img)
                else:
                    if street_img is None:
                        st.info("ℹ️ No street image provided - skipping street-level GVI")
                
                progress_bar.progress(70)
                
                # All analysis functions
                st.info("🔬 Computing advanced metrics...")
                disc_results = compute_discrepancy(float(aerial_gvi), float(street_gvi))
                fusion_results = compute_uncertainty_weighted_fusion(float(aerial_gvi), float(street_gvi), entropy_map)
                hotspot_results = detect_hotspots(mean_pred, entropy_map, float(aerial_gvi), float(street_gvi))
                equity_results = compute_equity_index(mean_pred)
                gradcam_results = compute_gradcam(unet_model, image_tensor, device)
                season_results = simulate_winter(unet_model, aerial_img, mean_pred, device)
                cooling_results = compute_cooling(mean_pred)
                progress_bar.progress(90)
                
                # Resize maps for visualization
                gvi_map_full = cv2.resize(
                    equity_results['gvi_map'].astype(np.float32),
                    (orig_np.shape[1], orig_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                
                cooling_map_full = cv2.resize(
                    cooling_results['cooling_map'].astype(np.float32),
                    (orig_np.shape[1], orig_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                
                # Generate all visualizations
                st.info("📈 Generating visualizations...")
                fig_seg = plot_segmentation_overlay(orig_np, mean_pred_full, CLASS_COLORS, CLASS_NAMES)
                fig_unc = plot_uncertainty_heatmap(orig_np, entropy_map_full, gvi_mean, gvi_ci_lower, gvi_ci_upper)
                fig_equity = plot_equity_map(orig_np, gvi_map_full, equity_results['patch_gvis'], equity_results['gini_coef'])
                fig_gradcam = plot_gradcam(orig_np, gradcam_results['cam_lowveg'], gradcam_results['cam_tree'])
                fig_season = plot_seasonal(orig_np, season_results['winter_image'], season_results['summer_gvi'], season_results['winter_gvi'], season_results['delta'])
                fig_cooling = plot_cooling_map(orig_np, cooling_map_full, cooling_results['city_avg_cooling'])
                fig_discrepancy = plot_discrepancy(float(aerial_gvi), float(street_gvi), disc_results['discrepancy'], disc_results['label'])
                fig_hotspots = plot_hotspots(orig_np, hotspot_results['flagged_patches'], hotspot_results['hotspot_scores'])
                fig_fusion = plot_fusion(float(aerial_gvi), float(street_gvi), fusion_results['fused_gvi'], fusion_results['aerial_weight'], fusion_results['street_weight'])
                
                progress_bar.progress(100)
                
                # Store results in session state
                st.session_state.analysis_results = {
                    'gvi_aerial': float(aerial_gvi),
                    'gvi_street': float(street_gvi),
                    'gvi_fused': float(fusion_results['fused_gvi']),
                    'discrepancy': float(disc_results['discrepancy']),
                    'discrepancy_label': disc_results['label'],
                    'discrepancy_recommendation': disc_results['recommendation'],
                    'gini': float(equity_results['gini_coef']),
                    'city_avg_cooling': float(cooling_results['city_avg_cooling']),
                    'cooling_label': cooling_results['cooling_label'],
                    'gvi_ci_lower': float(gvi_ci_lower),
                    'gvi_ci_upper': float(gvi_ci_upper),
                    'gvi_mean': float(gvi_mean),
                    'summer_gvi': float(season_results['summer_gvi']),
                    'winter_gvi': float(season_results['winter_gvi']),
                    'seasonal_delta': float(season_results['delta']),
                    'hotspot_count': len(hotspot_results['flagged_patches']),
                    'aerial_weight': float(fusion_results['aerial_weight']),
                    'street_weight': float(fusion_results['street_weight']),
                    'fig_segmentation': fig_seg,
                    'fig_uncertainty': fig_unc,
                    'fig_equity': fig_equity,
                    'fig_gradcam': fig_gradcam,
                    'fig_seasonal': fig_season,
                    'fig_cooling': fig_cooling,
                    'fig_discrepancy': fig_discrepancy,
                    'fig_hotspots': fig_hotspots,
                    'fig_fusion': fig_fusion,
                    'location_name': st.session_state.get('location_name', 'Uploaded Image'),
                    'analysis_mode': 'location' if st.session_state.location_mode else 'upload',
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                st.success("✅ Analysis complete!")
                st.balloons()
                
        except Exception as e:
            st.error(f"❌ Pipeline failed: {e}")
            import traceback
            st.code(traceback.format_exc())
    
    # Display results if available
    if st.session_state.analysis_results:
        data = st.session_state.analysis_results
        
        st.markdown("---")
        st.markdown("## 📊 Analysis Results")
        
        # Key metrics row
        st.markdown("### 🎯 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🌍 Aerial GVI", f"{data['gvi_aerial']:.1f}%")
        
        with col2:
            st.metric("📸 Street GVI", f"{data['gvi_street']:.1f}%")
        
        with col3:
            st.metric("🔄 Fused GVI", f"{data['gvi_fused']:.1f}%")
        
        with col4:
            st.metric("⚖️ Gini Coefficient", f"{data['gini']:.3f}")
        
        # Results tabs
        tabs = st.tabs([
            "🌿 GVI & Uncertainty",
            "🔄 Dual-View Fusion",
            "📍 Hotspot Detection",
            "⚖️ Equity Map",
            "🌡️ Cooling Potential",
            "🎯 Grad-CAM",
            "🍂 Seasonal Simulation"
        ])
        
        with tabs[0]:
            st.pyplot(data['fig_uncertainty'])
            st.info(f"📊 **Interpretation**: The GVI is {data['gvi_aerial']:.1f}% with a 95% confidence interval of [{data['gvi_ci_lower']:.1f}%, {data['gvi_ci_upper']:.1f}%]. The entropy heatmap shows areas where the model is uncertain about vegetation classification.")
        
        with tabs[1]:
            st.pyplot(data['fig_fusion'])
            st.markdown(f"""
            <div style="background: #e8f4fd; border-left: 4px solid #2196F3; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <strong>🔍 Discrepancy Analysis</strong><br>
                <strong>Value:</strong> {data['discrepancy']:+.1f}%<br>
                <strong>Classification:</strong> {data['discrepancy_label']}<br>
                <strong>Recommendation:</strong> {data['discrepancy_recommendation']}
            </div>
            """, unsafe_allow_html=True)
        
        with tabs[2]:
            st.pyplot(data['fig_hotspots'])
            st.markdown(f"""
            <div style="background: #fff3e0; border-left: 4px solid #ff9800; padding: 1rem; border-radius: 8px; margin: 1rem 0;">
                <strong>⚠️ Hotspot Zones Identified: {data['hotspot_count']}</strong><br>
                These zones combine high prediction uncertainty with significant aerial-street discrepancy.
                Prioritize field validation and targeted greening interventions here.
            </div>
            """, unsafe_allow_html=True)
        
        with tabs[3]:
            st.pyplot(data['fig_equity'])
            gini = data['gini']
            if gini < 0.3:
                equity_status = "Excellent - Green space is evenly distributed"
            elif gini < 0.5:
                equity_status = "Moderate - Some patches have less green coverage"
            else:
                equity_status = "Poor - Significant inequality in green space distribution"
            st.info(f"📊 **Equity Assessment**: {equity_status}")
        
        with tabs[4]:
            st.pyplot(data['fig_cooling'])
            cooling = data['city_avg_cooling']
            if cooling < 0.5:
                cooling_status = "Low cooling potential - Increase tree canopy"
            elif cooling < 1.0:
                cooling_status = "Moderate cooling potential - Good start, can improve"
            else:
                cooling_status = "High cooling potential - Excellent heat mitigation"
            st.info(f"🌡️ **Cooling Assessment**: {cooling_status}. Estimated temperature reduction: {cooling:.2f}°C")
        
        with tabs[5]:
            st.pyplot(data['fig_gradcam'])
            st.info("🎯 **Model Attention**: Red regions show where the model focuses to identify vegetation. This helps verify that the model is looking at relevant features (trees, grass, parks).")
        
        with tabs[6]:
            st.pyplot(data['fig_seasonal'])
            delta = data['seasonal_delta']
            if delta < -10:
                season_status = "⚠️ Significant seasonal variation - Consider evergreen species"
            elif delta < -5:
                season_status = "📉 Moderate seasonal variation - Some winter loss expected"
            else:
                season_status = "✅ Good year-round green visibility"
            st.info(f"🍂 **Seasonal Assessment**: {season_status}. Summer GVI: {data['summer_gvi']:.1f}%, Winter GVI: {data['winter_gvi']:.1f}%")
        
        # PDF Download
        st.markdown("---")
        st.markdown("### 📄 Generate Report")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            with st.spinner("Preparing PDF report..."):
                report_bytes = generate_pdf_report(data)

            st.download_button(
                label="📥 Download Full Report (PDF)",
                data=report_bytes,
                file_name=f"urban_gvi_report_{data['location_name'].replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

# For direct execution (if needed)
if __name__ == "__main__":
    # This allows the file to be run directly for testing
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from utils.inference import load_unet, load_yolo
    from utils.superres import load_sr_model
    
    @st.cache_resource
    def get_models():
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        unet = load_unet(os.path.join('models', 'unet_ISPRS_30epochs_complete.pth'), device)
        yolo = load_yolo(os.path.join('models', 'yolo_street.pt'))
        sr = load_sr_model()
        return unet, yolo, sr, device
    
    unet_model, yolo_model, sr_model, device = get_models()
    show(unet_model, yolo_model, sr_model, device)