# === FILE: pages/page_compare.py ===
import os
import streamlit as st
import numpy as np
import torch
from PIL import Image
import time
import pandas as pd

from utils.maps import fetch_images_for_location, fetch_images_for_location_osm_fallback
from utils.superres import apply_sr_if_needed
from utils.inference import run_unet, run_yolo
from utils.analysis import (
    run_mc_dropout,
    compute_discrepancy,
    compute_uncertainty_weighted_fusion,
    detect_hotspots,
    compute_equity_index,
    compute_cooling
)

def process_city(aerial_img, street_img, unet_model, yolo_model, sr_model, device, city_name="City"):
    """
    Process a single city's images and return metrics
    """
    if aerial_img is None:
        return None
    
    try:
        # Apply super-resolution if needed
        if sr_model is not None:
            aerial_img, _ = apply_sr_if_needed(aerial_img, sr_model)
            if street_img is not None:
                street_img, _ = apply_sr_if_needed(street_img, sr_model)
        
        # UNet inference
        pred_mask, _ = run_unet(unet_model, aerial_img, device)
        
        # Prepare for MC Dropout
        resized = aerial_img.resize((256, 256), Image.BILINEAR)
        arr = np.array(resized).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
        
        # MC Dropout
        mc = run_mc_dropout(unet_model, image_tensor, device, n_passes=20)
        mean_pred = mc['mean_pred']
        aerial_gvi = ((mean_pred == 2).sum() + (mean_pred == 3).sum()) / (mean_pred.size + 1e-8) * 100
        
        # YOLO street GVI
        if street_img is not None and yolo_model is not None:
            street_gvi, _ = run_yolo(yolo_model, street_img)
        else:
            street_gvi = 0.0
        
        # Compute metrics
        disc = compute_discrepancy(aerial_gvi, street_gvi)
        fusion = compute_uncertainty_weighted_fusion(aerial_gvi, street_gvi, mc['entropy_map'])
        equity = compute_equity_index(mean_pred)
        cooling = compute_cooling(mean_pred)
        hotspots = detect_hotspots(mean_pred, mc['entropy_map'], aerial_gvi, street_gvi)
        
        return {
            'name': city_name,
            'aerial_gvi': float(aerial_gvi),
            'street_gvi': float(street_gvi),
            'fused_gvi': float(fusion['fused_gvi']),
            'discrepancy': float(disc['discrepancy']),
            'discrepancy_label': disc['label'],
            'gini': float(equity['gini_coef']),
            'cooling': float(cooling['city_avg_cooling']),
            'cooling_label': cooling['cooling_label'],
            'uncertainty_width': float(mc['gvi_ci_upper'] - mc['gvi_ci_lower']),
            'hotspot_count': len(hotspots['flagged_patches']),
            'aerial_weight': float(fusion['aerial_weight']),
            'street_weight': float(fusion['street_weight'])
        }
    
    except Exception as e:
        st.error(f"Error processing {city_name}: {e}")
        return None

def show(unet_model, yolo_model, sr_model, device):
    """Main compare page function"""
    
    st.markdown("## 🔄 City Comparison Tool")
    st.markdown("Compare green space metrics between two cities or locations")
    
    # Initialize session state
    if 'city_a_data' not in st.session_state:
        st.session_state.city_a_data = None
    if 'city_b_data' not in st.session_state:
        st.session_state.city_b_data = None
    
    # Create two columns for City A and City B
    col_a, col_b = st.columns(2, gap="large")
    
    # City A Section
    with col_a:
        st.markdown("### 🏙️ City / Location A")
        
        input_mode_a = st.radio(
            "Input method",
            ["Upload Images", "Search Location"],
            key="mode_a",
            horizontal=True
        )
        
        city_a_data = {'sat': None, 'street': None, 'name': 'City A'}
        
        if input_mode_a == "Upload Images":
            sat_a = st.file_uploader(
                "Satellite Image A", 
                type=['jpg', 'jpeg', 'png'], 
                key='sat_a'
            )
            street_a = st.file_uploader(
                "Street Image A", 
                type=['jpg', 'jpeg', 'png'], 
                key='street_a'
            )
            
            if sat_a is not None:
                city_a_data['sat'] = Image.open(sat_a).convert('RGB')
            if street_a is not None:
                city_a_data['street'] = Image.open(street_a).convert('RGB')
            
            city_a_data['name'] = st.text_input("City A Name", value="City A", key="name_a")
        
        else:  # Search Location
            location_a = st.text_input(
                "Location A", 
                placeholder="e.g., Tokyo, Japan",
                key="loc_a"
            )
            
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                fetch_google_a = st.button("🔍 Google Maps", key="fetch_a_google", use_container_width=True)
            with col_a2:
                fetch_osm_a = st.button("🗺️ OSM/Esri", key="fetch_a_osm", use_container_width=True)
            
            google_api_key = os.getenv('GOOGLE_MAPS_API_KEY', '')
            
            if fetch_google_a and location_a:
                if not google_api_key:
                    st.error("⚠️ Google API key not set")
                else:
                    with st.spinner(f"Fetching images for {location_a}..."):
                        try:
                            fetched = fetch_images_for_location(
                                location_a, 
                                google_api_key,
                                os.getenv('MAPILLARY_ACCESS_TOKEN', '')
                            )
                            city_a_data['sat'] = fetched['satellite']
                            city_a_data['street'] = fetched['street']
                            city_a_data['name'] = fetched['address'].split(',')[0]
                            st.success(f"✅ Loaded: {city_a_data['name']}")
                            if fetched['street'] is None:
                                st.warning("⚠️ No street view available")
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            if fetch_osm_a and location_a:
                with st.spinner(f"Fetching from OSM for {location_a}..."):
                    try:
                        fetched = fetch_images_for_location_osm_fallback(
                            location_a,
                            os.getenv('MAPILLARY_ACCESS_TOKEN', '')
                        )
                        city_a_data['sat'] = fetched['satellite']
                        city_a_data['street'] = fetched['street']
                        city_a_data['name'] = fetched['address'].split(',')[0]
                        st.success(f"✅ Loaded: {city_a_data['name']}")
                        if fetched['street'] is None:
                            st.warning("⚠️ No street view available")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # Display preview
        if city_a_data['sat'] is not None:
            st.image(city_a_data['sat'], caption=f"📍 {city_a_data['name']} - Satellite", use_container_width=True)
        if city_a_data['street'] is not None:
            st.image(city_a_data['street'], caption=f"📍 {city_a_data['name']} - Street", use_container_width=True)
    
    # City B Section
    with col_b:
        st.markdown("### 🌆 City / Location B")
        
        input_mode_b = st.radio(
            "Input method",
            ["Upload Images", "Search Location"],
            key="mode_b",
            horizontal=True
        )
        
        city_b_data = {'sat': None, 'street': None, 'name': 'City B'}
        
        if input_mode_b == "Upload Images":
            sat_b = st.file_uploader(
                "Satellite Image B", 
                type=['jpg', 'jpeg', 'png'], 
                key='sat_b'
            )
            street_b = st.file_uploader(
                "Street Image B", 
                type=['jpg', 'jpeg', 'png'], 
                key='street_b'
            )
            
            if sat_b is not None:
                city_b_data['sat'] = Image.open(sat_b).convert('RGB')
            if street_b is not None:
                city_b_data['street'] = Image.open(street_b).convert('RGB')
            
            city_b_data['name'] = st.text_input("City B Name", value="City B", key="name_b")
        
        else:  # Search Location
            location_b = st.text_input(
                "Location B", 
                placeholder="e.g., Berlin, Germany",
                key="loc_b"
            )
            
            col_b1, col_b2 = st.columns(2)
            with col_b1:
                fetch_google_b = st.button("🔍 Google Maps", key="fetch_b_google", use_container_width=True)
            with col_b2:
                fetch_osm_b = st.button("🗺️ OSM/Esri", key="fetch_b_osm", use_container_width=True)
            
            google_api_key = os.getenv('GOOGLE_MAPS_API_KEY', '')
            
            if fetch_google_b and location_b:
                if not google_api_key:
                    st.error("⚠️ Google API key not set")
                else:
                    with st.spinner(f"Fetching images for {location_b}..."):
                        try:
                            fetched = fetch_images_for_location(
                                location_b, 
                                google_api_key,
                                os.getenv('MAPILLARY_ACCESS_TOKEN', '')
                            )
                            city_b_data['sat'] = fetched['satellite']
                            city_b_data['street'] = fetched['street']
                            city_b_data['name'] = fetched['address'].split(',')[0]
                            st.success(f"✅ Loaded: {city_b_data['name']}")
                            if fetched['street'] is None:
                                st.warning("⚠️ No street view available")
                        except Exception as e:
                            st.error(f"Error: {e}")
            
            if fetch_osm_b and location_b:
                with st.spinner(f"Fetching from OSM for {location_b}..."):
                    try:
                        fetched = fetch_images_for_location_osm_fallback(
                            location_b,
                            os.getenv('MAPILLARY_ACCESS_TOKEN', '')
                        )
                        city_b_data['sat'] = fetched['satellite']
                        city_b_data['street'] = fetched['street']
                        city_b_data['name'] = fetched['address'].split(',')[0]
                        st.success(f"✅ Loaded: {city_b_data['name']}")
                        if fetched['street'] is None:
                            st.warning("⚠️ No street view available")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        # Display preview
        if city_b_data['sat'] is not None:
            st.image(city_b_data['sat'], caption=f"📍 {city_b_data['name']} - Satellite", use_container_width=True)
        if city_b_data['street'] is not None:
            st.image(city_b_data['street'], caption=f"📍 {city_b_data['name']} - Street", use_container_width=True)
    
    # Compare Button
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        compare_btn = st.button("🔄 Compare Cities", type="primary", use_container_width=True)
    
    if compare_btn:
        if city_a_data['sat'] is None or city_b_data['sat'] is None:
            st.error("❌ Please provide satellite images for both cities")
        else:
            with st.spinner("Processing both cities..."):
                # Process both cities
                res_a = process_city(
                    city_a_data['sat'], city_a_data['street'],
                    unet_model, yolo_model, sr_model, device,
                    city_a_data['name']
                )
                res_b = process_city(
                    city_b_data['sat'], city_b_data['street'],
                    unet_model, yolo_model, sr_model, device,
                    city_b_data['name']
                )
            
            if res_a is None or res_b is None:
                st.error("❌ Failed to process one or both cities")
            else:
                st.session_state.city_a_data = res_a
                st.session_state.city_b_data = res_b
                
                # Display comparison results
                st.markdown("---")
                st.markdown("## 📊 Comparison Results")
                
                # Winner announcement
                winner = res_a['name'] if res_a['fused_gvi'] > res_b['fused_gvi'] else res_b['name']
                winner_gvi = max(res_a['fused_gvi'], res_b['fused_gvi'])
                
                st.markdown(f"""
                <div class="success-box" style="text-align: center;">
                    <h3>🏆 Higher Green Visibility: {winner}</h3>
                    <p style="font-size: 1.2rem;">Fused GVI: {winner_gvi:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Metrics table
                metrics = [
                    "Aerial GVI (%)",
                    "Street GVI (%)", 
                    "Fused GVI (%)",
                    "Discrepancy (%)",
                    "Gini Coefficient",
                    "Cooling Potential (°C)",
                    "Uncertainty Width",
                    "Hotspot Zones"
                ]
                
                values_a = [
                    res_a['aerial_gvi'],
                    res_a['street_gvi'],
                    res_a['fused_gvi'],
                    res_a['discrepancy'],
                    res_a['gini'],
                    res_a['cooling'],
                    res_a['uncertainty_width'],
                    res_a['hotspot_count']
                ]
                
                values_b = [
                    res_b['aerial_gvi'],
                    res_b['street_gvi'],
                    res_b['fused_gvi'],
                    res_b['discrepancy'],
                    res_b['gini'],
                    res_b['cooling'],
                    res_b['uncertainty_width'],
                    res_b['hotspot_count']
                ]
                
                # Determine winners for each metric
                better_a = []
                better_b = []
                
                # Define which direction is better for each metric
                better_direction = [
                    'higher',  # Aerial GVI
                    'higher',  # Street GVI
                    'higher',  # Fused GVI
                    'lower',   # Discrepancy (closer to zero is better)
                    'lower',   # Gini (lower inequality)
                    'higher',  # Cooling
                    'lower',   # Uncertainty width
                    'lower'    # Hotspot count
                ]
                
                for i, direction in enumerate(better_direction):
                    if direction == 'higher':
                        better_a.append(values_a[i] > values_b[i])
                        better_b.append(values_b[i] > values_a[i])
                    else:
                        better_a.append(values_a[i] < values_b[i])
                        better_b.append(values_b[i] < values_a[i])
                
                # Display as a styled table
                df_data = []
                for i, metric in enumerate(metrics):
                    df_data.append({
                        "Metric": metric,
                        res_a['name']: f"{values_a[i]:.1f}" if isinstance(values_a[i], float) else str(values_a[i]),
                        res_b['name']: f"{values_b[i]:.1f}" if isinstance(values_b[i], float) else str(values_b[i]),
                        "Better": res_a['name'] if better_a[i] else (res_b['name'] if better_b[i] else "Tie")
                    })
                
                df = pd.DataFrame(df_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
                
                # Detailed comparison visualizations
                st.markdown("### 📈 Metric Comparison Chart")
                
                # Prepare data for bar chart
                import matplotlib.pyplot as plt
                
                fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                
                # GVI Comparison
                ax1 = axes[0, 0]
                x = np.arange(3)
                width = 0.35
                gvi_a = [res_a['aerial_gvi'], res_a['street_gvi'], res_a['fused_gvi']]
                gvi_b = [res_b['aerial_gvi'], res_b['street_gvi'], res_b['fused_gvi']]
                ax1.bar(x - width/2, gvi_a, width, label=res_a['name'], color='#2E8B57')
                ax1.bar(x + width/2, gvi_b, width, label=res_b['name'], color='#4682B4')
                ax1.set_xticks(x)
                ax1.set_xticklabels(['Aerial', 'Street', 'Fused'])
                ax1.set_ylabel('GVI (%)')
                ax1.set_title('Green Visibility Index Comparison')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Equity & Cooling
                ax2 = axes[0, 1]
                x = np.arange(2)
                equity_a = [res_a['gini'], res_a['cooling']]
                equity_b = [res_b['gini'], res_b['cooling']]
                ax2.bar(x - width/2, equity_a, width, label=res_a['name'], color='#2E8B57')
                ax2.bar(x + width/2, equity_b, width, label=res_b['name'], color='#4682B4')
                ax2.set_xticks(x)
                ax2.set_xticklabels(['Gini (lower better)', 'Cooling °C'])
                ax2.set_title('Equity & Cooling Comparison')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Discrepancy
                ax3 = axes[1, 0]
                discrepancy_values = [res_a['discrepancy'], res_b['discrepancy']]
                colors = ['#FF6B6B' if abs(v) > 15 else '#4CAF50' for v in discrepancy_values]
                bars = ax3.bar([res_a['name'], res_b['name']], discrepancy_values, color=colors)
                ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
                ax3.axhline(y=15, color='orange', linestyle='--', label='Hidden Canopy Threshold')
                ax3.axhline(y=-15, color='purple', linestyle='--', label='Vertical Greenery Threshold')
                ax3.set_ylabel('Discrepancy (%)')
                ax3.set_title('Aerial-Street Discrepancy')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Uncertainty & Hotspots
                ax4 = axes[1, 1]
                x = np.arange(2)
                uncertainty_a = [res_a['uncertainty_width'], res_a['hotspot_count']]
                uncertainty_b = [res_b['uncertainty_width'], res_b['hotspot_count']]
                ax4.bar(x - width/2, uncertainty_a, width, label=res_a['name'], color='#2E8B57')
                ax4.bar(x + width/2, uncertainty_b, width, label=res_b['name'], color='#4682B4')
                ax4.set_xticks(x)
                ax4.set_xticklabels(['Uncertainty Width', 'Hotspot Count'])
                ax4.set_title('Uncertainty & Hotspots')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Recommendations based on comparison
                st.markdown("### 💡 Comparative Recommendations")
                
                rec_col1, rec_col2 = st.columns(2)
                
                with rec_col1:
                    st.markdown(f"#### 📍 {res_a['name']}")
                    if res_a['gini'] > res_b['gini']:
                        st.warning(f"Higher inequality than {res_b['name']} - Focus on equitable distribution")
                    if res_a['cooling'] < res_b['cooling']:
                        st.warning(f"Lower cooling potential - Increase tree canopy")
                    if res_a['hotspot_count'] > res_b['hotspot_count']:
                        st.warning(f"More intervention zones - Prioritize field validation")
                    if res_a['discrepancy'] > 15:
                        st.info("Hidden canopy detected - Improve street-level visibility")
                    elif res_a['discrepancy'] < -15:
                        st.info("Vertical greenery dominant - Balance with canopy")
                
                with rec_col2:
                    st.markdown(f"#### 📍 {res_b['name']}")
                    if res_b['gini'] > res_a['gini']:
                        st.warning(f"Higher inequality than {res_a['name']} - Focus on equitable distribution")
                    if res_b['cooling'] < res_a['cooling']:
                        st.warning(f"Lower cooling potential - Increase tree canopy")
                    if res_b['hotspot_count'] > res_a['hotspot_count']:
                        st.warning(f"More intervention zones - Prioritize field validation")
                    if res_b['discrepancy'] > 15:
                        st.info("Hidden canopy detected - Improve street-level visibility")
                    elif res_b['discrepancy'] < -15:
                        st.info("Vertical greenery dominant - Balance with canopy")