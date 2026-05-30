# === FILE: utils/pdf_report.py ===
import io
from fpdf import FPDF
from PIL import Image
import textwrap
from datetime import datetime

def safe_multicell(pdf, text, width=190):
    safe_text = str(text)
    # Replace bullets
    safe_text = safe_text.replace("•", "-")
    # Remove emojis / unsupported unicode
    safe_text = safe_text.encode("ascii", "ignore").decode()
    # Remove newlines
    safe_text = safe_text.replace("\n", " ")
    wrapped = "\n".join(textwrap.wrap(safe_text, width=80))
    pdf.multi_cell(width, 6, wrapped)

def figure_to_png_bytes(fig):
    """Convert matplotlib figure to PNG bytes"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return buf.read()

def generate_pdf_report(results: dict) -> bytes:
    """Generate comprehensive PDF report from analysis results"""
    
    pdf = FPDF(format='A4')
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Extract data
    location = results.get('location_name', 'Unknown Location')
    mode = results.get('analysis_mode', 'Upload')
    timestamp = results.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    # ===== PAGE 1: TITLE PAGE =====
    pdf.add_page()
    
    # Header
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(46, 139, 87)  # Forest green
    pdf.cell(0, 20, 'Urban Green Space Assessment Report', ln=True, align='C')
    
    pdf.set_font('Helvetica', 'B', 16)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 15, f'Location: {location}', ln=True, align='C')
    
    pdf.set_font('Helvetica', '', 12)
    pdf.cell(0, 10, f'Analysis Mode: {mode}', ln=True, align='C')
    pdf.cell(0, 10, f'Report Generated: {timestamp}', ln=True, align='C')
    pdf.ln(15)
    
    # Model info
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Model Information', ln=True)
    pdf.set_font('Helvetica', '', 11)
    safe_multicell(pdf,
        '- UNet Architecture: 31M parameters, trained on ISPRS Potsdam + Vaihingen\n'
        '- YOLOv11: Trained on Cityscapes for street-level vegetation detection\n'
        '- Super-Resolution: Real-ESRGAN x4 for low-resolution inputs\n'
        '- Uncertainty: Monte Carlo Dropout with 20 stochastic forward passes')
    
    pdf.ln(10)
    
    # Methodology summary
    pdf.set_font('Helvetica', 'B', 14)
    pdf.cell(0, 10, 'Methodology Summary', ln=True)
    pdf.set_font('Helvetica', '', 11)
    safe_multicell(pdf,
        'Green Visibility Index (GVI) = (Low Vegetation + Tree pixels) / Total pixels × 100\n\n'
        'This report combines aerial (satellite) and street-level perspectives to assess:\n'
        '- Green space coverage and distribution\n'
        '- Spatial equity across urban patches\n'
        '- Urban cooling potential (Bowler et al. 2010 coefficient: 0.3°C per 10% GVI)\n'
        '- Model uncertainty and prediction confidence\n'
        '- Seasonal variation impact on vegetation visibility')
    
    # ===== PAGE 2: KEY METRICS =====
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 12, 'Key Performance Metrics', ln=True)
    pdf.ln(5)
    
    # Metrics table
    metrics = [
        ('Aerial GVI', f"{results.get('gvi_aerial', 0):.1f}%"),
        ('Street GVI', f"{results.get('gvi_street', 0):.1f}%"),
        ('Fused GVI', f"{results.get('gvi_fused', 0):.1f}%"),
        ('GVI Discrepancy', f"{results.get('discrepancy', 0):+.1f}%"),
        ('Discrepancy Classification', results.get('discrepancy_label', 'N/A')),
        ('Gini Coefficient', f"{results.get('gini', 0):.4f}"),
        ('City Average Cooling', f"{results.get('city_avg_cooling', 0):.2f}°C"),
        ('GVI 95% CI Lower', f"{results.get('gvi_ci_lower', 0):.1f}%"),
        ('GVI 95% CI Upper', f"{results.get('gvi_ci_upper', 0):.1f}%"),
        ('GVI Mean (Uncertainty)', f"{results.get('gvi_mean', 0):.1f}%"),
        ('Summer GVI', f"{results.get('summer_gvi', 0):.1f}%"),
        ('Winter GVI', f"{results.get('winter_gvi', 0):.1f}%"),
        ('Seasonal Change', f"{results.get('seasonal_delta', 0):+.1f}%"),
        ('Hotspot Zones', str(results.get('hotspot_count', 0))),
        ('Fusion Weights (Aerial/Street)', f"{results.get('aerial_weight', 0):.2f} / {results.get('street_weight', 0):.2f}")
    ]
    
    # Create table
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_fill_color(46, 139, 87)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(80, 10, 'Metric', 1, 0, 'C', True)
    pdf.cell(100, 10, 'Value', 1, 1, 'C', True)
    
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(0, 0, 0)
    for metric, value in metrics:
        pdf.cell(80, 8, metric, 1)
        pdf.cell(100, 8, value, 1, 1)
    
    pdf.ln(10)
    
    # Interpretation
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Quick Interpretation', ln=True)
    pdf.set_font('Helvetica', '', 10)
    
    gvi = results.get('gvi_fused', 0)
    if gvi < 20:
        gvi_status = "Low green coverage - urgent intervention needed"
    elif gvi < 40:
        gvi_status = "Moderate green coverage - room for improvement"
    else:
        gvi_status = "Good green coverage - maintain and protect"
    
    gini = results.get('gini', 1)
    if gini < 0.3:
        gini_status = "Excellent distribution of green space"
    elif gini < 0.5:
        gini_status = "Moderate equity - some areas underserved"
    else:
        gini_status = "High inequality - prioritize underserved patches"
    
    safe_multicell(pdf, f'- GVI Status: {gvi_status}')
    safe_multicell(pdf, f'- Equity Status: {gini_status}')
    
    # ===== PAGES 3-11: FIGURES =====
    figure_keys = [
        ('fig_segmentation', 'Segmentation Results'),
        ('fig_uncertainty', 'Uncertainty & Confidence Intervals'),
        ('fig_equity', 'Green Space Equity Analysis'),
        ('fig_gradcam', 'Model Attention Maps (Grad-CAM)'),
        ('fig_seasonal', 'Seasonal Simulation'),
        ('fig_cooling', 'Urban Cooling Potential'),
        ('fig_discrepancy', 'Dual-View Discrepancy Analysis'),
        ('fig_hotspots', 'Intervention Hotspots'),
        ('fig_fusion', 'Uncertainty-Weighted Fusion')
    ]
    
    for key, title in figure_keys:
        fig = results.get(key)
        if fig is not None:
            pdf.add_page()
            pdf.set_font('Helvetica', 'B', 16)
            pdf.cell(0, 12, title, ln=True)
            pdf.ln(5)
            
            try:
                img_bytes = figure_to_png_bytes(fig)
                # Calculate image dimensions to fit page
                img = Image.open(io.BytesIO(img_bytes))
                img_width = img.size[0]
                img_height = img.size[1]
                
                # Scale to fit page width (190mm usable)
                max_width = 190
                if img_width > max_width:
                    ratio = max_width / img_width
                    img_height = img_height * ratio
                    img_width = max_width
                
                pdf.image(io.BytesIO(img_bytes), x=10, y=30, w=img_width)
            except Exception as e:
                pdf.set_font('Helvetica', '', 10)
                safe_multicell(pdf, f'[Figure could not be embedded: {str(e)}]')
    
    # ===== FINAL PAGE: RECOMMENDATIONS =====
    pdf.add_page()
    pdf.set_font('Helvetica', 'B', 18)
    pdf.cell(0, 12, 'Actionable Recommendations', ln=True)
    pdf.ln(5)
    
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Based on your analysis results:', ln=True)
    pdf.ln(3)
    
    recommendations = []
    
    # Discrepancy-based recommendations
    disc_label = results.get('discrepancy_label', '')
    if 'Hidden' in disc_label:
        recommendations.append(
            "🔍 Hidden Canopy Detected: Your aerial imagery shows more vegetation than street-level views. "
            "Consider creating pedestrian green corridors and improving visibility of existing canopy from ground level."
        )
    elif 'Vertical' in disc_label:
        recommendations.append(
            "🌿 Vertical Greenery Dominance: Street-level vegetation exceeds aerial coverage. "
            "Focus on expanding tree canopy to balance the green infrastructure vertically."
        )
    else:
        recommendations.append(
            "✅ Consistent Green Visibility: Maintain current vegetation management practices "
            "while monitoring for changes using periodic assessments."
        )
    
    # Gini-based recommendations
    gini = results.get('gini', 1)
    if gini > 0.5:
        recommendations.append(
            "⚖️ High Green Space Inequality: Prioritize green infrastructure investment in low-GVI patches. "
            "Community engagement and targeted tree planting in underserved areas are recommended."
        )
    elif gini > 0.3:
        recommendations.append(
            "📊 Moderate Green Space Inequality: Some patches have lower green coverage. "
            "Consider equitable distribution strategies for new green space development."
        )
    
    # Cooling-based recommendations
    cooling = results.get('city_avg_cooling', 0)
    if cooling < 0.5:
        recommendations.append(
            "🌡️ Low Cooling Potential: Increase tree canopy cover to at least 30% GVI "
            "to achieve meaningful urban heat island mitigation (target: 1°C cooling)."
        )
    elif cooling < 1.0:
        recommendations.append(
            "🌤️ Moderate Cooling Potential: Current green spaces provide some heat mitigation. "
            "Strategic addition of trees in heat-vulnerable areas will maximize benefits."
        )
    
    # Hotspot-based recommendations
    hotspot_count = results.get('hotspot_count', 0)
    if hotspot_count > 0:
        recommendations.append(
            f"📍 {hotspot_count} Intervention Hotspots Identified: These zones combine high uncertainty "
            "and significant aerial-street discrepancy. Prioritize field validation and targeted greening here."
        )
    
    # Seasonal recommendation
    seasonal_delta = results.get('seasonal_delta', 0)
    if seasonal_delta < -10:
        recommendations.append(
            "🍂 Significant Seasonal Variation: Winter GVI drops substantially. "
            "Consider evergreen species to maintain year-round green visibility."
        )
    
    # Add recommendations to PDF
    pdf.set_font('Helvetica', '', 10)
    for i, rec in enumerate(recommendations, 1):
        wrapped = '\n'.join(textwrap.wrap(rec, width=85))
        safe_multicell(pdf, f'{i}. {rec}')
        pdf.ln(3)
    
    pdf.ln(10)
    pdf.set_font('Helvetica', 'I', 9)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 6, 'Report generated by Urban Green Space Assessment System', ln=True, align='C')
    pdf.cell(0, 6, 'Powered by UNet + YOLO + Real-ESRGAN', ln=True, align='C')
    
    # Return PDF as bytes
    return bytes(pdf.output(dest='S'))