# 🌿 Urban Green Space Assessment System

An AI-powered web application for analyzing urban vegetation, calculating Green Visibility Index (GVI), and estimating urban cooling potential using deep learning segmentation.

## 📋 Overview

This application uses a trained UNet segmentation model (31M parameters, 30 epochs) on the ISPRS urban dataset (Potsdam + Vaihingen) to analyze urban vegetation from satellite/aerial imagery. It provides comprehensive vegetation analysis including:

- **Green Visibility Index (GVI)** calculation with uncertainty quantification
- **Urban Cooling Potential** estimation based on Bowler et al. 2010 methodology
- **Spatial Equity Analysis** using Gini coefficient
- **Grad-CAM Explainability** for model attention visualization
- **Seasonal GVI Simulation** (Summer vs Winter)
- **Monte Carlo Dropout** for uncertainty-aware predictions

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM (8GB recommended)
- GPU (optional, for faster inference)

### Installation

1. **Clone or download the repository**
```bash
git clone https://github.com/Nivetha2005/Green-Urban-Assessment.git
cd urban-green-space-assessment
```

2. **Install required dependencies**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit numpy pandas matplotlib pillow torch torchvision scikit-learn opencv-python scipy seaborn
```

3. **Place the trained model file**
- Download `unet_ISPRS_50epochs_complete.pth` from your Kaggle dataset
- Place it in the appropriate directory or update the path in `app.py`

### Running the Application

```bash
streamlit run app.py
```

The application will open automatically in your default browser at `http://localhost:8501`

### Alternative run commands

```bash
# Run on specific port
streamlit run app.py --server.port 8080

# Run without browser auto-open
streamlit run app.py --server.headless true

# Run with increased max upload size (for large images)
streamlit run app.py --server.maxUploadSize 100
```

## 📁 Project Structure

```
urban-green-space-assessment/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── model/
│   └── unet_ISPRS_30epochs_complete.pth  # Trained UNet model
│
├── visualizations/                 # Generated analysis plots
│   ├── gvi_equity_analysis.png
│   ├── gradcam_analysis.png
│   ├── seasonal_gvi_simulation.png
│   ├── seasonal_gvi_barchart.png
│   ├── uncertainty_gvi_analysis.png
│   └── urban_cooling_potential.png
│
└── utils/                         # Optional utility scripts
    └── visualization_utils
```

## 📊 Features & Usage

### 1. Select Analysis Mode
- **Satellite/Aerial View**: For analyzing top-down urban imagery
- **Street View**: For ground-level vegetation analysis

### 2. Upload Image
- Supported formats: JPG, JPEG, PNG, TIFF
- Recommended resolution: 256x256 to 1024x1024 pixels

### 3. Analysis Tabs

| Tab | Description |
|-----|-------------|
| 🌱 GVI & Cooling | Green Visibility Index, cooling potential, uncertainty metrics |
| 📈 Equity Analysis | Gini coefficient, Lorenz curve, spatial inequality assessment |
| 🎯 Attention Maps | Grad-CAM visualizations for vegetation classes |
| ❄️ Seasonal Simulation | Summer vs Winter GVI comparison |
| 📋 Summary Report | Comprehensive analysis report with recommendations |

## 🎯 Model Performance Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | 80.29% |
| Mean IoU | 54.10% |
| Mean F1-Score | 69.2% |
| GVI Correlation | 0.95 |
| Avg Cooling Potential | 1.28°C |
| Gini Coefficient | 0.214 (Low inequality) |

### Per-Class IoU Scores
- Impervious (Roads): 59.2%
- Building: 72.4%
- Low Veg: 40.9%
- Tree: 48.2%
- Car: 0.1%
- Clutter: 10.4%

## 🔧 Troubleshooting

### Common Issues & Solutions

1. **ModuleNotFoundError: No module named 'streamlit'**
   ```bash
   pip install streamlit
   ```

2. **CUDA out of memory error**
   - Run on CPU mode by modifying `DEVICE = "cpu"` in the code
   - Reduce batch size or image resolution

3. **Model file not found**
   - Ensure the model path in `app.py` points to the correct location
   - Download the model from your Kaggle dataset

4. **Port already in use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

5. **Image upload fails**
   - Check file size (max 200MB default)
   - Increase limit: `streamlit run app.py --server.maxUploadSize 500`

## 📖 Usage Examples

### Example 1: Analyzing a Satellite Image
1. Select "Satellite/Aerial View"
2. Upload a city satellite image
3. Click "Analyze Vegetation"
4. Review GVI score, cooling potential, and spatial equity metrics

### Example 2: Seasonal Comparison
1. Upload a summer image
2. Navigate to "Seasonal Simulation" tab
3. View winter simulation and GVI reduction percentage

## 🤝 Integration Guide

### To integrate with your trained model:

1. **Update the model path in `app.py`:**
```python
MODEL_PATH = "/path/to/your/unet_ISPRS_30epochs_complete.pth"
```

2. **Replace mock values with actual inference:**
```python
# Find this section in app.py and replace with your model inference
if st.session_state.analysis_mode == "satellite":
    gvi_value = 42.7  # Replace with actual model output
    cooling_potential = 1.28  # Replace with calculation
```

3. **Add your segmentation function:**
```python
def run_segmentation(image):
    # Your model inference code here
    return prediction_mask, gvi_score, cooling_potential
```

## 📝 Output Files Generated

The application expects the following visualization files (generated from your notebook):

| File | Description |
|------|-------------|
| `gvi_equity_analysis.png` | Gini coefficient and Lorenz curve |
| `gradcam_analysis.png` | Model attention heatmaps |
| `seasonal_gvi_simulation.png` | Summer vs Winter comparison |
| `seasonal_gvi_barchart.png` | Seasonal variation bar chart |
| `uncertainty_gvi_analysis.png` | MC Dropout uncertainty maps |
| `urban_cooling_potential.png` | Temperature reduction estimates |

## 🌟 Citation

If you use this system in your research, please cite:

```
Urban Green Space Assessment System. (2026). 
Based on ISPRS Urban Segmentation Dataset and UNet Architecture.
Methodology: Bowler et al. (2010) Urban cooling coefficients.
```

## 📄 License

This project is licensed under the MIT License. See the LICENSE file for details.

## 📧 Contact

For questions or support, please contact:
- Email : nivethatk03@gmail.com

## 🙏 Acknowledgments

- ISPRS for the Potsdam and Vaihingen datasets
- Bowler et al. (2010) for urban cooling methodology
- Kaggle for computational resources

---

**Happy Analyzing! 🌿🏙️**
```

## Quick Setup Commands (Copy-Paste Ready):

```bash
# 1. Create project directory
mkdir urban-green-space-assessment
cd urban-green-space-assessment

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Create requirements.txt and install
pip install streamlit numpy pandas matplotlib pillow torch torchvision scikit-learn opencv-python scipy seaborn

# 4. Save as requirements.txt
pip freeze > requirements.txt

# 5. Run the app
streamlit run app.py
