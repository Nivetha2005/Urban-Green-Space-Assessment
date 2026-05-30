# 🚀 HOW TO RUN – Urban Green Space Assessment System

This guide provides step-by-step instructions to run the Urban Green Space Assessment System locally.

---

## 🧰 Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- pip (Python package manager)
- 4GB RAM minimum (8GB recommended)
- GPU (optional, for faster processing)

---

## 📥 Step 1: Clone the Repository

```bash
git clone https://github.com/Srilakshmi2717/urban-green-space-assessment.git
cd urban-green-space-assessment
````

---

## 📦 Step 2: Install Dependencies

### Option 1: Using requirements.txt

```bash
pip install -r requirements.txt
```

### Option 2: Manual Installation

```bash
pip install streamlit numpy pandas matplotlib pillow torch torchvision scikit-learn opencv-python scipy seaborn
```

---

## 🤖 Step 3: Add Trained Model

* Download the trained model file:

  ```
  unet_ISPRS_50epochs_complete.pth
  best.pt (yolo model)
  ```
* Place it inside the `model/` directory

OR

* Update model path in `app.py`:

```python
MODEL_PATH = "path/to/your/model.pth"
```

---

## ▶️ Step 4: Run the Application

```bash
streamlit run app.py
```

* The app will open in your browser at:

  ```
  http://localhost:8501
  ```

---

## ⚙️ Optional Run Commands

```bash
# Run on different port
streamlit run app.py --server.port 8080

# Run without opening browser
streamlit run app.py --server.headless true

# Increase upload size limit
streamlit run app.py --server.maxUploadSize 100
```

---

## 🖼️ Step 5: Using the Application

1. Select analysis mode:

   * Satellite View
   * Street View

2. Upload an image (JPG/PNG/TIFF)

3. Click **Analyze**

4. View results in tabs:

   * GVI & Cooling
   * Equity Analysis
   * Grad-CAM
   * Seasonal Simulation
   * Summary Report

---

## 📊 Expected Outputs

* Green View Index (GVI)
* Cooling Potential
* Gini Coefficient (Equity)
* Grad-CAM Heatmaps
* Seasonal GVI Comparison

---

## 🛠️ Troubleshooting

### Module not found

```bash
pip install streamlit
```

### Port already in use

```bash
streamlit run app.py --server.port 8502
```

### CUDA memory error

* Switch to CPU:

```python
DEVICE = "cpu"
```

### Model not found

* Verify model path in `app.py`

### Image upload fails

```bash
streamlit run app.py --server.maxUploadSize 500
```

---

## 🧪 (Optional) Virtual Environment Setup

```bash
python -m venv venv
```

### Activate:

* Windows:

```bash
venv\Scripts\activate
```

* Mac/Linux:

```bash
source venv/bin/activate
```

---

## ✅ Quick Run Summary

```bash
git clone https://github.com/Srilakshmi2717/urban-green-space-assessment.git
cd urban-green-space-assessment
pip install -r requirements.txt
streamlit run app.py
```

---

## 🎉 You're Ready!

Your Urban Green Space Assessment System should now be running successfully.

Happy Analyzing 🌿