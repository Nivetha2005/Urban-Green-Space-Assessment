# === FILE: pages/2_Compare.py ===
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from pages.page_compare import show
from utils.inference import load_unet, load_yolo
from utils.superres import load_sr_model
import torch

@st.cache_resource
def get_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    unet = load_unet(os.path.join('models', 'unet_ISPRS_30epochs_complete.pth'), device)
    yolo = load_yolo(os.path.join('models', 'yolo_street.pt'))
    sr = load_sr_model()
    return unet, yolo, sr, device

unet_model, yolo_model, sr_model, device = get_models()

show(unet_model, yolo_model, sr_model, device)