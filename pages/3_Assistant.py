# === FILE: pages/3_Assistant.py ===
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from pages.page_assistant import show

show()