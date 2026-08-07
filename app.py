# app.py
import streamlit as st
from models import LocalAITradingEngine, AdvancedTradingModels, EnsembleTradingModel
from models.model_utils import ModelUtils

# اختيار النموذج
model_type = st.sidebar.selectbox(
    "اختر نوع النموذج:",
    ["Basic (Random Forest)", "Advanced (Multiple Models)", "Ensemble (Voting/Stacking)"]
)

# إنشاء النموذج المناسب
if model_type == "Basic (Random Forest)":
    engine = LocalAITradingEngine()
elif model_type == "Advanced (Multiple Models)":
    engine = AdvancedTradingModels()
else:
    engine = EnsembleTradingModel()