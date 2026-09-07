import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import io
import uuid
import pandas as pd
from datetime import datetime
import html

from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as PDFImage
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch

from gradcam import (
    get_gradcam_heatmap,
    overlay_gradcam,
    calculate_severity_percentage
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Mulberry AI",
    page_icon="🌿",
    layout="wide"
)


# ============================================================
# PROFESSIONAL UI
# ============================================================

st.markdown(
    """
<style>

.block-container {
    max-width: 1450px;
    padding-top: 1.2rem;
    padding-bottom: 3rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3 {
    letter-spacing: -0.3px;
}

.hero {
    border-radius: 20px;
    padding: 25px 28px;
    margin-bottom: 22px;

    background:
        linear-gradient(
            135deg,
            rgba(34, 80, 52, 0.45),
            rgba(16, 21, 28, 0.96)
        );

    border: 1px solid rgba(255,255,255,0.09);
}

.hero-title {
    font-size: 38px;
    font-weight: 800;
    line-height: 1.1;
}

.hero-subtitle {
    font-size: 16px;
    opacity: 0.72;
    margin-top: 7px;
}

.hero-status {
    margin-top: 15px;
    font-size: 13px;
    opacity: 0.85;
}

.section-title {
    font-size: 25px;
    font-weight: 750;
}

.section-description {
    font-size: 14px;
    opacity: 0.62;
    margin-top: 3px;
}

.card {
    border-radius: 17px;
    padding: 20px;
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 14px;
}

.diagnosis-card {
    padding: 25px;
    border-radius: 19px;
    text-align: center;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.045),
            rgba(255,255,255,0.015)
        );

    border: 1px solid rgba(255,255,255,0.10);
}

.diagnosis-small {
    font-size: 14px;
    opacity: 0.65;
}

.diagnosis-name {
    font-size: 35px;
    font-weight: 800;
    margin-top: 6px;
}

.diagnosis-confidence {
    font-size: 16px;
    margin-top: 7px;
}

.action-card {
    border-radius: 18px;
    padding: 23px;

    background:
        linear-gradient(
            145deg,
            rgba(35,85,55,0.25),
            rgba(255,255,255,0.015)
        );

    border: 1px solid rgba(100,180,120,0.18);
}

.action-title {
    font-size: 22px;
    font-weight: 750;
}

.recommendation-step {
    display: flex;
    align-items: flex-start;
    gap: 13px;
    margin-bottom: 13px;
    line-height: 1.5;
}

.recommendation-number {
    min-width: 30px;
    height: 30px;
    border-radius: 50%;

    display: flex;
    align-items: center;
    justify-content: center;

    font-weight: 700;
    background: rgba(255,255,255,0.10);
}

.recommendation-text {
    padding-top: 3px;
    font-size: 16px;
}

@media only screen and (max-width: 768px) {

    .block-container {
        padding-left: 0.75rem;
        padding-right: 0.75rem;
        padding-top: 0.7rem;
    }

    .hero {
        padding: 20px;
    }

    .hero-title {
        font-size: 29px;
    }

    .hero-subtitle {
        font-size: 14px;
    }

    .diagnosis-name {
        font-size: 28px;
    }

    .section-title {
        font-size: 21px;
    }

    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
    }

    .stButton button {
        width: 100%;
    }

    .stDownloadButton button {
        width: 100%;
    }

    img {
        max-width: 100%;
        height: auto;
    }
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# COMPLETE LANGUAGE DATA
# ============================================================

LANG = {

"English": {

    "title": "🌿 Mulberry AI",
    "subtitle": "AI-Powered Mulberry Leaf Disease Detection",

    "settings": "⚙️ Settings",
    "language": "Language",
    "model": "🤖 AI Model",
    "model_ready": "🟢 Model Ready",
    "three_classes": "3 disease classes",

    "detection": "🔬 Detection",
    "healthy_class": "🟢 Healthy",
    "rust_class": "🟠 Leaf Rust",
    "spot_class": "🟤 Leaf Spot",

    "image_tips": "📷 Image Tips",
    "tip1": "✓ Clear leaf",
    "tip2": "✓ Good lighting",
    "tip3": "✓ Leaf centered",
    "tip4": "✓ Avoid blur",

    "session_analyses": "Session Analyses",
    "clear_history": "🗑️ Clear History",

    "farm_overview": "🌾 Farm Overview",
    "leaves_checked": "Leaves Checked",
    "healthy": "Healthy",
    "needs_attention": "Needs Attention",
    "farm_health": "Farm Health",

    "check_leaf": "📷 Check a Mulberry Leaf",
    "check_description":
        "Upload a clear photograph of a mulberry leaf. "
        "The AI will analyze the leaf and show the result below.",

    "upload": "Upload a Mulberry Leaf Image",

    "farm_info": "🗺️ Farm Information (Optional)",
    "farm_zone": "Farm Zone / Plot",
    "farm_zone_placeholder": "Example: Block A - Plot 3",
    "observation": "Observation",
    "observation_placeholder": "Example: Brown spots noticed",

    "diagnosis_section": "🔬 Your Leaf Diagnosis",
    "diagnosis_description":
        "Here is the main result from the AI analysis.",
    "diagnosis": "AI DIAGNOSIS",
    "confidence": "AI Confidence",
    "severity": "Disease Severity",
    "health_score": "Leaf Health",
    "high_confidence": "✓ High-confidence AI classification",
    "moderate_confidence":
        "⚠️ AI confidence is moderate. Review the visual analysis below.",
    "low_confidence":
        "⚠️ AI confidence is low. Please capture another clear leaf image.",

    "what_do": "👨‍🌾 What Should I Do?",
    "what_do_desc":
        "Simple recommended steps based on the AI result.",
    "recommended_action": "Recommended Action",

    "healthy_rec": [
        "Continue regular monitoring.",
        "Maintain good farm and leaf hygiene.",
        "Check surrounding plants regularly.",
        "Recheck the leaf if visible symptoms appear."
    ],

    "rust_rec": [
        "Inspect surrounding leaves for similar symptoms.",
        "Remove heavily affected leaves where appropriate.",
        "Improve air circulation around affected plants.",
        "Follow locally approved disease-management guidance.",
        "Continue monitoring for progression."
    ],

    "spot_rec": [
        "Inspect surrounding leaves for similar spots.",
        "Remove heavily affected leaves where appropriate.",
        "Maintain good farm hygiene.",
        "Follow locally approved disease-management guidance.",
        "Continue monitoring for progression."
    ],

    "safety_warning":
        "⚠️ Safety Warning: Follow locally approved agricultural "
        "guidance and observe all required safety periods before feeding silkworms.",

    "see_ai": "👁️ See What the AI Sees",
    "see_ai_desc":
        "The highlighted image shows the regions that contributed "
        "to the AI analysis.",
    "original_leaf": "📷 Original Leaf",
    "uploaded_leaf": "Uploaded Leaf",
    "ai_vision": "🧠 AI Vision Analysis",

    "leaf_health": "📊 Leaf Health",
    "disease_risk": "Disease Risk",
    "risk_level": "Risk Level",

    "detailed_analysis": "🔎 Detailed AI Analysis",
    "probabilities": "📈 Prediction Probabilities",
    "second_opinion": "🧠 AI Second Opinion",
    "symptoms": "🔍 Observed Symptoms",
    "quality": "📸 Image Quality Check",
    "environment": "🌦️ Environmental Risk Assessment",
    "progression": "📅 Disease Progression / Monitoring",
    "before_after": "🔄 Before & After Comparison",

    "strong_opinion":
        "Strong AI prediction. The classification confidence is high. "
        "Review the highlighted regions for additional visual context.",
    "moderate_opinion":
        "Moderate AI confidence. The result should be reviewed along "
        "with the highlighted regions and visible symptoms.",
    "low_opinion":
        "Low AI confidence. Treat this result as uncertain and capture another clear image.",

    "no_symptoms":
        "No major disease symptoms were identified by the classification model.",
    "rust_symptoms":
        "Rust-colored or brown lesion patterns may be present. "
        "The AI Vision Analysis highlights regions that influenced the prediction.",
    "spot_symptoms":
        "Spots or localized lesion patterns may be present. "
        "The AI Vision Analysis highlights regions that influenced the prediction.",

    "image_quality": "Image Quality",
    "quality_good": "✓ Image quality is suitable for analysis.",
    "quality_medium": "⚠️ Image quality is moderate.",
    "quality_bad":
        "⚠️ Image quality is poor. Consider capturing another image.",

    "environment_note":
        "Indicative assessment only. This does not modify the AI model prediction.",
    "humidity": "Humidity (%)",
    "temperature": "Temperature (°C)",
    "leaf_wetness": "Leaf Wetness",
    "low": "Low",
    "medium": "Medium",
    "high": "High",
    "environmental_risk": "Environmental Risk",
    "high_environment": "🔴 High environmental risk",
    "medium_environment": "🟠 Medium environmental risk",
    "low_environment": "🟢 Low environmental risk",

    "monitoring_text":
        "Use this section to record severity over time.",
    "monitoring_day": "Monitoring Day",
    "observed_severity": "Observed Severity (%)",
    "save_monitoring": "📌 Save Monitoring Record",
    "monitor_saved": "✓ Monitoring record saved.",

    "previous_image": "Upload a previous leaf image",
    "previous": "Previous",
    "current": "Current",
    "compare_info":
        "Use the two images to visually compare leaf condition over time.",

    "save_analysis": "💾 Save This Analysis",
    "save_button": "💾 Save Analysis",
    "saved": "✓ Analysis saved successfully",

    "reports": "📄 Reports",
    "download_report": "📄 Download Complete AI Report",

    "batch": "📦 Batch Analysis — Multiple Leaves",
    "batch_description":
        "Analyze several leaf images together for a quick farm-level overview.",
    "multiple_upload": "Upload Multiple Leaf Images",
    "analyze_all": "🚀 Analyze All Leaves",
    "leaves_analyzed": "leaves analyzed.",
    "batch_results": "📊 View Batch Results",
    "total": "Total",
    "download_csv": "📥 Download Batch CSV",

    "farm_analytics": "🌾 Farm Analytics",
    "distribution": "🌿 Disease Distribution",
    "zone_distribution": "🗺️ Farm Zone Distribution",
    "farm_health_trend": "📈 Farm Health Trend",
    "history": "📚 Analysis History",
    "download_history": "📥 Download Analysis History",

    "analysis_id": "Analysis ID",
    "time": "Time",
    "zone": "Farm Zone",
    "image": "Image",
    "risk": "Disease Risk",
    "observation_col": "Observation",

    "report_title": "Leaf Disease Analysis Report",
    "report_prediction": "Prediction Probabilities",
    "report_opinion": "AI Second Opinion",
    "report_symptoms": "Observed Symptoms",
    "report_recommendation": "Recommended Action",
    "report_health": "Health Score",
    "report_risk": "Disease Risk",
    "report_quality": "Image Quality",
    "report_safety":
        "Safety Warning: Follow locally approved agricultural guidance "
        "and product-label requirements.",
    "generated": "Generated by Mulberry AI",

    "footer":
        "🌿 Mulberry AI • AI-assisted mulberry leaf analysis",
    "footer_note":
        "AI results are intended to assist observation and should be "
        "verified with appropriate agricultural guidance."
},


"Kannada": {

    "title": "🌿 ಮಲ್ಬರಿ AI",
    "subtitle": "AI ಆಧಾರಿತ ಹಿಪ್ಪುನೇರಳೆ ಎಲೆ ರೋಗ ಪತ್ತೆ",

    "settings": "⚙️ ಸೆಟ್ಟಿಂಗ್‌ಗಳು",
    "language": "ಭಾಷೆ",
    "model": "🤖 AI ಮಾದರಿ",
    "model_ready": "🟢 ಮಾದರಿ ಸಿದ್ಧವಾಗಿದೆ",
    "three_classes": "3 ರೋಗ ವರ್ಗಗಳು",

    "detection": "🔬 ಪತ್ತೆ",
    "healthy_class": "🟢 ಆರೋಗ್ಯಕರ",
    "rust_class": "🟠 ಎಲೆ ತುಕ್ಕು",
    "spot_class": "🟤 ಎಲೆ ಚುಕ್ಕೆ",

    "image_tips": "📷 ಚಿತ್ರ ಸಲಹೆಗಳು",
    "tip1": "✓ ಸ್ಪಷ್ಟವಾದ ಎಲೆ",
    "tip2": "✓ ಉತ್ತಮ ಬೆಳಕು",
    "tip3": "✓ ಎಲೆಯನ್ನು ಮಧ್ಯದಲ್ಲಿ ಇರಿಸಿ",
    "tip4": "✓ ಮಸುಕಾದ ಚಿತ್ರ ತಪ್ಪಿಸಿ",

    "session_analyses": "ಸೆಷನ್ ವಿಶ್ಲೇಷಣೆಗಳು",
    "clear_history": "🗑️ ಇತಿಹಾಸ ಅಳಿಸಿ",

    "farm_overview": "🌾 ಕೃಷಿ ಅವಲೋಕನ",
    "leaves_checked": "ಪರಿಶೀಲಿಸಿದ ಎಲೆಗಳು",
    "healthy": "ಆರೋಗ್ಯಕರ",
    "needs_attention": "ಗಮನ ಅಗತ್ಯ",
    "farm_health": "ಕೃಷಿ ಆರೋಗ್ಯ",

    "check_leaf": "📷 ಹಿಪ್ಪುನೇರಳೆ ಎಲೆ ಪರಿಶೀಲಿಸಿ",
    "check_description":
        "ಹಿಪ್ಪುನೇರಳೆ ಎಲೆಯ ಸ್ಪಷ್ಟ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ. "
        "AI ಚಿತ್ರವನ್ನು ವಿಶ್ಲೇಷಿಸಿ ಫಲಿತಾಂಶವನ್ನು ತೋರಿಸುತ್ತದೆ.",

    "upload": "ಹಿಪ್ಪುನೇರಳೆ ಎಲೆಯ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",

    "farm_info": "🗺️ ಕೃಷಿ ಮಾಹಿತಿ (ಐಚ್ಛಿಕ)",
    "farm_zone": "ಕೃಷಿ ವಲಯ / ಪ್ಲಾಟ್",
    "farm_zone_placeholder": "ಉದಾಹರಣೆ: ಬ್ಲಾಕ್ A - ಪ್ಲಾಟ್ 3",
    "observation": "ಗಮನಿಸಿದ ಲಕ್ಷಣ",
    "observation_placeholder": "ಉದಾಹರಣೆ: ಕಂದು ಚುಕ್ಕೆಗಳು",

    "diagnosis_section": "🔬 ನಿಮ್ಮ ಎಲೆಯ ರೋಗ ಪತ್ತೆ",
    "diagnosis_description":
        "AI ವಿಶ್ಲೇಷಣೆಯ ಮುಖ್ಯ ಫಲಿತಾಂಶ ಇಲ್ಲಿದೆ.",
    "diagnosis": "AI ರೋಗ ಪತ್ತೆ",
    "confidence": "AI ವಿಶ್ವಾಸ",
    "severity": "ರೋಗದ ತೀವ್ರತೆ",
    "health_score": "ಎಲೆಯ ಆರೋಗ್ಯ",
    "high_confidence": "✓ ಹೆಚ್ಚಿನ ವಿಶ್ವಾಸದ AI ವರ್ಗೀಕರಣ",
    "moderate_confidence":
        "⚠️ AI ವಿಶ್ವಾಸ ಮಧ್ಯಮವಾಗಿದೆ. ಕೆಳಗಿನ ದೃಶ್ಯ ವಿಶ್ಲೇಷಣೆಯನ್ನು ಪರಿಶೀಲಿಸಿ.",
    "low_confidence":
        "⚠️ AI ವಿಶ್ವಾಸ ಕಡಿಮೆಯಾಗಿದೆ. ಮತ್ತೊಂದು ಸ್ಪಷ್ಟ ಚಿತ್ರ ತೆಗೆದುಕೊಳ್ಳಿ.",

    "what_do": "👨‍🌾 ನಾನು ಏನು ಮಾಡಬೇಕು?",
    "what_do_desc":
        "AI ಫಲಿತಾಂಶದ ಆಧಾರದ ಮೇಲೆ ಸರಳ ಶಿಫಾರಸುಗಳು.",
    "recommended_action": "ಶಿಫಾರಸು ಮಾಡಲಾದ ಕ್ರಮ",

    "healthy_rec": [
        "ನಿಯಮಿತವಾಗಿ ಮೇಲ್ವಿಚಾರಣೆ ಮಾಡಿ.",
        "ಉತ್ತಮ ಕೃಷಿ ಮತ್ತು ಎಲೆ ಸ್ವಚ್ಛತೆಯನ್ನು ಕಾಪಾಡಿಕೊಳ್ಳಿ.",
        "ಸುತ್ತಮುತ್ತಲಿನ ಸಸ್ಯಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಪರಿಶೀಲಿಸಿ.",
        "ರೋಗದ ಲಕ್ಷಣಗಳು ಕಂಡುಬಂದರೆ ಎಲೆಯನ್ನು ಮತ್ತೆ ಪರಿಶೀಲಿಸಿ."
    ],

    "rust_rec": [
        "ಇದೇ ರೀತಿಯ ಲಕ್ಷಣಗಳಿಗಾಗಿ ಸುತ್ತಮುತ್ತಲಿನ ಎಲೆಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
        "ಹೆಚ್ಚು ಬಾಧಿತ ಎಲೆಗಳನ್ನು ಸೂಕ್ತವಾಗಿ ತೆಗೆದುಹಾಕಿ.",
        "ಸಸ್ಯಗಳ ನಡುವೆ ಉತ್ತಮ ಗಾಳಿಯ ಹರಿವನ್ನು ಕಾಪಾಡಿಕೊಳ್ಳಿ.",
        "ಸ್ಥಳೀಯವಾಗಿ ಅನುಮೋದಿತ ರೋಗ ನಿರ್ವಹಣಾ ಮಾರ್ಗದರ್ಶನವನ್ನು ಅನುಸರಿಸಿ.",
        "ರೋಗದ ಬೆಳವಣಿಗೆಯನ್ನು ನಿರಂತರವಾಗಿ ಗಮನಿಸಿ."
    ],

    "spot_rec": [
        "ಇದೇ ರೀತಿಯ ಚುಕ್ಕೆಗಳಿಗಾಗಿ ಸುತ್ತಮುತ್ತಲಿನ ಎಲೆಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.",
        "ಹೆಚ್ಚು ಬಾಧಿತ ಎಲೆಗಳನ್ನು ಸೂಕ್ತವಾಗಿ ತೆಗೆದುಹಾಕಿ.",
        "ಉತ್ತಮ ಕೃಷಿ ಸ್ವಚ್ಛತೆಯನ್ನು ಕಾಪಾಡಿಕೊಳ್ಳಿ.",
        "ಸ್ಥಳೀಯವಾಗಿ ಅನುಮೋದಿತ ರೋಗ ನಿರ್ವಹಣಾ ಮಾರ್ಗದರ್ಶನವನ್ನು ಅನುಸರಿಸಿ.",
        "ರೋಗದ ಬೆಳವಣಿಗೆಯನ್ನು ನಿರಂತರವಾಗಿ ಗಮನಿಸಿ."
    ],

    "safety_warning":
        "⚠️ ಸುರಕ್ಷತಾ ಎಚ್ಚರಿಕೆ: ಸ್ಥಳೀಯವಾಗಿ ಅನುಮೋದಿತ ಕೃಷಿ ಮಾರ್ಗದರ್ಶನವನ್ನು "
        "ಅನುಸರಿಸಿ ಮತ್ತು ರೇಷ್ಮೆ ಹುಳುಗಳಿಗೆ ಎಲೆ ನೀಡುವ ಮೊದಲು ಅಗತ್ಯ ಸುರಕ್ಷತಾ ಅವಧಿಯನ್ನು ಪಾಲಿಸಿ.",

    "see_ai": "👁️ AI ಏನು ನೋಡುತ್ತಿದೆ?",
    "see_ai_desc":
        "ಹೈಲೈಟ್ ಮಾಡಲಾದ ಪ್ರದೇಶಗಳು AI ವಿಶ್ಲೇಷಣೆಗೆ ಮುಖ್ಯವಾಗಿವೆ.",
    "original_leaf": "📷 ಮೂಲ ಎಲೆ",
    "uploaded_leaf": "ಅಪ್‌ಲೋಡ್ ಮಾಡಿದ ಎಲೆ",
    "ai_vision": "🧠 AI ದೃಶ್ಯ ವಿಶ್ಲೇಷಣೆ",

    "leaf_health": "📊 ಎಲೆಯ ಆರೋಗ್ಯ",
    "disease_risk": "ರೋಗದ ಅಪಾಯ",
    "risk_level": "ಅಪಾಯದ ಮಟ್ಟ",

    "detailed_analysis": "🔎 ವಿವರವಾದ AI ವಿಶ್ಲೇಷಣೆ",
    "probabilities": "📈 ಭವಿಷ್ಯವಾಣಿ ಸಾಧ್ಯತೆಗಳು",
    "second_opinion": "🧠 AI ಎರಡನೇ ಅಭಿಪ್ರಾಯ",
    "symptoms": "🔍 ಗಮನಿಸಿದ ಲಕ್ಷಣಗಳು",
    "quality": "📸 ಚಿತ್ರದ ಗುಣಮಟ್ಟ ಪರಿಶೀಲನೆ",
    "environment": "🌦️ ಪರಿಸರ ಅಪಾಯ ಮೌಲ್ಯಮಾಪನ",
    "progression": "📅 ರೋಗ ಬೆಳವಣಿಗೆ / ಮೇಲ್ವಿಚಾರಣೆ",
    "before_after": "🔄 ಮೊದಲು ಮತ್ತು ನಂತರ ಹೋಲಿಕೆ",

    "strong_opinion":
        "AI ಫಲಿತಾಂಶದ ವಿಶ್ವಾಸ ಹೆಚ್ಚು ಇದೆ. ಹೈಲೈಟ್ ಮಾಡಲಾದ ಪ್ರದೇಶಗಳನ್ನು "
        "ಹೆಚ್ಚುವರಿ ದೃಶ್ಯ ಮಾಹಿತಿಗಾಗಿ ಪರಿಶೀಲಿಸಿ.",
    "moderate_opinion":
        "AI ವಿಶ್ವಾಸ ಮಧ್ಯಮವಾಗಿದೆ. ಹೈಲೈಟ್ ಮಾಡಲಾದ ಪ್ರದೇಶಗಳು ಮತ್ತು ಗೋಚರಿಸುವ "
        "ಲಕ್ಷಣಗಳೊಂದಿಗೆ ಫಲಿತಾಂಶವನ್ನು ಪರಿಶೀಲಿಸಿ.",
    "low_opinion":
        "AI ವಿಶ್ವಾಸ ಕಡಿಮೆಯಾಗಿದೆ. ಫಲಿತಾಂಶವನ್ನು ಖಚಿತವೆಂದು ಪರಿಗಣಿಸದೆ ಮತ್ತೊಂದು ಸ್ಪಷ್ಟ ಚಿತ್ರ ತೆಗೆದುಕೊಳ್ಳಿ.",

    "no_symptoms":
        "ವರ್ಗೀಕರಣ ಮಾದರಿಯಿಂದ ಪ್ರಮುಖ ರೋಗ ಲಕ್ಷಣಗಳು ಪತ್ತೆಯಾಗಿಲ್ಲ.",
    "rust_symptoms":
        "ತುಕ್ಕು ಬಣ್ಣದ ಅಥವಾ ಕಂದು ಗಾಯದ ಮಾದರಿಗಳು ಇರಬಹುದು. "
        "AI ದೃಶ್ಯ ವಿಶ್ಲೇಷಣೆ ಫಲಿತಾಂಶಕ್ಕೆ ಕಾರಣವಾದ ಪ್ರದೇಶಗಳನ್ನು ತೋರಿಸುತ್ತದೆ.",
    "spot_symptoms":
        "ಚುಕ್ಕೆಗಳು ಅಥವಾ ಸ್ಥಳೀಯ ಗಾಯದ ಮಾದರಿಗಳು ಇರಬಹುದು. "
        "AI ದೃಶ್ಯ ವಿಶ್ಲೇಷಣೆ ಫಲಿತಾಂಶಕ್ಕೆ ಕಾರಣವಾದ ಪ್ರದೇಶಗಳನ್ನು ತೋರಿಸುತ್ತದೆ.",

    "image_quality": "ಚಿತ್ರದ ಗುಣಮಟ್ಟ",
    "quality_good": "✓ ಚಿತ್ರದ ಗುಣಮಟ್ಟ ವಿಶ್ಲೇಷಣೆಗೆ ಸೂಕ್ತವಾಗಿದೆ.",
    "quality_medium": "⚠️ ಚಿತ್ರದ ಗುಣಮಟ್ಟ ಮಧ್ಯಮವಾಗಿದೆ.",
    "quality_bad":
        "⚠️ ಚಿತ್ರದ ಗುಣಮಟ್ಟ ಕಡಿಮೆಯಾಗಿದೆ. ಮತ್ತೊಂದು ಚಿತ್ರ ತೆಗೆದುಕೊಳ್ಳಿ.",

    "environment_note":
        "ಇದು ಸೂಚಕ ಮೌಲ್ಯಮಾಪನ ಮಾತ್ರ. ಇದು AI ಮಾದರಿಯ ಭವಿಷ್ಯವಾಣಿಯನ್ನು ಬದಲಾಯಿಸುವುದಿಲ್ಲ.",
    "humidity": "ಆರ್ದ್ರತೆ (%)",
    "temperature": "ತಾಪಮಾನ (°C)",
    "leaf_wetness": "ಎಲೆಯ ತೇವಾಂಶ",
    "low": "ಕಡಿಮೆ",
    "medium": "ಮಧ್ಯಮ",
    "high": "ಹೆಚ್ಚು",
    "environmental_risk": "ಪರಿಸರ ಅಪಾಯ",
    "high_environment": "🔴 ಹೆಚ್ಚಿನ ಪರಿಸರ ಅಪಾಯ",
    "medium_environment": "🟠 ಮಧ್ಯಮ ಪರಿಸರ ಅಪಾಯ",
    "low_environment": "🟢 ಕಡಿಮೆ ಪರಿಸರ ಅಪಾಯ",

    "monitoring_text": "ಕಾಲಕ್ರಮೇಣ ರೋಗದ ತೀವ್ರತೆಯನ್ನು ದಾಖಲಿಸಲು ಈ ವಿಭಾಗ ಬಳಸಿ.",
    "monitoring_day": "ಮೇಲ್ವಿಚಾರಣೆ ದಿನ",
    "observed_severity": "ಗಮನಿಸಿದ ತೀವ್ರತೆ (%)",
    "save_monitoring": "📌 ಮೇಲ್ವಿಚಾರಣೆ ದಾಖಲಿಸಿ",
    "monitor_saved": "✓ ಮೇಲ್ವಿಚಾರಣೆ ದಾಖಲಾಗಿದೆ.",

    "previous_image": "ಹಿಂದಿನ ಎಲೆಯ ಚಿತ್ರವನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
    "previous": "ಹಿಂದಿನ",
    "current": "ಪ್ರಸ್ತುತ",
    "compare_info":
        "ಕಾಲಕ್ರಮೇಣ ಎಲೆಯ ಸ್ಥಿತಿಯನ್ನು ದೃಶ್ಯವಾಗಿ ಹೋಲಿಸಲು ಎರಡು ಚಿತ್ರಗಳನ್ನು ಬಳಸಿ.",

    "save_analysis": "💾 ಈ ವಿಶ್ಲೇಷಣೆಯನ್ನು ಉಳಿಸಿ",
    "save_button": "💾 ವಿಶ್ಲೇಷಣೆ ಉಳಿಸಿ",
    "saved": "✓ ವಿಶ್ಲೇಷಣೆ ಯಶಸ್ವಿಯಾಗಿ ಉಳಿಸಲಾಗಿದೆ",

    "reports": "📄 ವರದಿಗಳು",
    "download_report": "📄 ಸಂಪೂರ್ಣ AI ವರದಿ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",

    "batch": "📦 ಬ್ಯಾಚ್ ವಿಶ್ಲೇಷಣೆ — ಹಲವು ಎಲೆಗಳು",
    "batch_description":
        "ತ್ವರಿತ ಕೃಷಿ ಮಟ್ಟದ ಅವಲೋಕನಕ್ಕಾಗಿ ಹಲವು ಎಲೆಗಳನ್ನು ಒಟ್ಟಿಗೆ ವಿಶ್ಲೇಷಿಸಿ.",
    "multiple_upload": "ಹಲವು ಎಲೆಗಳ ಚಿತ್ರಗಳನ್ನು ಅಪ್‌ಲೋಡ್ ಮಾಡಿ",
    "analyze_all": "🚀 ಎಲ್ಲಾ ಎಲೆಗಳನ್ನು ವಿಶ್ಲೇಷಿಸಿ",
    "leaves_analyzed": "ಎಲೆಗಳನ್ನು ವಿಶ್ಲೇಷಿಸಲಾಗಿದೆ.",
    "batch_results": "📊 ಬ್ಯಾಚ್ ಫಲಿತಾಂಶಗಳನ್ನು ನೋಡಿ",
    "total": "ಒಟ್ಟು",
    "download_csv": "📥 ಬ್ಯಾಚ್ CSV ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",

    "farm_analytics": "🌾 ಕೃಷಿ ವಿಶ್ಲೇಷಣೆ",
    "distribution": "🌿 ರೋಗ ವಿತರಣೆ",
    "zone_distribution": "🗺️ ಕೃಷಿ ವಲಯ ವಿತರಣೆ",
    "farm_health_trend": "📈 ಕೃಷಿ ಆರೋಗ್ಯದ ಪ್ರವೃತ್ತಿ",
    "history": "📚 ವಿಶ್ಲೇಷಣೆ ಇತಿಹಾಸ",
    "download_history": "📥 ವಿಶ್ಲೇಷಣೆ ಇತಿಹಾಸ ಡೌನ್‌ಲೋಡ್ ಮಾಡಿ",

    "analysis_id": "ವಿಶ್ಲೇಷಣೆ ID",
    "time": "ಸಮಯ",
    "zone": "ಕೃಷಿ ವಲಯ",
    "image": "ಚಿತ್ರ",
    "risk": "ರೋಗದ ಅಪಾಯ",
    "observation_col": "ಗಮನಿಸಿದ ಲಕ್ಷಣ",

    "report_title": "ಎಲೆ ರೋಗ ವಿಶ್ಲೇಷಣೆ ವರದಿ",
    "report_prediction": "ಭವಿಷ್ಯವಾಣಿ ಸಾಧ್ಯತೆಗಳು",
    "report_opinion": "AI ಎರಡನೇ ಅಭಿಪ್ರಾಯ",
    "report_symptoms": "ಗಮನಿಸಿದ ಲಕ್ಷಣಗಳು",
    "report_recommendation": "ಶಿಫಾರಸು ಮಾಡಲಾದ ಕ್ರಮ",
    "report_health": "ಆರೋಗ್ಯ ಸ್ಕೋರ್",
    "report_risk": "ರೋಗದ ಅಪಾಯ",
    "report_quality": "ಚಿತ್ರದ ಗುಣಮಟ್ಟ",
    "report_safety":
        "ಸುರಕ್ಷತಾ ಎಚ್ಚರಿಕೆ: ಸ್ಥಳೀಯವಾಗಿ ಅನುಮೋದಿತ ಕೃಷಿ ಮಾರ್ಗದರ್ಶನ "
        "ಮತ್ತು ಉತ್ಪನ್ನದ ಲೇಬಲ್ ಅವಶ್ಯಕತೆಗಳನ್ನು ಅನುಸರಿಸಿ.",
    "generated": "Mulberry AI ಮೂಲಕ ರಚಿಸಲಾಗಿದೆ",

    "footer":
        "🌿 Mulberry AI • AI ಆಧಾರಿತ ಹಿಪ್ಪುನೇರಳೆ ಎಲೆ ವಿಶ್ಲೇಷಣೆ",
    "footer_note":
        "AI ಫಲಿತಾಂಶಗಳು ಪರಿಶೀಲನೆಗೆ ಸಹಾಯ ಮಾಡಲು ಮಾತ್ರ. ಸೂಕ್ತ ಕೃಷಿ ಮಾರ್ಗದರ್ಶನದೊಂದಿಗೆ ಪರಿಶೀಲಿಸಿ."
},


"Telugu": {

    "title": "🌿 మల్బరీ AI",
    "subtitle": "AI ఆధారిత మల్బరీ ఆకు వ్యాధి గుర్తింపు",

    "settings": "⚙️ సెట్టింగ్స్",
    "language": "భాష",
    "model": "🤖 AI మోడల్",
    "model_ready": "🟢 మోడల్ సిద్ధంగా ఉంది",
    "three_classes": "3 వ్యాధి తరగతులు",

    "detection": "🔬 గుర్తింపు",
    "healthy_class": "🟢 ఆరోగ్యకరమైనది",
    "rust_class": "🟠 ఆకు తుప్పు",
    "spot_class": "🟤 ఆకు మచ్చ",

    "image_tips": "📷 చిత్ర సూచనలు",
    "tip1": "✓ స్పష్టమైన ఆకు",
    "tip2": "✓ మంచి వెలుతురు",
    "tip3": "✓ ఆకును మధ్యలో ఉంచండి",
    "tip4": "✓ మసక చిత్రాన్ని నివారించండి",

    "session_analyses": "సెషన్ విశ్లేషణలు",
    "clear_history": "🗑️ చరిత్రను తొలగించండి",

    "farm_overview": "🌾 వ్యవసాయ అవలోకనం",
    "leaves_checked": "పరిశీలించిన ఆకులు",
    "healthy": "ఆరోగ్యకరమైనవి",
    "needs_attention": "శ్రద్ధ అవసరం",
    "farm_health": "వ్యవసాయ ఆరోగ్యం",

    "check_leaf": "📷 మల్బరీ ఆకు పరిశీలించండి",
    "check_description":
        "మల్బరీ ఆకు యొక్క స్పష్టమైన చిత్రాన్ని అప్లోడ్ చేయండి. "
        "AI చిత్రాన్ని విశ్లేషించి ఫలితాన్ని చూపిస్తుంది.",

    "upload": "మల్బరీ ఆకు చిత్రాన్ని అప్లోడ్ చేయండి",

    "farm_info": "🗺️ వ్యవసాయ సమాచారం (ఐచ్ఛికం)",
    "farm_zone": "వ్యవసాయ ప్రాంతం / ప్లాట్",
    "farm_zone_placeholder": "ఉదాహరణ: బ్లాక్ A - ప్లాట్ 3",
    "observation": "గమనిక",
    "observation_placeholder": "ఉదాహరణ: గోధుమ రంగు మచ్చలు",

    "diagnosis_section": "🔬 మీ ఆకు వ్యాధి నిర్ధారణ",
    "diagnosis_description": "AI విశ్లేషణ యొక్క ప్రధాన ఫలితం ఇక్కడ ఉంది.",
    "diagnosis": "AI వ్యాధి నిర్ధారణ",
    "confidence": "AI నమ్మకం",
    "severity": "వ్యాధి తీవ్రత",
    "health_score": "ఆకు ఆరోగ్యం",
    "high_confidence": "✓ అధిక నమ్మకంతో AI వర్గీకరణ",
    "moderate_confidence":
        "⚠️ AI నమ్మకం మధ్యస్థంగా ఉంది. క్రింది విజువల్ విశ్లేషణను పరిశీలించండి.",
    "low_confidence":
        "⚠️ AI నమ్మకం తక్కువగా ఉంది. మరో స్పష్టమైన చిత్రాన్ని తీయండి.",

    "what_do": "👨‍🌾 నేను ఏమి చేయాలి?",
    "what_do_desc": "AI ఫలితంపై ఆధారపడిన సులభమైన సిఫార్సులు.",
    "recommended_action": "సిఫార్సు చేసిన చర్య",

    "healthy_rec": [
        "క్రమం తప్పకుండా పర్యవేక్షించండి.",
        "మంచి వ్యవసాయ మరియు ఆకు పరిశుభ్రతను పాటించండి.",
        "చుట్టుపక్కల మొక్కలను క్రమం తప్పకుండా పరిశీలించండి.",
        "వ్యాధి లక్షణాలు కనిపిస్తే ఆకును మళ్లీ పరిశీలించండి."
    ],

    "rust_rec": [
        "ఇలాంటి లక్షణాల కోసం చుట్టుపక్కల ఆకులను పరిశీలించండి.",
        "తీవ్రంగా ప్రభావితమైన ఆకులను తగిన విధంగా తొలగించండి.",
        "మొక్కల మధ్య మంచి గాలి ప్రసరణ ఉండేలా చూడండి.",
        "స్థానికంగా ఆమోదించబడిన వ్యాధి నిర్వహణ మార్గదర్శకాలను పాటించండి.",
        "వ్యాధి పురోగతిని నిరంతరం పర్యవేక్షించండి."
    ],

    "spot_rec": [
        "ఇలాంటి మచ్చల కోసం చుట్టుపక్కల ఆకులను పరిశీలించండి.",
        "తీవ్రంగా ప్రభావితమైన ఆకులను తగిన విధంగా తొలగించండి.",
        "మంచి వ్యవసాయ పరిశుభ్రతను పాటించండి.",
        "స్థానికంగా ఆమోదించబడిన వ్యాధి నిర్వహణ మార్గదర్శకాలను పాటించండి.",
        "వ్యాధి పురోగతిని నిరంతరం పర్యవేక్షించండి."
    ],

    "safety_warning":
        "⚠️ భద్రతా హెచ్చరిక: స్థానికంగా ఆమోదించబడిన వ్యవసాయ మార్గదర్శకాలను "
        "పాటించండి మరియు పట్టుపురుగులకు ఆకులు ఇవ్వడానికి ముందు అవసరమైన భద్రతా కాలాన్ని పాటించండి.",

    "see_ai": "👁️ AI ఏమి చూస్తుందో చూడండి",
    "see_ai_desc":
        "హైలైట్ చేసిన ప్రాంతాలు AI విశ్లేషణకు సహాయపడిన ప్రాంతాలను చూపుతాయి.",
    "original_leaf": "📷 అసలు ఆకు",
    "uploaded_leaf": "అప్లోడ్ చేసిన ఆకు",
    "ai_vision": "🧠 AI విజువల్ విశ్లేషణ",

    "leaf_health": "📊 ఆకు ఆరోగ్యం",
    "disease_risk": "వ్యాధి ప్రమాదం",
    "risk_level": "ప్రమాద స్థాయి",

    "detailed_analysis": "🔎 వివరణాత్మక AI విశ్లేషణ",
    "probabilities": "📈 అంచనా అవకాశాలు",
    "second_opinion": "🧠 AI రెండవ అభిప్రాయం",
    "symptoms": "🔍 గుర్తించిన లక్షణాలు",
    "quality": "📸 చిత్ర నాణ్యత తనిఖీ",
    "environment": "🌦️ పర్యావరణ ప్రమాద అంచనా",
    "progression": "📅 వ్యాధి పురోగతి / పర్యవేక్షణ",
    "before_after": "🔄 ముందు & తరువాత పోలిక",

    "strong_opinion":
        "AI అంచనాపై నమ్మకం ఎక్కువగా ఉంది. అదనపు విజువల్ సమాచారం కోసం "
        "హైలైట్ చేసిన ప్రాంతాలను పరిశీలించండి.",
    "moderate_opinion":
        "AI నమ్మకం మధ్యస్థంగా ఉంది. హైలైట్ చేసిన ప్రాంతాలు మరియు కనిపించే "
        "లక్షణాలతో ఫలితాన్ని పరిశీలించండి.",
    "low_opinion":
        "AI నమ్మకం తక్కువగా ఉంది. ఈ ఫలితాన్ని ఖచ్చితమైనదిగా పరిగణించకుండా మరో స్పష్టమైన చిత్రాన్ని తీయండి.",

    "no_symptoms":
        "వర్గీకరణ మోడల్ ద్వారా ప్రధాన వ్యాధి లక్షణాలు గుర్తించబడలేదు.",
    "rust_symptoms":
        "తుప్పు రంగు లేదా గోధుమ రంగు గాయాల నమూనాలు ఉండవచ్చు. "
        "AI విజువల్ విశ్లేషణ అంచనాకు ప్రభావం చూపిన ప్రాంతాలను చూపిస్తుంది.",
    "spot_symptoms":
        "మచ్చలు లేదా స్థానిక గాయాల నమూనాలు ఉండవచ్చు. "
        "AI విజువల్ విశ్లేషణ అంచనాకు ప్రభావం చూపిన ప్రాంతాలను చూపిస్తుంది.",

    "image_quality": "చిత్ర నాణ్యత",
    "quality_good": "✓ చిత్ర నాణ్యత విశ్లేషణకు అనుకూలంగా ఉంది.",
    "quality_medium": "⚠️ చిత్ర నాణ్యత మధ్యస్థంగా ఉంది.",
    "quality_bad":
        "⚠️ చిత్ర నాణ్యత తక్కువగా ఉంది. మరో చిత్రాన్ని తీయండి.",

    "environment_note":
        "ఇది సూచనాత్మక అంచనా మాత్రమే. ఇది AI మోడల్ అంచనాను మార్చదు.",
    "humidity": "తేమ (%)",
    "temperature": "ఉష్ణోగ్రత (°C)",
    "leaf_wetness": "ఆకు తేమ",
    "low": "తక్కువ",
    "medium": "మధ్యస్థం",
    "high": "ఎక్కువ",
    "environmental_risk": "పర్యావరణ ప్రమాదం",
    "high_environment": "🔴 అధిక పర్యావరణ ప్రమాదం",
    "medium_environment": "🟠 మధ్యస్థ పర్యావరణ ప్రమాదం",
    "low_environment": "🟢 తక్కువ పర్యావరణ ప్రమాదం",

    "monitoring_text":
        "కాలక్రమేణా వ్యాధి తీవ్రతను నమోదు చేయడానికి ఈ విభాగాన్ని ఉపయోగించండి.",
    "monitoring_day": "పర్యవేక్షణ రోజు",
    "observed_severity": "గమనించిన తీవ్రత (%)",
    "save_monitoring": "📌 పర్యవేక్షణ రికార్డును సేవ్ చేయండి",
    "monitor_saved": "✓ పర్యవేక్షణ రికార్డు సేవ్ చేయబడింది.",

    "previous_image": "మునుపటి ఆకు చిత్రాన్ని అప్లోడ్ చేయండి",
    "previous": "మునుపటి",
    "current": "ప్రస్తుత",
    "compare_info":
        "కాలక్రమేణా ఆకు పరిస్థితిని పోల్చడానికి రెండు చిత్రాలను ఉపయోగించండి.",

    "save_analysis": "💾 ఈ విశ్లేషణను సేవ్ చేయండి",
    "save_button": "💾 విశ్లేషణను సేవ్ చేయండి",
    "saved": "✓ విశ్లేషణ విజయవంతంగా సేవ్ చేయబడింది",

    "reports": "📄 నివేదికలు",
    "download_report": "📄 పూర్తి AI నివేదికను డౌన్‌లోడ్ చేయండి",

    "batch": "📦 బ్యాచ్ విశ్లేషణ — అనేక ఆకులు",
    "batch_description":
        "త్వరిత వ్యవసాయ స్థాయి అవలోకనం కోసం అనేక ఆకులను కలిసి విశ్లేషించండి.",
    "multiple_upload": "అనేక ఆకు చిత్రాలను అప్లోడ్ చేయండి",
    "analyze_all": "🚀 అన్ని ఆకులను విశ్లేషించండి",
    "leaves_analyzed": "ఆకులు విశ్లేషించబడ్డాయి.",
    "batch_results": "📊 బ్యాచ్ ఫలితాలను చూడండి",
    "total": "మొత్తం",
    "download_csv": "📥 బ్యాచ్ CSV డౌన్‌లోడ్ చేయండి",

    "farm_analytics": "🌾 వ్యవసాయ విశ్లేషణ",
    "distribution": "🌿 వ్యాధి పంపిణీ",
    "zone_distribution": "🗺️ వ్యవసాయ ప్రాంత పంపిణీ",
    "farm_health_trend": "📈 వ్యవసాయ ఆరోగ్య ధోరణి",
    "history": "📚 విశ్లేషణ చరిత్ర",
    "download_history": "📥 విశ్లేషణ చరిత్రను డౌన్‌లోడ్ చేయండి",

    "analysis_id": "విశ్లేషణ ID",
    "time": "సమయం",
    "zone": "వ్యవసాయ ప్రాంతం",
    "image": "చిత్రం",
    "risk": "వ్యాధి ప్రమాదం",
    "observation_col": "గమనిక",

    "report_title": "ఆకు వ్యాధి విశ్లేషణ నివేదిక",
    "report_prediction": "అంచనా అవకాశాలు",
    "report_opinion": "AI రెండవ అభిప్రాయం",
    "report_symptoms": "గుర్తించిన లక్షణాలు",
    "report_recommendation": "సిఫార్సు చేసిన చర్య",
    "report_health": "ఆరోగ్య స్కోర్",
    "report_risk": "వ్యాధి ప్రమాదం",
    "report_quality": "చిత్ర నాణ్యత",
    "report_safety":
        "భద్రతా హెచ్చరిక: స్థానికంగా ఆమోదించబడిన వ్యవసాయ మార్గదర్శకాలు "
        "మరియు ఉత్పత్తి లేబుల్ అవసరాలను పాటించండి.",
    "generated": "Mulberry AI ద్వారా రూపొందించబడింది",

    "footer":
        "🌿 Mulberry AI • AI ఆధారిత మల్బరీ ఆకు విశ్లేషణ",
    "footer_note":
        "AI ఫలితాలు పరిశీలనకు సహాయపడటానికి మాత్రమే. తగిన వ్యవసాయ మార్గదర్శకంతో ధృవీకరించండి."
},


"Hindi": {

    "title": "🌿 Mulberry AI",
    "subtitle": "AI आधारित शहतूत पत्ती रोग पहचान",

    "settings": "⚙️ सेटिंग्स",
    "language": "भाषा",
    "model": "🤖 AI मॉडल",
    "model_ready": "🟢 मॉडल तैयार है",
    "three_classes": "3 रोग श्रेणियां",

    "detection": "🔬 पहचान",
    "healthy_class": "🟢 स्वस्थ",
    "rust_class": "🟠 पत्ती का रस्ट",
    "spot_class": "🟤 पत्ती के धब्बे",

    "image_tips": "📷 तस्वीर के सुझाव",
    "tip1": "✓ साफ पत्ती",
    "tip2": "✓ अच्छी रोशनी",
    "tip3": "✓ पत्ती को बीच में रखें",
    "tip4": "✓ धुंधली तस्वीर से बचें",

    "session_analyses": "इस सत्र के विश्लेषण",
    "clear_history": "🗑️ इतिहास साफ करें",

    "farm_overview": "🌾 खेत का अवलोकन",
    "leaves_checked": "जाँची गई पत्तियां",
    "healthy": "स्वस्थ",
    "needs_attention": "ध्यान आवश्यक",
    "farm_health": "खेत का स्वास्थ्य",

    "check_leaf": "📷 शहतूत की पत्ती जांचें",
    "check_description":
        "शहतूत की पत्ती की एक साफ तस्वीर अपलोड करें। "
        "AI तस्वीर का विश्लेषण करके नीचे परिणाम दिखाएगा.",

    "upload": "शहतूत की पत्ती की तस्वीर अपलोड करें",

    "farm_info": "🗺️ खेत की जानकारी (वैकल्पिक)",
    "farm_zone": "खेत का क्षेत्र / प्लॉट",
    "farm_zone_placeholder": "उदाहरण: ब्लॉक A - प्लॉट 3",
    "observation": "देखे गए लक्षण",
    "observation_placeholder": "उदाहरण: भूरे धब्बे दिखाई दिए",

    "diagnosis_section": "🔬 आपकी पत्ती का रोग निदान",
    "diagnosis_description":
        "AI विश्लेषण का मुख्य परिणाम यहां है.",
    "diagnosis": "AI रोग निदान",
    "confidence": "AI विश्वास",
    "severity": "रोग की गंभीरता",
    "health_score": "पत्ती का स्वास्थ्य",
    "high_confidence": "✓ उच्च विश्वास वाला AI वर्गीकरण",
    "moderate_confidence":
        "⚠️ AI का विश्वास मध्यम है। नीचे दिए गए दृश्य विश्लेषण की जांच करें.",
    "low_confidence":
        "⚠️ AI का विश्वास कम है। कृपया एक और साफ तस्वीर लें.",

    "what_do": "👨‍🌾 मुझे क्या करना चाहिए?",
    "what_do_desc":
        "AI परिणाम के आधार पर सरल अनुशंसित कदम.",
    "recommended_action": "अनुशंसित कार्रवाई",

    "healthy_rec": [
        "नियमित रूप से निगरानी करते रहें.",
        "अच्छी खेत और पत्ती स्वच्छता बनाए रखें.",
        "आसपास के पौधों की नियमित जांच करें.",
        "लक्षण दिखाई देने पर पत्ती की दोबारा जांच करें."
    ],

    "rust_rec": [
        "इसी तरह के लक्षणों के लिए आसपास की पत्तियों की जांच करें.",
        "जहां उचित हो, अधिक प्रभावित पत्तियों को हटा दें.",
        "पौधों के बीच हवा का अच्छा प्रवाह बनाए रखें.",
        "स्थानीय रूप से अनुमोदित रोग प्रबंधन दिशानिर्देशों का पालन करें.",
        "रोग की प्रगति की निगरानी करते रहें."
    ],

    "spot_rec": [
        "इसी तरह के धब्बों के लिए आसपास की पत्तियों की जांच करें.",
        "जहां उचित हो, अधिक प्रभावित पत्तियों को हटा दें.",
        "अच्छी खेत स्वच्छता बनाए रखें.",
        "स्थानीय रूप से अनुमोदित रोग प्रबंधन दिशानिर्देशों का पालन करें.",
        "रोग की प्रगति की निगरानी करते रहें."
    ],

    "safety_warning":
        "⚠️ सुरक्षा चेतावनी: स्थानीय रूप से अनुमोदित कृषि दिशानिर्देशों का पालन करें "
        "और रेशम के कीड़ों को पत्ती देने से पहले आवश्यक सुरक्षा अवधि का पालन करें.",

    "see_ai": "👁️ AI क्या देख रहा है?",
    "see_ai_desc":
        "हाइलाइट किए गए क्षेत्र वे क्षेत्र दिखाते हैं जिन्होंने AI विश्लेषण में योगदान दिया.",
    "original_leaf": "📷 मूल पत्ती",
    "uploaded_leaf": "अपलोड की गई पत्ती",
    "ai_vision": "🧠 AI दृश्य विश्लेषण",

    "leaf_health": "📊 पत्ती का स्वास्थ्य",
    "disease_risk": "रोग का जोखिम",
    "risk_level": "जोखिम स्तर",

    "detailed_analysis": "🔎 विस्तृत AI विश्लेषण",
    "probabilities": "📈 भविष्यवाणी की संभावनाएं",
    "second_opinion": "🧠 AI दूसरी राय",
    "symptoms": "🔍 देखे गए लक्षण",
    "quality": "📸 तस्वीर की गुणवत्ता जांच",
    "environment": "🌦️ पर्यावरणीय जोखिम मूल्यांकन",
    "progression": "📅 रोग की प्रगति / निगरानी",
    "before_after": "🔄 पहले और बाद की तुलना",

    "strong_opinion":
        "AI भविष्यवाणी पर विश्वास अधिक है। अतिरिक्त दृश्य जानकारी के लिए "
        "हाइलाइट किए गए क्षेत्रों की जांच करें.",
    "moderate_opinion":
        "AI का विश्वास मध्यम है। हाइलाइट किए गए क्षेत्रों और दिखाई देने वाले "
        "लक्षणों के साथ परिणाम की जांच करें.",
    "low_opinion":
        "AI का विश्वास कम है। इस परिणाम को निश्चित न मानें और एक और साफ तस्वीर लें.",

    "no_symptoms":
        "वर्गीकरण मॉडल द्वारा कोई प्रमुख रोग लक्षण नहीं पहचाने गए.",
    "rust_symptoms":
        "जंग जैसे लाल-भूरे या भूरे घाव दिखाई दे सकते हैं। "
        "AI दृश्य विश्लेषण उन क्षेत्रों को दिखाता है जिन्होंने भविष्यवाणी में योगदान दिया.",
    "spot_symptoms":
        "धब्बे या स्थानीय घाव दिखाई दे सकते हैं। "
        "AI दृश्य विश्लेषण उन क्षेत्रों को दिखाता है जिन्होंने भविष्यवाणी में योगदान दिया.",

    "image_quality": "तस्वीर की गुणवत्ता",
    "quality_good": "✓ तस्वीर की गुणवत्ता विश्लेषण के लिए उपयुक्त है.",
    "quality_medium": "⚠️ तस्वीर की गुणवत्ता मध्यम है.",
    "quality_bad":
        "⚠️ तस्वीर की गुणवत्ता खराब है। दूसरी तस्वीर लेने की सलाह दी जाती है.",

    "environment_note":
        "यह केवल संकेतात्मक मूल्यांकन है। यह AI मॉडल की भविष्यवाणी को नहीं बदलता.",
    "humidity": "नमी (%)",
    "temperature": "तापमान (°C)",
    "leaf_wetness": "पत्ती की नमी",
    "low": "कम",
    "medium": "मध्यम",
    "high": "अधिक",
    "environmental_risk": "पर्यावरणीय जोखिम",
    "high_environment": "🔴 उच्च पर्यावरणीय जोखिम",
    "medium_environment": "🟠 मध्यम पर्यावरणीय जोखिम",
    "low_environment": "🟢 कम पर्यावरणीय जोखिम",

    "monitoring_text":
        "समय के साथ रोग की गंभीरता दर्ज करने के लिए इस भाग का उपयोग करें.",
    "monitoring_day": "निगरानी का दिन",
    "observed_severity": "देखी गई गंभीरता (%)",
    "save_monitoring": "📌 निगरानी रिकॉर्ड सेव करें",
    "monitor_saved": "✓ निगरानी रिकॉर्ड सेव हो गया.",

    "previous_image": "पिछली पत्ती की तस्वीर अपलोड करें",
    "previous": "पिछली",
    "current": "वर्तमान",
    "compare_info":
        "समय के साथ पत्ती की स्थिति की तुलना करने के लिए दोनों तस्वीरों का उपयोग करें.",

    "save_analysis": "💾 यह विश्लेषण सेव करें",
    "save_button": "💾 विश्लेषण सेव करें",
    "saved": "✓ विश्लेषण सफलतापूर्वक सेव हो गया",

    "reports": "📄 रिपोर्ट",
    "download_report": "📄 पूरी AI रिपोर्ट डाउनलोड करें",

    "batch": "📦 बैच विश्लेषण — कई पत्तियां",
    "batch_description":
        "खेत का त्वरित अवलोकन प्राप्त करने के लिए कई पत्तियों का एक साथ विश्लेषण करें.",
    "multiple_upload": "कई पत्तियों की तस्वीरें अपलोड करें",
    "analyze_all": "🚀 सभी पत्तियों का विश्लेषण करें",
    "leaves_analyzed": "पत्तियों का विश्लेषण किया गया.",
    "batch_results": "📊 बैच परिणाम देखें",
    "total": "कुल",
    "download_csv": "📥 बैच CSV डाउनलोड करें",

    "farm_analytics": "🌾 खेत का विश्लेषण",
    "distribution": "🌿 रोग वितरण",
    "zone_distribution": "🗺️ खेत क्षेत्र वितरण",
    "farm_health_trend": "📈 खेत के स्वास्थ्य की प्रवृत्ति",
    "history": "📚 विश्लेषण इतिहास",
    "download_history": "📥 विश्लेषण इतिहास डाउनलोड करें",

    "analysis_id": "विश्लेषण ID",
    "time": "समय",
    "zone": "खेत का क्षेत्र",
    "image": "तस्वीर",
    "risk": "रोग का जोखिम",
    "observation_col": "देखे गए लक्षण",

    "report_title": "पत्ती रोग विश्लेषण रिपोर्ट",
    "report_prediction": "भविष्यवाणी की संभावनाएं",
    "report_opinion": "AI दूसरी राय",
    "report_symptoms": "देखे गए लक्षण",
    "report_recommendation": "अनुशंसित कार्रवाई",
    "report_health": "स्वास्थ्य स्कोर",
    "report_risk": "रोग का जोखिम",
    "report_quality": "तस्वीर की गुणवत्ता",
    "report_safety":
        "सुरक्षा चेतावनी: स्थानीय रूप से अनुमोदित कृषि दिशानिर्देशों "
        "और उत्पाद लेबल आवश्यकताओं का पालन करें.",
    "generated": "Mulberry AI द्वारा तैयार किया गया",

    "footer":
        "🌿 Mulberry AI • AI आधारित शहतूत पत्ती विश्लेषण",
    "footer_note":
        "AI परिणाम केवल निरीक्षण में सहायता के लिए हैं और उचित कृषि मार्गदर्शन के साथ सत्यापित किए जाने चाहिए."
}

}


# ============================================================
# LANGUAGE SELECTOR
# ============================================================

st.sidebar.markdown(
    "## ⚙️ Settings"
)

language = st.sidebar.selectbox(
    "Language / भाषा / ಭಾಷೆ / భాష",
    [
        "English",
        "Kannada",
        "Telugu",
        "Hindi"
    ]
)

t = LANG[language]


# ============================================================
# SESSION STATE
# ============================================================

if "analysis_history" not in st.session_state:
    st.session_state.analysis_history = []

if "progress_history" not in st.session_state:
    st.session_state.progress_history = []

if "batch_results" not in st.session_state:
    st.session_state.batch_results = []


# ============================================================
# MODEL
# DO NOT CHANGE
# ============================================================

@st.cache_resource
def load_model():

    model_path = os.path.join(
        "model",
        "efficientnetb0_final.keras"
    )

    return tf.keras.models.load_model(
        model_path
    )


model = None


# ============================================================
# CLASS NAMES
# DO NOT CHANGE
# ============================================================

CLASS_NAMES = [
    "Healthy",
    "Leaf Rust",
    "Leaf Spot"
]


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    f"### {t['model']}"
)

st.sidebar.success(
    t["model_ready"]
)

st.sidebar.caption(
    "EfficientNetB0"
)

st.sidebar.caption(
    t["three_classes"]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"### {t['detection']}"
)

st.sidebar.write(
    t["healthy_class"]
)

st.sidebar.write(
    t["rust_class"]
)

st.sidebar.write(
    t["spot_class"]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    f"### {t['image_tips']}"
)

st.sidebar.write(t["tip1"])
st.sidebar.write(t["tip2"])
st.sidebar.write(t["tip3"])
st.sidebar.write(t["tip4"])

st.sidebar.markdown("---")

st.sidebar.metric(
    t["session_analyses"],
    len(
        st.session_state.analysis_history
    )
)

if st.sidebar.button(
    t["clear_history"]
):

    st.session_state.analysis_history = []
    st.session_state.progress_history = []
    st.session_state.batch_results = []

    st.rerun()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_image_quality(img):

    image_array = np.array(
        img
    ).astype(np.float32)

    brightness = np.mean(
        image_array
    )

    grayscale = np.mean(
        image_array,
        axis=2
    )

    sharpness = np.var(
        np.gradient(grayscale)
    )

    brightness_score = (
        100
        - abs(brightness - 128)
        / 128
        * 100
    )

    sharpness_score = min(
        sharpness / 2,
        100
    )

    quality = (
        brightness_score * 0.5
        + sharpness_score * 0.5
    )

    return max(
        0,
        min(
            100,
            quality
        )
    )


def calculate_health_score(
    label,
    confidence,
    severity
):

    if label == "Healthy":

        score = (
            75
            + confidence * 0.25
            - severity * 0.15
        )

    else:

        score = (
            100
            - severity * 0.80
            + (100 - confidence) * 0.10
        )

    return max(
        0,
        min(
            100,
            score
        )
    )


def calculate_disease_risk(
    label,
    confidence,
    severity
):

    if label == "Healthy":

        risk = (
            (100 - confidence) * 0.30
            + severity * 0.20
        )

    else:

        risk = (
            severity * 0.70
            + confidence * 0.30
        )

    return max(
        0,
        min(
            100,
            risk
        )
    )


def get_risk_level(score):

    if score >= 70:
        return "High"

    if score >= 40:
        return "Medium"

    return "Low"


def create_analysis_id():

    return (
        "MAI-"
        + datetime.now().strftime("%Y%m%d")
        + "-"
        + uuid.uuid4().hex[:6].upper()
    )


# ============================================================
# PDF FONT
# ============================================================

def get_pdf_font():

    # Use a Unicode font that supports the selected language.
    # The old version could fall back to Helvetica, which does not
    # contain Hindi/Kannada/Telugu glyphs and caused black squares.
    font_candidates = {

        "Hindi": [
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansDevanagariUI-Regular.ttf",
            "C:\\Windows\\Fonts\\Nirmala.ttf",
            "C:\\Windows\\Fonts\\NirmalaUI.ttf"
        ],

        "Kannada": [
            "/usr/share/fonts/truetype/noto/NotoSansKannada-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansKannadaUI-Regular.ttf",
            "C:\\Windows\\Fonts\\Nirmala.ttf",
            "C:\\Windows\\Fonts\\NirmalaUI.ttf"
        ],

        "Telugu": [
            "/usr/share/fonts/truetype/noto/NotoSansTelugu-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSansTeluguUI-Regular.ttf",
            "C:\\Windows\\Fonts\\Nirmala.ttf",
            "C:\\Windows\\Fonts\\NirmalaUI.ttf"
        ],

        "English": [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "C:\\Windows\\Fonts\\arial.ttf"
        ]
    }

    candidates = font_candidates.get(
        language,
        font_candidates["English"]
    )

    for path in candidates:

        if os.path.exists(path):

            font_name = (
                "MulberryFont_"
                + language.replace(
                    " ",
                    "_"
                )
            )

            try:

                # Register once. This prevents duplicate-registration
                # problems when Streamlit reruns the script.
                if font_name not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(
                        TTFont(
                            font_name,
                            path
                        )
                    )

                return font_name

            except Exception:
                pass

    return "Helvetica"


# ============================================================
# PDF REPORT
# ============================================================

def create_pdf_report(
    img,
    label,
    confidence,
    severity,
    sev_text,
    ai_opinion,
    symptoms,
    recommendations,
    probabilities,
    health_score,
    disease_risk,
    quality_score,
    analysis_id,
    farm_zone
):

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    font_name = get_pdf_font()

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "MulberryTitle",
        parent=styles["Title"],
        fontName=font_name,
        alignment=TA_CENTER,
        fontSize=20,
        leading=25
    )

    heading_style = ParagraphStyle(
        "MulberryHeading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=14,
        leading=19
    )

    body_style = ParagraphStyle(
        "MulberryBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=10,
        leading=15
    )

    story = []

    # Do not use emoji or <b> tags here. Unsupported emoji glyphs and
    # automatic Helvetica-Bold fallback were causing black squares.
    story.append(
        Paragraph(
            "Mulberry AI",
            title_style
        )
    )

    story.append(
        Paragraph(
            html.escape(
                t["report_title"]
            ),
            heading_style
        )
    )

    story.append(
        Spacer(
            1,
            15
        )
    )

    image_buffer = io.BytesIO()

    img.save(
        image_buffer,
        format="JPEG"
    )

    image_buffer.seek(0)

    story.append(
        PDFImage(
            image_buffer,
            width=4.5 * inch,
            height=4.5 * inch
        )
    )

    story.append(
        Spacer(
            1,
            15
        )
    )

    translated_label = label

    if label == "Healthy":
        translated_label = t["healthy"]
    elif label == "Leaf Rust":
        translated_label = t["rust_class"]
    elif label == "Leaf Spot":
        translated_label = t["spot_class"]

    translated_severity = {
        "Low": t["low"],
        "Medium": t["medium"],
        "High": t["high"]
    }.get(
        sev_text,
        sev_text
    )

    translated_not_specified = {
        "English": "Not specified",
        "Hindi": "निर्दिष्ट नहीं",
        "Kannada": "ನಿರ್ದಿಷ್ಟಪಡಿಸಲಾಗಿಲ್ಲ",
        "Telugu": "పేర్కొనబడలేదు"
    }.get(
        language,
        "Not specified"
    )

    display_zone = (
        farm_zone
        if farm_zone
        else translated_not_specified
    )

    details = [

        (
            t["analysis_id"],
            analysis_id
        ),

        (
            t["zone"],
            display_zone
        ),

        (
            t["diagnosis"],
            translated_label
        ),

        (
            t["confidence"],
            f"{confidence:.2f}%"
        ),

        (
            t["severity"],
            f"{severity:.1f}% ({translated_severity})"
        ),

        (
            t["report_health"],
            f"{health_score:.1f}/100"
        ),

        (
            t["report_risk"],
            f"{disease_risk:.1f}/100"
        ),

        (
            t["report_quality"],
            f"{quality_score:.1f}%"
        )
    ]

    for name, value in details:

        # Avoid <b> because ReportLab can switch to Helvetica-Bold,
        # which does not contain Indic characters.
        story.append(
            Paragraph(
                f"{html.escape(str(name))}: "
                f"{html.escape(str(value))}",
                body_style
            )
        )

        story.append(
            Spacer(
                1,
                7
            )
        )


    story.append(
        Spacer(
            1,
            8
        )
    )

    story.append(
        Paragraph(
            html.escape(
                t["report_prediction"]
            ),
            heading_style
        )
    )

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities
    ):

        translated_class = class_name

        if class_name == "Healthy":
            translated_class = t["healthy"]
        elif class_name == "Leaf Rust":
            translated_class = t["rust_class"]
        elif class_name == "Leaf Spot":
            translated_class = t["spot_class"]

        story.append(
            Paragraph(
                f"{html.escape(translated_class)}: "
                f"{float(probability):.2f}%",
                body_style
            )
        )


    story.append(
        Spacer(
            1,
            12
        )
    )

    story.append(
        Paragraph(
            html.escape(
                t["report_opinion"]
            ),
            heading_style
        )
    )

    story.append(
        Paragraph(
            html.escape(
                ai_opinion
            ),
            body_style
        )
    )


    story.append(
        Spacer(
            1,
            12
        )
    )

    story.append(
        Paragraph(
            html.escape(
                t["report_symptoms"]
            ),
            heading_style
        )
    )

    story.append(
        Paragraph(
            html.escape(
                symptoms
            ),
            body_style
        )
    )


    story.append(
        Spacer(
            1,
            12
        )
    )

    story.append(
        Paragraph(
            html.escape(
                t["report_recommendation"]
            ),
            heading_style
        )
    )

    for number, recommendation in enumerate(
        recommendations,
        start=1
    ):

        story.append(
            Paragraph(
                f"{number}. "
                f"{html.escape(recommendation)}",
                body_style
            )
        )

        story.append(
            Spacer(
                1,
                5
            )
        )


    story.append(
        Spacer(
            1,
            12
        )
    )

    story.append(
        Paragraph(
            html.escape(
                t["report_safety"]
            ),
            body_style
        )
    )

    story.append(
        Spacer(
            1,
            18
        )
    )

    story.append(
        Paragraph(
            html.escape(
                t["generated"]
            ),
            body_style
        )
    )

    doc.build(
        story
    )

    buffer.seek(0)

    return buffer


# ============================================================
# HERO
# ============================================================

st.markdown(
    f"""
<div class="hero">

<div class="hero-title">
{t["title"]}
</div>

<div class="hero-subtitle">
{t["subtitle"]}
</div>

<div class="hero-status">
🟢 AI System Ready &nbsp; • &nbsp;
🔬 3 Disease Classes &nbsp; • &nbsp;
📱 Farmer Friendly
</div>

</div>
""",
    unsafe_allow_html=True
)


# ============================================================
# FARM OVERVIEW
# ============================================================

if len(
    st.session_state.analysis_history
) > 0:

    st.markdown(
        f'<div class="section-title">{t["farm_overview"]}</div>',
        unsafe_allow_html=True
    )

    history_df = pd.DataFrame(
        st.session_state.analysis_history
    )

    total = len(
        history_df
    )

    healthy = len(
        history_df[
            history_df["Diagnosis"]
            == "Healthy"
        ]
    )

    diseased = (
        total
        - healthy
    )

    avg_health = history_df[
        "Health Score"
    ].mean()

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            t["leaves_checked"],
            total
        )

    with c2:

        st.metric(
            t["healthy"],
            healthy
        )

    with c3:

        st.metric(
            t["needs_attention"],
            diseased
        )

    with c4:

        st.metric(
            t["farm_health"],
            f"{avg_health:.0f}/100"
        )


# ============================================================
# UPLOAD
# ============================================================

st.markdown(
    f'<div class="section-title">{t["check_leaf"]}</div>',
    unsafe_allow_html=True
)

st.markdown(
    f'<div class="section-description">{t["check_description"]}</div>',
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    t["upload"],
    type=[
        "jpg",
        "png",
        "jpeg"
    ]
)


# ============================================================
# IMAGE ANALYSIS
# ============================================================

if uploaded:

    img = Image.open(
        uploaded
    ).convert("RGB")


    # ========================================================
    # FARM INFORMATION
    # ========================================================

    with st.expander(
        t["farm_info"]
    ):

        info1, info2 = st.columns(2)

        with info1:

            farm_zone = st.text_input(
                t["farm_zone"],
                placeholder=t[
                    "farm_zone_placeholder"
                ]
            )

        with info2:

            observation_note = st.text_input(
                t["observation"],
                placeholder=t[
                    "observation_placeholder"
                ]
            )

    if "farm_zone" not in locals():
        farm_zone = ""

    if "observation_note" not in locals():
        observation_note = ""


    # ========================================================
    # ORIGINAL PREPROCESSING
    # ========================================================

    img_resized = img.resize(
        (224, 224)
    )

    img_array = np.array(
        img_resized
    ) / 255.0

    img_array = np.expand_dims(
        img_array,
        axis=0
    ).astype(
        np.float32
    )


    # ========================================================
    # ORIGINAL MODEL PREDICTION
    # ========================================================

    preds = model.predict(
        img_array
    )

    idx = np.argmax(
        preds
    )

    label = CLASS_NAMES[
        idx
    ]

    confidence = float(
        np.max(preds) * 100
    )


    # ========================================================
    # ORIGINAL GRAD-CAM
    # ========================================================

    heatmap = get_gradcam_heatmap(
        img_array,
        model
    )

    overlay = overlay_gradcam(
        np.array(img),
        heatmap,
        alpha=0.45
    )


    # ========================================================
    # ORIGINAL SEVERITY
    # ========================================================

    pct, sev_text, emoji = (
        calculate_severity_percentage(
            heatmap
        )
    )


    # ========================================================
    # ADDITIONAL FEATURES
    # ========================================================

    health_score = calculate_health_score(
        label,
        confidence,
        pct
    )

    disease_risk = calculate_disease_risk(
        label,
        confidence,
        pct
    )

    risk_level = get_risk_level(
        disease_risk
    )

    quality_score = calculate_image_quality(
        img
    )

    analysis_id = create_analysis_id()


    # ========================================================
    # DIAGNOSIS
    # ========================================================

    st.markdown("---")

    st.markdown(
        f'<div class="section-title">{t["diagnosis_section"]}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="section-description">{t["diagnosis_description"]}</div>',
        unsafe_allow_html=True
    )

    result1, result2 = st.columns(
        [1.6, 1]
    )

    with result1:

        st.markdown(
            f"""
<div class="diagnosis-card">

<div class="diagnosis-small">
{t["diagnosis"]}
</div>

<div class="diagnosis-name">
{emoji} {label}
</div>

<div class="diagnosis-confidence">
{t["confidence"]}: <b>{confidence:.2f}%</b>
</div>

</div>
""",
            unsafe_allow_html=True
        )

    with result2:

        st.metric(
            t["severity"],
            f"{pct:.1f}%"
        )

        st.metric(
            t["health_score"],
            f"{health_score:.0f}/100"
        )


    # ========================================================
    # CONFIDENCE
    # ========================================================

    if confidence < 50:

        st.error(
            t["low_confidence"]
        )

    elif confidence < 80:

        st.warning(
            t["moderate_confidence"]
        )

    else:

        st.success(
            t["high_confidence"]
        )


    # ========================================================
    # RECOMMENDATIONS
    # ========================================================

    st.markdown("---")

    st.markdown(
        f'<div class="section-title">{t["what_do"]}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="section-description">{t["what_do_desc"]}</div>',
        unsafe_allow_html=True
    )


    if label == "Healthy":

        recommendations = t[
            "healthy_rec"
        ]

    elif label == "Leaf Rust":

        recommendations = t[
            "rust_rec"
        ]

    else:

        recommendations = t[
            "spot_rec"
        ]


    recommendation_html = f"""
<div class="action-card">

<div class="action-title">
{t["recommended_action"]}
</div>

<div style="margin-top:18px;">
"""


    for number, recommendation in enumerate(
        recommendations,
        start=1
    ):

        recommendation_html += f"""
<div class="recommendation-step">

<div class="recommendation-number">
{number}
</div>

<div class="recommendation-text">
{recommendation}
</div>

</div>
"""


    recommendation_html += """
</div>
</div>
"""


    st.markdown(
        recommendation_html,
        unsafe_allow_html=True
    )


    if label != "Healthy":

        st.warning(
            t["safety_warning"]
        )


    # ========================================================
    # AI VISION
    # ========================================================

    st.markdown("---")

    st.markdown(
        f'<div class="section-title">{t["see_ai"]}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<div class="section-description">{t["see_ai_desc"]}</div>',
        unsafe_allow_html=True
    )

    image1, image2 = st.columns(2)

    with image1:

        st.markdown(
            f"### {t['original_leaf']}"
        )

        st.image(
            img,
            caption=t["uploaded_leaf"],
            use_column_width=True
        )

    with image2:

        st.markdown(
            f"### {t['ai_vision']}"
        )

        st.image(
            overlay,
            caption=t["ai_vision"],
            use_column_width=True
        )


    # ========================================================
    # HEALTH / RISK
    # ========================================================

    st.markdown("---")

    st.markdown(
        f'<div class="section-title">{t["leaf_health"]}</div>',
        unsafe_allow_html=True
    )

    h1, h2, h3 = st.columns(3)

    with h1:

        st.metric(
            t["health_score"],
            f"{health_score:.1f}/100"
        )

    with h2:

        st.metric(
            t["disease_risk"],
            f"{disease_risk:.1f}/100"
        )

    with h3:

        display_risk = {
            "Low": t["low"],
            "Medium": t["medium"],
            "High": t["high"]
        }.get(
            risk_level,
            risk_level
        )

        st.metric(
            t["risk_level"],
            display_risk
        )

    st.progress(
        min(
            health_score / 100,
            1
        )
    )


    # ========================================================
    # DETAILED ANALYSIS
    # ========================================================

    st.markdown("---")

    st.markdown(
        f'<div class="section-title">{t["detailed_analysis"]}</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    with st.expander(
        t["probabilities"]
    ):

        for i, class_name in enumerate(
            CLASS_NAMES
        ):

            probability = float(
                preds[0][i] * 100
            )

            translated_class = class_name

            if class_name == "Healthy":
                translated_class = t["healthy"]

            elif class_name == "Leaf Rust":
                translated_class = t["rust_class"]

            elif class_name == "Leaf Spot":
                translated_class = t["spot_class"]

            st.write(
                f"**{translated_class}: "
                f"{probability:.2f}%**"
            )

            st.progress(
                min(
                    probability / 100,
                    1
                )
            )


    # ========================================================
    # SECOND OPINION
    # ========================================================

    with st.expander(
        t["second_opinion"]
    ):

        if confidence >= 80:

            ai_opinion = t[
                "strong_opinion"
            ]

            st.success(
                ai_opinion
            )

        elif confidence >= 50:

            ai_opinion = t[
                "moderate_opinion"
            ]

            st.warning(
                ai_opinion
            )

        else:

            ai_opinion = t[
                "low_opinion"
            ]

            st.error(
                ai_opinion
            )


    # ========================================================
    # SYMPTOMS
    # ========================================================

    with st.expander(
        t["symptoms"]
    ):

        if label == "Healthy":

            symptoms = t[
                "no_symptoms"
            ]

        elif label == "Leaf Rust":

            symptoms = t[
                "rust_symptoms"
            ]

        else:

            symptoms = t[
                "spot_symptoms"
            ]

        st.write(
            symptoms
        )


    # ========================================================
    # IMAGE QUALITY
    # ========================================================

    with st.expander(
        t["quality"]
    ):

        st.metric(
            t["image_quality"],
            f"{quality_score:.1f}%"
        )

        st.progress(
            min(
                quality_score / 100,
                1
            )
        )

        if quality_score >= 70:

            st.success(
                t["quality_good"]
            )

        elif quality_score >= 45:

            st.warning(
                t["quality_medium"]
            )

        else:

            st.error(
                t["quality_bad"]
            )


    # ========================================================
    # ENVIRONMENT
    # ========================================================

    with st.expander(
        t["environment"]
    ):

        st.caption(
            t["environment_note"]
        )

        e1, e2, e3 = st.columns(3)

        with e1:

            humidity = st.slider(
                t["humidity"],
                0,
                100,
                70
            )

        with e2:

            temperature = st.slider(
                t["temperature"],
                0,
                50,
                25
            )

        with e3:

            leaf_wetness = st.selectbox(
                t["leaf_wetness"],
                [
                    t["low"],
                    t["medium"],
                    t["high"]
                ]
            )

        environmental_score = 0

        if humidity >= 80:
            environmental_score += 40

        elif humidity >= 65:
            environmental_score += 20

        if 20 <= temperature <= 30:
            environmental_score += 30

        elif 15 <= temperature <= 35:
            environmental_score += 15

        if leaf_wetness == t["high"]:
            environmental_score += 30

        elif leaf_wetness == t["medium"]:
            environmental_score += 15

        environmental_score = min(
            environmental_score,
            100
        )

        environmental_level = get_risk_level(
            environmental_score
        )

        st.metric(
            t["environmental_risk"],
            f"{environmental_score}%"
        )

        if environmental_level == "High":

            st.error(
                t["high_environment"]
            )

        elif environmental_level == "Medium":

            st.warning(
                t["medium_environment"]
            )

        else:

            st.success(
                t["low_environment"]
            )


    # ========================================================
    # MONITORING
    # ========================================================

    with st.expander(
        t["progression"]
    ):

        st.write(
            t["monitoring_text"]
        )

        monitoring_day = st.selectbox(
            t["monitoring_day"],
            [
                "Day 1",
                "Day 2",
                "Day 3",
                "Day 4",
                "Day 5",
                "Day 6",
                "Day 7",
                "Day 14",
                "Day 30"
            ]
        )

        monitoring_severity = st.number_input(
            t["observed_severity"],
            min_value=0.0,
            max_value=100.0,
            value=float(
                round(
                    pct,
                    1
                )
            ),
            step=0.1
        )

        if st.button(
            t["save_monitoring"]
        ):

            st.session_state.progress_history.append(
                {
                    "Day":
                        monitoring_day,

                    "Severity":
                        monitoring_severity,

                    "Diagnosis":
                        label,

                    "Time":
                        datetime.now().strftime(
                            "%d-%m-%Y %H:%M"
                        )
                }
            )

            st.success(
                t["monitor_saved"]
            )

            st.rerun()


        if len(
            st.session_state.progress_history
        ) > 0:

            progress_df = pd.DataFrame(
                st.session_state.progress_history
            )

            chart_df = progress_df[
                [
                    "Day",
                    "Severity"
                ]
            ].copy()

            day_order = [
                "Day 1",
                "Day 2",
                "Day 3",
                "Day 4",
                "Day 5",
                "Day 6",
                "Day 7",
                "Day 14",
                "Day 30"
            ]

            chart_df["Day"] = pd.Categorical(
                chart_df["Day"],
                categories=day_order,
                ordered=True
            )

            chart_df = chart_df.sort_values(
                "Day"
            )

            chart_df = chart_df.drop_duplicates(
                subset=["Day"],
                keep="last"
            )

            chart_df = chart_df.set_index(
                "Day"
            )

            st.line_chart(
                chart_df
            )


    # ========================================================
    # BEFORE / AFTER
    # ========================================================

    with st.expander(
        t["before_after"]
    ):

        previous_files = st.file_uploader(
            t["previous_image"],
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            key="before_after"
        )

        if previous_files:

            previous_img = Image.open(
                previous_files
            ).convert("RGB")

            before_col, after_col = st.columns(2)

            with before_col:

                st.markdown(
                    f"### {t['previous']}"
                )

                st.image(
                    previous_img,
                    use_column_width=True
                )

            with after_col:

                st.markdown(
                    f"### {t['current']}"
                )

                st.image(
                    img,
                    use_column_width=True
                )

            st.info(
                t["compare_info"]
            )


    # ========================================================
    # SAVE ANALYSIS
    # ========================================================

    st.markdown("---")

    st.markdown(
        f'<div class="section-title">{t["save_analysis"]}</div>',
        unsafe_allow_html=True
    )

    if st.button(
        t["save_button"]
    ):

        record = {

            t["analysis_id"]:
                analysis_id,

            t["time"]:
                datetime.now().strftime(
                    "%d-%m-%Y %H:%M"
                ),

            t["zone"]:
                farm_zone
                if farm_zone
                else "Not specified",

            "Diagnosis":
                label,

            t["confidence"]:
                round(
                    confidence,
                    2
                ),

            t["severity"]:
                round(
                    pct,
                    2
                ),

            "Health Score":
                round(
                    health_score,
                    2
                ),

            t["disease_risk"]:
                round(
                    disease_risk,
                    2
                ),

            t["image_quality"]:
                round(
                    quality_score,
                    2
                ),

            t["observation_col"]:
                observation_note
        }

        st.session_state.analysis_history.append(
            record
        )

        st.success(
            f"{t['saved']} — {analysis_id}"
        )


    # ========================================================
    # PDF REPORT
    # ========================================================

    st.markdown("---")

    st.markdown(
        f'<div class="section-title">{t["reports"]}</div>',
        unsafe_allow_html=True
    )

    pdf_file = create_pdf_report(
        img,
        label,
        confidence,
        pct,
        sev_text,
        ai_opinion,
        symptoms,
        recommendations,
        preds[0] * 100,
        health_score,
        disease_risk,
        quality_score,
        analysis_id,
        farm_zone
        if farm_zone
        else "Not specified"
    )

    st.download_button(
        t["download_report"],
        data=pdf_file,
        file_name=(
            f"mulberry_ai_{analysis_id}.pdf"
        ),
        mime="application/pdf"
    )


# ============================================================
# BATCH ANALYSIS
# ============================================================

st.markdown("---")

with st.expander(
    t["batch"]
):

    st.write(
        t["batch_description"]
    )

    batch_files = st.file_uploader(
        t["multiple_upload"],
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        accept_multiple_files=True,
        key="batch_uploader"
    )

    if batch_files:

        if st.button(
            t["analyze_all"]
        ):

            batch_results = []

            progress_bar = st.progress(
                0
            )

            total_files = len(
                batch_files
            )

            for number, batch_file in enumerate(
                batch_files,
                start=1
            ):

                try:

                    batch_img = Image.open(
                        batch_file
                    ).convert("RGB")

                    batch_resized = batch_img.resize(
                        (224, 224)
                    )

                    batch_array = (
                        np.array(
                            batch_resized
                        ) / 255.0
                    )

                    batch_array = np.expand_dims(
                        batch_array,
                        axis=0
                    ).astype(
                        np.float32
                    )

                    batch_prediction = model.predict(
                        batch_array,
                        verbose=0
                    )

                    batch_index = np.argmax(
                        batch_prediction
                    )

                    batch_label = CLASS_NAMES[
                        batch_index
                    ]

                    batch_confidence = float(
                        np.max(
                            batch_prediction
                        ) * 100
                    )

                    batch_heatmap = (
                        get_gradcam_heatmap(
                            batch_array,
                            model
                        )
                    )

                    batch_pct, _, _ = (
                        calculate_severity_percentage(
                            batch_heatmap
                        )
                    )

                    batch_health = (
                        calculate_health_score(
                            batch_label,
                            batch_confidence,
                            batch_pct
                        )
                    )

                    batch_risk = (
                        calculate_disease_risk(
                            batch_label,
                            batch_confidence,
                            batch_pct
                        )
                    )

                    batch_results.append(
                        {
                            t["image"]:
                                batch_file.name,

                            "Diagnosis":
                                batch_label,

                            t["confidence"]:
                                round(
                                    batch_confidence,
                                    2
                                ),

                            t["severity"]:
                                round(
                                    batch_pct,
                                    2
                                ),

                            "Health Score":
                                round(
                                    batch_health,
                                    2
                                ),

                            t["disease_risk"]:
                                round(
                                    batch_risk,
                                    2
                                )
                        }
                    )

                except Exception:

                    batch_results.append(
                        {
                            t["image"]:
                                batch_file.name,

                            "Diagnosis":
                                "Analysis Error",

                            t["confidence"]:
                                0,

                            t["severity"]:
                                0,

                            "Health Score":
                                0,

                            t["disease_risk"]:
                                0
                        }
                    )

                progress_bar.progress(
                    number / total_files
                )

            st.session_state.batch_results = (
                batch_results
            )

            st.success(
                f"✓ {total_files} {t['leaves_analyzed']}"
            )


# ============================================================
# BATCH RESULTS
# ============================================================

if len(
    st.session_state.batch_results
) > 0:

    with st.expander(
        t["batch_results"]
    ):

        batch_df = pd.DataFrame(
            st.session_state.batch_results
        )

        st.dataframe(
            batch_df,
            use_container_width=True
        )

        healthy_batch = len(
            batch_df[
                batch_df["Diagnosis"]
                == "Healthy"
            ]
        )

        rust_batch = len(
            batch_df[
                batch_df["Diagnosis"]
                == "Leaf Rust"
            ]
        )

        spot_batch = len(
            batch_df[
                batch_df["Diagnosis"]
                == "Leaf Spot"
            ]
        )

        b1, b2, b3, b4 = st.columns(4)

        with b1:

            st.metric(
                t["total"],
                len(batch_df)
            )

        with b2:

            st.metric(
                t["healthy"],
                healthy_batch
            )

        with b3:

            st.metric(
                "Leaf Rust",
                rust_batch
            )

        with b4:

            st.metric(
                "Leaf Spot",
                spot_batch
            )

        batch_csv = (
            batch_df
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )

        st.download_button(
            t["download_csv"],
            data=batch_csv,
            file_name="mulberry_batch_analysis.csv",
            mime="text/csv"
        )


# ============================================================
# FARM ANALYTICS
# ============================================================

if len(
    st.session_state.analysis_history
) > 0:

    st.markdown("---")

    st.markdown(
        f'<div class="section-title">{t["farm_analytics"]}</div>',
        unsafe_allow_html=True
    )


    # ========================================================
    # DISEASE DISTRIBUTION
    # ========================================================

    with st.expander(
        t["distribution"]
    ):

        disease_counts = (
            history_df[
                "Diagnosis"
            ]
            .value_counts()
        )

        d1, d2, d3 = st.columns(3)

        with d1:

            st.metric(
                t["healthy"],
                int(
                    disease_counts.get(
                        "Healthy",
                        0
                    )
                )
            )

        with d2:

            st.metric(
                t["rust_class"],
                int(
                    disease_counts.get(
                        "Leaf Rust",
                        0
                    )
                )
            )

        with d3:

            st.metric(
                t["spot_class"],
                int(
                    disease_counts.get(
                        "Leaf Spot",
                        0
                    )
                )
            )

        disease_chart = (
            disease_counts
            .reindex(
                [
                    "Healthy",
                    "Leaf Rust",
                    "Leaf Spot"
                ],
                fill_value=0
            )
            .to_frame(
                name=t["leaves_checked"]
            )
        )

        st.bar_chart(
            disease_chart
        )


    # ========================================================
    # ZONES
    # ========================================================

    with st.expander(
        t["zone_distribution"]
    ):

        zone_counts = (
            history_df[
                t["zone"]
            ]
            .value_counts()
            .reset_index()
        )

        zone_counts.columns = [
            t["zone"],
            t["leaves_checked"]
        ]

        if len(zone_counts) == 1:

            z1, z2 = st.columns(2)

            with z1:

                st.metric(
                    t["zone"],
                    zone_counts.iloc[0][
                        t["zone"]
                    ]
                )

            with z2:

                st.metric(
                    t["leaves_checked"],
                    int(
                        zone_counts.iloc[0][
                            t["leaves_checked"]
                        ]
                    )
                )

        else:

            st.bar_chart(
                zone_counts.set_index(
                    t["zone"]
                )
            )


    # ========================================================
    # HEALTH TREND
    # ========================================================

    with st.expander(
        t["farm_health_trend"]
    ):

        health_chart = history_df[
            [
                t["time"],
                "Health Score"
            ]
        ].copy()

        health_chart = (
            health_chart
            .set_index(
                t["time"]
            )
        )

        st.line_chart(
            health_chart
        )


    # ========================================================
    # HISTORY
    # ========================================================

    with st.expander(
        t["history"]
    ):

        st.dataframe(
            history_df,
            use_container_width=True
        )

        history_csv = (
            history_df
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )

        st.download_button(
            t["download_history"],
            data=history_csv,
            file_name="mulberry_ai_analysis_history.csv",
            mime="text/csv"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    t["footer"]
)

st.caption(
    t["footer_note"]
)