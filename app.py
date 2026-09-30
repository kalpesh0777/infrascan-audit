import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import io
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="InfraScan — AI Pothole Detection & Audit",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        padding: 1.8rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 1rem;
        border-left: 5px solid #0d6efd;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .badge-p1 { background-color: #dc3545; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-p2 { background-color: #fd7e14; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-p3 { background-color: #ffc107; color: black; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-p4 { background-color: #198754; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# Load Model with Caching
@st.cache_resource
def load_yolo_model():
    model_path = 'best.pt'
    return YOLO(model_path)

try:
    model = load_yolo_model()
except Exception as e:
    st.error(f"Error loading model weights: {e}")
    st.stop()

# Priority Calculation Helper
def evaluate_priority(area_cm2):
    if area_cm2 >= 1500:
        return "CRITICAL (P1)", "Immediate Asphalt Patching (< 48 hrs)", "#dc3545"
    elif area_cm2 >= 800:
        return "HIGH (P2)", "Scheduled Maintenance (< 14 days)", "#fd7e14"
    elif area_cm2 >= 300:
        return "MEDIUM (P3)", "Routine Maintenance Cycle", "#ffc107"
    else:
        return "LOW (P4)", "Continuous UAV Monitoring", "#198754"

# Header Banner
st.markdown("""
<div class="main-header">
    <h1 style="margin: 0; font-size: 2.2rem;">🛣️ InfraScan AI Road Audit Portal</h1>
    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Automated UAV Pothole Detection, Calibrated Surface Dimensioning & Action Prioritization</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Controls
st.sidebar.header("⚙️ Inspection Settings")
uploaded_file = st.sidebar.file_uploader("Upload Road Inspection Image", type=["jpg", "jpeg", "png"])
road_name = st.sidebar.text_input("Road / Sector Label", value="Sector-4 Roadway A")
cm_per_pixel = st.sidebar.number_input("GSD Calibration (cm/pixel)", value=0.15, step=0.01, format="%.2f",
                                       help="Ground Sampling Distance based on UAV altitude and focal length.")
conf_threshold = st.sidebar.slider("Detection Confidence Threshold", min_value=0.10, max_value=0.90, value=0.25, step=0.05)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Priority Criteria:**
- 🔴 **P1 (Critical):** Area ≥ 1500 cm²
- 🟠 **P2 (High):** Area ≥ 800 cm²
- 🟡 **P3 (Medium):** Area ≥ 300 cm²
- 🟢 **P4 (Low):** Continuous Monitoring
""")

st.sidebar.info("📌 **Hardware Limitation Note:** Measurements represent 2D visible planar surface area. Volumetric depth estimation requires stereoscopic/LiDAR payloads (Future Scope).")

# Main Section
if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file).convert('RGB')
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    annotated_bgr = img_bgr.copy()

    # Inference
    with st.spinner("Processing inspection imagery..."):
        results = model.predict(source=img_bgr, conf=conf_threshold, verbose=False)[0]

    potholes = []
    total_area_cm2 = 0.0
    highest_priority = "LOW (P4)"
    priority_order = {"CRITICAL (P1)": 4, "HIGH (P2)": 3, "MEDIUM (P3)": 2, "LOW (P4)": 1}
    max_rank = 0

    for idx, box in enumerate(results.boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])

        w_cm = (x2 - x1) * cm_per_pixel
        l_cm = (y2 - y1) * cm_per_pixel
        area_cm2 = (np.pi / 4.0) * w_cm * l_cm
        total_area_cm2 += area_cm2

        priority, action, color_hex = evaluate_priority(area_cm2)
        if priority_order[priority] > max_rank:
            max_rank = priority_order[priority]
            highest_priority = priority

        potholes.append({
            "Defect ID": f"POT-{idx+1:03d}",
            "Confidence (%)": round(conf * 100, 1),
            "Width (cm)": round(w_cm, 1),
            "Length (cm)": round(l_cm, 1),
            "Area (cm²)": round(area_cm2, 1),
            "Priority": priority,
            "Recommended Action": action
        })

        # Draw overlays
        cv2.rectangle(annotated_bgr, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 50), 2)
        label = f"#{idx+1}: {area_cm2:.0f}cm2 [{priority.split()[0]}]"
        cv2.putText(annotated_bgr, label, (int(x1), max(22, int(y1) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 50), 2)

    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    # Top KPI Metrics Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric(label="Road Section", value=road_name)
    with kpi2:
        st.metric(label="Defects Detected", value=len(potholes))
    with kpi3:
        st.metric(label="Highest Severity", value=highest_priority if potholes else "None")
    with kpi4:
        st.metric(label="Total Damaged Area", value=f"{total_area_cm2:.1f} cm²", delta=f"{total_area_cm2/10000:.3f} m²")

    st.markdown("---")

    # Side-by-Side Images
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.subheader("📷 Original Road Survey")
        st.image(image, use_container_width=True)

    with col_img2:
        st.subheader("🎯 Calibrated AI Detection Overlay")
        st.image(annotated_rgb, use_container_width=True)

    st.markdown("---")

    # Telemetry Table & Export
    st.subheader("📋 Defect Assessment & Maintenance Schedule")
    if potholes:
        df = pd.DataFrame(potholes)
        st.dataframe(df, use_container_width=True)

        # CSV Download Button
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.download_button(
            label="📥 Download Structured CSV Audit Report",
            data=csv_buffer.getvalue(),
            file_name=f"infrascan_audit_{timestamp}.csv",
            mime="text/csv",
            type="primary"
        )
    else:
        st.success("✅ No road defects detected within the given confidence threshold. Surface condition is optimal.")

else:
    st.info("👆 Please upload a road survey image using the sidebar on the left to begin automated inspection.")
