import os
import json
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


IMG_SIZE = (224, 224)
MODEL_PATH = os.getenv("FRACTURE_MODEL_PATH", "fracture_model.keras")
MODEL_THRESHOLD = float(os.getenv("FRACTURE_THRESHOLD", "0.50"))
METADATA_PATH = os.getenv("FRACTURE_METADATA_PATH", "training_metadata.json")
INJURY_LOCATIONS = ["Arm", "Leg", "Spine", "Rib", "Hip"]
FEATURE_NAMES = [
    "age_norm",
    "gender_male",
    "diabetes",
    "osteoporosis",
    "smoking",
    "previous_fracture",
    "pain_norm",
    "injury_arm",
    "injury_leg",
    "injury_spine",
    "injury_rib",
    "injury_hip",
    "treatment_cast",
    "treatment_surgery",
    "treatment_physiotherapy",
    "treatment_medication",
    "treatment_none",
]


st.set_page_config(
    page_title="X-Ray Fracture Reporting",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_fracture_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


@st.cache_data(show_spinner=False)
def load_training_metadata():
    if not os.path.exists(METADATA_PATH):
        return {}
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return {}


def encode_patient_features(
    age,
    gender,
    diabetes,
    osteoporosis,
    smoking,
    previous_fracture,
    pain_level,
    injury_location,
    treatment_history,
):
    text = (treatment_history or "").lower()
    location = (injury_location or "").lower()
    age_norm = np.clip(float(age), 0.0, 110.0) / 110.0
    pain_norm = np.clip(float(pain_level), 1.0, 10.0) / 10.0
    values = [
        age_norm,
        1.0 if gender == "Male" else 0.0,
        float(bool(diabetes)),
        float(bool(osteoporosis)),
        float(bool(smoking)),
        float(bool(previous_fracture)),
        pain_norm,
        1.0 if location == "arm" else 0.0,
        1.0 if location == "leg" else 0.0,
        1.0 if location == "spine" else 0.0,
        1.0 if location == "rib" else 0.0,
        1.0 if location == "hip" else 0.0,
        1.0 if "cast" in text else 0.0,
        1.0 if "surgery" in text or "operation" in text else 0.0,
        1.0 if "physio" in text or "rehab" in text else 0.0,
        1.0 if "medication" in text or "analgesic" in text or "painkiller" in text else 0.0,
        1.0 if not text.strip() or "none" in text or "no treatment" in text else 0.0,
    ]
    return np.asarray(values, dtype=np.float32).reshape(1, -1)


def preprocess_xray(image):
    image = image.convert("RGB").resize(IMG_SIZE)
    return np.asarray(image, dtype=np.float32).reshape(1, IMG_SIZE[0], IMG_SIZE[1], 3)


def predict_fracture(model, image_batch, tabular_batch):
    try:
        pred = model(
            {"xray_image": image_batch, "patient_features": tabular_batch},
            training=False,
        )
    except Exception:
        pred = model([image_batch, tabular_batch], training=False)
    return float(np.asarray(pred).reshape(-1)[0])


def find_gradcam_layer(model):
    for name in ("last_conv_features", "Conv_1", "out_relu"):
        try:
            model.get_layer(name)
            return name
        except Exception:
            pass

    for layer in reversed(model.layers):
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is not None and len(output_shape) == 4:
            return layer.name
        try:
            if len(layer.output.shape) == 4:
                return layer.name
        except Exception:
            continue
    return None


def make_gradcam_overlay(model, pil_image, image_batch, tabular_batch):
    layer_name = find_gradcam_layer(model)
    if layer_name is None:
        return None

    grad_model = tf.keras.models.Model(
        model.inputs,
        [model.get_layer(layer_name).output, model.output],
    )
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model([image_batch, tabular_batch], training=False)
        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        return None

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    max_value = tf.reduce_max(heatmap)
    if float(max_value) > 0:
        heatmap = heatmap / max_value

    heatmap = heatmap.numpy()
    original = np.asarray(pil_image.convert("RGB"))
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(original, 0.62, heatmap, 0.38, 0)
    return overlay


def estimate_risk(probability, age, diabetes, osteoporosis, smoking, previous_fracture, pain, location):
    clinical_points = 0
    clinical_points += int(age >= 65)
    clinical_points += int(diabetes)
    clinical_points += int(osteoporosis)
    clinical_points += int(smoking)
    clinical_points += int(previous_fracture)
    clinical_points += int(pain >= 8)
    clinical_points += int(location in {"Hip", "Spine"})
    clinical_risk = min(clinical_points / 7.0, 1.0)
    risk_score = 100.0 * (0.65 * probability + 0.35 * clinical_risk)

    if risk_score >= 80 or probability >= 0.90:
        level = "Critical"
    elif risk_score >= 60 or probability >= 0.70:
        level = "High"
    elif risk_score >= 35 or probability >= 0.45:
        level = "Moderate"
    else:
        level = "Low"
    return level, risk_score


def build_report(patient_name, probability, risk_level, risk_score, label, age, gender, location, pain):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""Automated X-Ray Fracture & Crack Decision Support Report

Generated: {generated_at}
Patient: {patient_name or "Not provided"}
Age/Gender: {age} / {gender}
Injury Location: {location}
Pain Level: {pain}/10

Model Finding: {label}
Fracture/Crack Probability: {probability:.1%}
Clinical Risk Level: {risk_level}
Risk Score: {risk_score:.1f}/100

Interpretation:
This report combines a CNN-based X-ray probability estimate with structured
patient risk factors. The heatmap highlights image regions that most influenced
the model output.

Clinical Safety Note:
This system is for educational decision support only. It is not a standalone
diagnostic device and must not replace radiologist or physician review.
"""


st.title("Automated Multi-Modal X-Ray Fracture & Crack Reporting System")
metadata = load_training_metadata()
threshold = float(metadata.get("threshold", MODEL_THRESHOLD))

model = load_fracture_model()
if model is None:
    st.error(
        f"`{MODEL_PATH}` was not found. Train the model in the Colab notebook, "
        "download `fracture_model.keras`, and place it beside this app."
    )
    st.stop()

left, right = st.columns([1.05, 0.95])

with left:
    uploaded_file = st.file_uploader("Upload X-Ray image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        xray_image = Image.open(uploaded_file)
        st.image(xray_image, caption="Input X-Ray", use_container_width=True)

with right:
    patient_name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=1, max_value=110, value=45)
    gender = st.selectbox("Gender", ["Male", "Female"])
    location = st.selectbox("Injury Location", INJURY_LOCATIONS)
    pain_level = st.slider("Pain Level", 1, 10, 5)
    diabetes = st.checkbox("Diabetes")
    osteoporosis = st.checkbox("Osteoporosis")
    smoking = st.checkbox("Smoking")
    previous_fracture = st.checkbox("Previous Fracture")
    treatment_history = st.text_area("Treatment History", placeholder="e.g., cast, surgery, medication")

if uploaded_file and st.button("Generate Report", type="primary"):
    image_batch = preprocess_xray(xray_image)
    tabular_batch = encode_patient_features(
        age,
        gender,
        diabetes,
        osteoporosis,
        smoking,
        previous_fracture,
        pain_level,
        location,
        treatment_history,
    )
    probability = predict_fracture(model, image_batch, tabular_batch)
    label = "Fracture/Crack suspected" if probability >= threshold else "No fracture pattern detected"
    risk_level, risk_score = estimate_risk(
        probability,
        age,
        diabetes,
        osteoporosis,
        smoking,
        previous_fracture,
        pain_level,
        location,
    )

    result_col, heatmap_col = st.columns(2)
    with result_col:
        st.metric("Fracture/Crack Probability", f"{probability:.1%}")
        st.metric("Clinical Risk Level", risk_level, f"{risk_score:.1f}/100")
        st.write(label)

    with heatmap_col:
        overlay = make_gradcam_overlay(model, xray_image, image_batch, tabular_batch)
        if overlay is not None:
            st.image(overlay, caption="Grad-CAM Suspicious Region Heatmap", use_container_width=True)
        else:
            st.warning("Heatmap could not be generated for this model architecture.")

    report = build_report(
        patient_name,
        probability,
        risk_level,
        risk_score,
        label,
        age,
        gender,
        location,
        pain_level,
    )
    st.text_area("Generated Clinical Support Report", report, height=330)
    st.download_button(
        "Download Report",
        data=report,
        file_name="fracture_support_report.txt",
        mime="text/plain",
    )
