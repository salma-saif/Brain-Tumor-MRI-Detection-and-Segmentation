import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os
import uuid
import cv2

# ---------------- CONFIG ----------------
st.set_page_config(page_title=" Brain Tumor Detector MRI", layout="centered")

# ---------------- PATHS ----------------
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- INPUT SHAPES ----------------
RESNET_INPUT_SIZE = (224, 224)
UNET_INPUT_SIZE = (128, 128)

# ---------------- LOAD MODELS ----------------
resnet_model = tf.saved_model.load('models/resnet_model_converted')
resnet_infer = resnet_model.signatures["serving_default"]

unet_model = tf.saved_model.load('models/seg_unet_model_converted')
unet_infer = unet_model.signatures["serving_default"]

# ---------------- FUNCTIONS ----------------
def preprocess_image(image: Image.Image, target_size):
    img = image.convert("RGB").resize(target_size)
    img = np.array(img) / 255.0
    return np.expand_dims(img.astype(np.float32), axis=0)

def classify_image(image, infer_fn):
    img_tensor = preprocess_image(image, RESNET_INPUT_SIZE)
    output_dict = infer_fn(tf.constant(img_tensor))
    output_key = list(output_dict.keys())[0]
    preds = output_dict[output_key].numpy()[0]
    label = "Tumor Detected" if np.argmax(preds) == 1 else "No Tumor Detected"
    confidence = round(float(np.max(preds)) * 100, 2)
    return label, confidence

def segment_tumor(image, infer_fn):
    img_tensor = preprocess_image(image, UNET_INPUT_SIZE)
    output_dict = infer_fn(tf.constant(img_tensor))
    output_key = list(output_dict.keys())[0]
    prediction = output_dict[output_key].numpy()[0]
    mask = (prediction > 0.5).astype(np.uint8) * 255  # shape: (128, 128, 1)

    # Resize to original
    original = np.array(image.convert("RGB"))
    resized_mask = cv2.resize(mask[:, :, 0], (original.shape[1], original.shape[0]))

    # Save binary mask
    mask_filename = f"{uuid.uuid4().hex[:8]}_mask.png"
    mask_path = os.path.join(OUTPUT_FOLDER, mask_filename)
    cv2.imwrite(mask_path, resized_mask)

    # Overlay
    color_mask = cv2.applyColorMap(resized_mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, color_mask, 0.4, 0)

    # Save overlay
    overlay_filename = f"{uuid.uuid4().hex[:8]}_overlay.png"
    overlay_path = os.path.join(OUTPUT_FOLDER, overlay_filename)
    cv2.imwrite(overlay_path, overlay)

    return overlay_path, mask_path

# ---------------- UI ----------------
st.title(" Brain Tumor Classification & Segmentation MRI")

uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png", "tif"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Run Prediction"):
        # CLASSIFICATION
        label, confidence = classify_image(image, resnet_infer)
        st.markdown(f"### 🔍 Prediction: **{label}** ({confidence}%)")

        # SEGMENTATION
        overlay_path, mask_path = segment_tumor(image, unet_infer)

        st.image(mask_path, caption="🩺 Segmentation Mask", use_column_width=True)
        st.image(overlay_path, caption="🎯 Overlay", use_column_width=True)
