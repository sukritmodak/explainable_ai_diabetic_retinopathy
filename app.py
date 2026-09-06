
import os
import io
import base64
import numpy as np
import tensorflow as tf
import cv2

from PIL import Image
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi import FastAPI, UploadFile, File, HTTPException

# ============================================================
# PATHS
# ============================================================

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "Website_Frontend")

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "IDRiD_EfficientNetB0_DR.keras"
)

# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AI Diabetic Retinopathy Screening",
    description="EfficientNet-B0 + Grad-CAM",
    version="1.0"
)

# ============================================================
# LOAD MODEL
# ============================================================

print("Loading trained model...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Model loaded successfully.")
print("Model parameters:", model.count_params())

# ============================================================
# GRAD-CAM MODEL
# ============================================================

efficientnet = model.get_layer("efficientnetb0")

target_layer = efficientnet.get_layer(
    "top_activation"
)

gap_layer = model.layers[2]
dropout_layer = model.layers[3]
dense_layer = model.layers[4]

grad_model = tf.keras.models.Model(
    inputs=efficientnet.input,
    outputs=[
        target_layer.output,
        dense_layer(
            dropout_layer(
                gap_layer(target_layer.output),
                training=False
            ),
            training=False
        )
    ]
)

print("Grad-CAM model ready.")
print("Target layer:", target_layer.name)
print("Target shape:", target_layer.output.shape)

# ============================================================
# GRAD-CAM FUNCTION
# ============================================================

def generate_gradcam(image_array):

    # Convert to tensor
    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    image_tensor = tf.expand_dims(
        image_tensor,
        axis=0
    )

    # --------------------------------------------------------
    # Forward pass + gradients
    # --------------------------------------------------------

    with tf.GradientTape() as tape:

        conv_output, prediction = grad_model(
            image_tensor,
            training=False
        )

        loss = prediction[:, 0]

    gradients = tape.gradient(
        loss,
        conv_output
    )

    # --------------------------------------------------------
    # Global-average-pool gradients
    # --------------------------------------------------------

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    # --------------------------------------------------------
    # Weighted feature maps
    # --------------------------------------------------------

    cam = tf.reduce_sum(
        weights[:, tf.newaxis, tf.newaxis, :]
        * conv_output,
        axis=-1
    )

    # --------------------------------------------------------
    # ReLU
    # --------------------------------------------------------

    cam = tf.maximum(
        cam,
        0
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    cam = cam / (
        tf.reduce_max(cam)
        + 1e-8
    )

    return (
        cam[0].numpy(),
        float(prediction[0][0])
    )

# ============================================================
# IMAGE → BASE64 PNG
# ============================================================

def image_to_base64(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    return base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():
    return FileResponse(
        os.path.join(FRONTEND_DIR, "index.html")
    )

# ============================================================
# PREDICTION + GRAD-CAM
# ============================================================


@app.get("/style.css")
def style():
    return FileResponse(
        os.path.join(FRONTEND_DIR, "style.css")
    )

@app.get("/script.js")
def script():
    return FileResponse(
        os.path.join(FRONTEND_DIR, "script.js")
    )

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if not file.content_type.startswith("image/"):

        raise HTTPException(
            status_code=400,
            detail="Please upload a valid fundus image."
        )

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    image_bytes = await file.read()

    try:

        original_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image."
        )

    # --------------------------------------------------------
    # Resize for EfficientNet-B0
    # --------------------------------------------------------

    image_224 = original_image.resize(
        (224, 224),
        Image.Resampling.LANCZOS
    )

    # IMPORTANT:
    # Model was trained using 0–255 input
    image_array = np.array(
        image_224
    ).astype(np.float32)

    image_input = np.expand_dims(
        image_array,
        axis=0
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        image_input,
        verbose=0
    )

    probability = float(
        prediction[0][0]
    )

    # --------------------------------------------------------
    # Classification
    # --------------------------------------------------------

    if probability >= 0.5:

        result = "Referable DR"

    else:

        result = "Non-referable DR"

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    cam, gradcam_probability = generate_gradcam(
        image_array
    )

    # --------------------------------------------------------
    # Resize CAM to original image size
    # --------------------------------------------------------

    original_width, original_height = (
        original_image.size
    )

    cam_resized = cv2.resize(
        cam,
        (original_width, original_height)
    )

    cam_uint8 = np.uint8(
        cam_resized * 255
    )

    # --------------------------------------------------------
    # Create heatmap
    # --------------------------------------------------------

    heatmap = cv2.applyColorMap(
        cam_uint8,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------------
    # Create overlay
    # --------------------------------------------------------

    original_np = np.array(
        original_image
    )

    overlay = cv2.addWeighted(
        original_np,
        0.60,
        heatmap,
        0.40,
        0
    )

    # --------------------------------------------------------
    # Convert images to PIL
    # --------------------------------------------------------

    heatmap_image = Image.fromarray(
        heatmap
    )

    overlay_image = Image.fromarray(
        overlay
    )

    # --------------------------------------------------------
    # Base64 images for website
    # --------------------------------------------------------

    original_base64 = image_to_base64(
        original_image
    )

    heatmap_base64 = image_to_base64(
        heatmap_image
    )

    overlay_base64 = image_to_base64(
        overlay_image
    )

    # --------------------------------------------------------
    # Return result
    # --------------------------------------------------------

    return {

        "filename": file.filename,

        "probability": probability,

        "probability_percent": round(
            probability * 100,
            2
        ),

        "prediction": result,

        "gradcam_probability": gradcam_probability,

        "gradcam_layer": "top_activation",

        "original_image": original_base64,

        "gradcam_heatmap": heatmap_base64,

        "gradcam_overlay": overlay_base64
    }
