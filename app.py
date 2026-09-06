
import os

# ============================================================
# TENSORFLOW CPU MEMORY / THREAD CONTROL
# ============================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

import io
import gc
import base64

import numpy as np
import tensorflow as tf
import cv2

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

# ============================================================
# TENSORFLOW THREAD LIMIT
# ============================================================

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except RuntimeError:
    pass

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FRONTEND_DIR = os.path.join(
    BASE_DIR,
    "Website_Frontend"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
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

efficientnet = model.get_layer(
    "efficientnetb0"
)

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
# GRAD-CAM
# ============================================================

def generate_gradcam(image_array):

    image_tensor = tf.convert_to_tensor(
        image_array,
        dtype=tf.float32
    )

    image_tensor = tf.expand_dims(
        image_tensor,
        axis=0
    )

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

    weights = tf.reduce_mean(
        gradients,
        axis=(1, 2)
    )

    cam = tf.reduce_sum(
        weights[:, tf.newaxis, tf.newaxis, :]
        * conv_output,
        axis=-1
    )

    cam = tf.maximum(
        cam,
        0
    )

    cam = cam / (
        tf.reduce_max(cam)
        + 1e-8
    )

    probability = float(
        prediction[0, 0].numpy()
    )

    # Convert only the final CAM to NumPy.
    cam_np = cam[0].numpy()

    # Explicitly release TensorFlow temporaries.
    del image_tensor
    del conv_output
    del prediction
    del gradients
    del weights
    del cam

    return cam_np, probability

# ============================================================
# IMAGE → BASE64 PNG
# ============================================================

def image_to_base64(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
        optimize=True
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    buffer.close()

    return encoded

# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return FileResponse(
        os.path.join(
            FRONTEND_DIR,
            "index.html"
        )
    )

# ============================================================
# HEALTH CHECK
# ============================================================

@app.head("/")
def health_check():
    return None

# ============================================================
# STATIC FILES
# ============================================================

@app.get("/style.css")
def style():

    return FileResponse(
        os.path.join(
            FRONTEND_DIR,
            "style.css"
        )
    )


@app.get("/script.js")
def script():

    return FileResponse(
        os.path.join(
            FRONTEND_DIR,
            "script.js"
        )
    )

# ============================================================
# PREDICTION + GRAD-CAM
# ============================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if not file.content_type:
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid fundus image."
        )

    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid fundus image."
        )

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    image_bytes = await file.read()

    # Basic upload size protection.
    if len(image_bytes) > 15 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image file is too large. Please upload an image below 15 MB."
        )

    try:

        original_image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image."
        )

    # Release uploaded bytes as soon as PIL has decoded them.
    del image_bytes

    # --------------------------------------------------------
    # Resize for EfficientNet-B0
    # --------------------------------------------------------

    image_224 = original_image.resize(
        (224, 224),
        Image.Resampling.LANCZOS
    )

    # Model was trained using 0–255 input.
    image_array = np.asarray(
        image_224,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # ONE FORWARD PASS:
    # Prediction + Grad-CAM
    # --------------------------------------------------------

    cam, probability = generate_gradcam(
        image_array
    )

    # Classification
    if probability >= 0.5:
        result = "Referable DR"
    else:
        result = "Non-referable DR"

    # --------------------------------------------------------
    # Resize CAM to original image size
    # --------------------------------------------------------

    original_width, original_height = (
        original_image.size
    )

    cam_resized = cv2.resize(
        cam,
        (original_width, original_height),
        interpolation=cv2.INTER_LINEAR
    )

    cam_uint8 = np.clip(
        cam_resized * 255,
        0,
        255
    ).astype(np.uint8)

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

    original_np = np.asarray(
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
    # Convert to PIL
    # --------------------------------------------------------

    heatmap_image = Image.fromarray(
        heatmap
    )

    overlay_image = Image.fromarray(
        overlay
    )

    # --------------------------------------------------------
    # Base64
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
    # Prepare response
    # --------------------------------------------------------

    response = {

        "filename": file.filename,

        "probability": probability,

        "probability_percent": round(
            probability * 100,
            2
        ),

        "prediction": result,

        "gradcam_probability": probability,

        "gradcam_layer": "top_activation",

        "original_image": original_base64,

        "gradcam_heatmap": heatmap_base64,

        "gradcam_overlay": overlay_base64
    }

    # --------------------------------------------------------
    # Release request-specific arrays/images
    # --------------------------------------------------------

    del image_224
    del image_array
    del cam
    del cam_resized
    del cam_uint8
    del heatmap
    del original_np
    del overlay
    del heatmap_image
    del overlay_image
    del original_image

    gc.collect()

    return response
