"""
==================================================================================
AI Chest X-ray Disease Detection - Streamlit Web Application
==================================================================================
Author  : Mahbubul Islam
Purpose : Upload a chest X-ray image and classify the disease using models
          the user uploads themselves — no models are bundled or predefined.

Run with:
    streamlit run app.py
==================================================================================
"""

import os
import re
import json
import time
import uuid
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# TensorFlow is imported lazily-safe: if it's missing, the app still loads
# and shows a friendly error instead of crashing on import.
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except Exception as e:  # pragma: no cover
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = e


# ==================================================================================
# CONFIGURATION
# ==================================================================================

APP_TITLE = "AI Chest X-ray Disease Detection"
APP_SUBTITLE = (
    "Upload a chest X-ray image and let an AI model assist in identifying "
    "potential thoracic diseases. This tool is for research/educational "
    "purposes and is NOT a substitute for professional medical diagnosis."
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder app.py lives in, regardless of CWD

MODELS_DIR = os.path.join(BASE_DIR, "models")
UPLOADED_MODELS_DIR = os.path.join(MODELS_DIR, "uploaded")  # user-uploaded models are saved here
REGISTRY_FILE = os.path.join(UPLOADED_MODELS_DIR, "registry.json")  # persists uploaded-model metadata
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
CLASS_NAMES_FILE = os.path.join(BASE_DIR, "class_names.txt")
IMAGE_SIZE = (224, 224)

MODEL_OPTIONS = {}
# No models are predefined — every model available in the app comes from the
# user uploading it via the sidebar. This dict intentionally stays empty; it
# only exists so the rest of the code (which merges "built-in" + "uploaded"
# models into one registry) keeps working unchanged if built-ins are ever
# reintroduced later.

# Prediction modes offered in the sidebar
MODE_SINGLE = "Single Model"
MODE_ENSEMBLE = "Ensemble (Multiple Models)"

# Key used to label the combined ensemble result in cards/charts
ENSEMBLE_RESULT_LABEL = "Ensemble (Average)"
ENSEMBLE_RESULT_COLOR = "#c2410c"  # warm deep rust — stands out from the cool emerald palette

# Internal labels only (not shown in the UI) describing each preprocessing method
PREPROCESSING_LABELS = {
    "rescale": "Simple [0, 1] rescale (divide by 255)",
    "resnet50": "ResNet50 preprocess_input (Caffe-style, ImageNet mean)",
}


# ==================================================================================
# PAGE CONFIGURATION
# ==================================================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================================================
# CUSTOM CSS - Medical dashboard theme (blue & white, rounded cards, soft shadows)
# ==================================================================================

def inject_custom_css():
    """Inject custom CSS for a modern medical-themed, responsive UI."""
    st.markdown(
        """
        <style>
            /* ---------- Palette ----------
               White base with a single deep accent color (emerald/teal),
               used consistently across header, buttons, cards, and sidebar. */
            :root {
                --primary: #0f6e5c;        /* deep emerald - main accent */
                --primary-dark: #0a4a3d;   /* darker shade for gradients/hover */
                --primary-light: #e6f4f0;  /* very light tint for backgrounds */
                --text-dark: #1f2937;
                --text-muted: #6b7280;
                --border: #e3e8e6;
            }

            /* ---------- Global: force the ACTUAL page containers to white ----------
               Overriding only ".main" is not enough in current Streamlit —
               the real scrollable page background lives on .stApp /
               stAppViewContainer, and the top toolbar has its own layer too. */
            html, body, [class*="css"] {
                font-family: 'Segoe UI', 'Helvetica Neue', Arial, sans-serif;
            }

            .stApp,
            [data-testid="stAppViewContainer"],
            [data-testid="stHeader"],
            [data-testid="stBottomBlockContainer"],
            [data-testid="stMain"],
            .main {
                background-color: #ffffff !important;
            }

            [data-testid="stHeader"] {
                background-color: rgba(255, 255, 255, 0) !important;
            }

            /* Default body text dark on the main page (header/result-card
               below override this locally with higher-specificity class rules) */
            [data-testid="stAppViewContainer"] p,
            [data-testid="stAppViewContainer"] span,
            [data-testid="stAppViewContainer"] label,
            [data-testid="stAppViewContainer"] li {
                color: var(--text-dark);
            }

            /* ---------- Header ---------- */
            .app-header, .app-header * {
                color: #ffffff !important;
            }
            .app-header {
                background: linear-gradient(90deg, var(--primary-dark) 0%, var(--primary) 100%);
                padding: 2rem 2.5rem;
                border-radius: 18px;
                box-shadow: 0 8px 24px rgba(15, 110, 92, 0.25);
                margin-bottom: 1.5rem;
            }
            .app-header h1 {
                margin: 0;
                font-size: 2.3rem;
                font-weight: 700;
            }
            .app-header p {
                margin-top: 0.5rem;
                font-size: 1.05rem;
                opacity: 0.95;
            }

            /* ---------- Bordered containers used as "cards" ----------
               st.container(border=True) renders with this test id; we
               restyle its default border into our rounded, shadowed card. */
            div[data-testid="stVerticalBlockBorderWrapper"] {
                background-color: #ffffff !important;
                border-radius: 16px !important;
                border: 1px solid var(--border) !important;
                box-shadow: 0 4px 18px rgba(15, 110, 92, 0.08);
                padding: 0.5rem 0.5rem;
            }

            .result-card {
                background: linear-gradient(135deg, #ffffff 0%, var(--primary-light) 100%);
                border-radius: 18px;
                padding: 1.8rem 2rem;
                box-shadow: 0 8px 22px rgba(15, 110, 92, 0.14);
                border: 1px solid #cfe8e0;
            }

            .result-row {
                display: flex;
                justify-content: space-between;
                padding: 0.55rem 0;
                border-bottom: 1px solid var(--border);
                font-size: 1.05rem;
            }
            .result-row:last-child { border-bottom: none; }
            .result-label { font-weight: 600; color: var(--primary); }
            .result-value { font-weight: 700; color: var(--text-dark); }

            /* ---------- Badges ---------- */
            .badge {
                display: inline-block;
                padding: 0.3rem 0.9rem;
                border-radius: 999px;
                background: var(--primary);
                color: white;
                font-weight: 600;
                font-size: 0.85rem;
            }

            /* ---------- Sidebar ---------- */
            section[data-testid="stSidebar"] {
                background-color: #ffffff !important;
                border-right: 1px solid var(--border);
            }
            section[data-testid="stSidebar"] * {
                color: var(--text-dark) !important;
            }
            section[data-testid="stSidebar"] h2,
            section[data-testid="stSidebar"] h3 {
                color: var(--primary) !important;
            }

            /* ---------- Select / dropdown (closed box + open popup) ----------
               The popup menu renders in its own overlay outside the sidebar's
               DOM, so it needs its own non-scoped rules to guarantee it never
               inherits a dark theme regardless of OS/browser dark mode. */
            div[data-baseweb="select"] > div {
                background-color: #ffffff !important;
                border: 1px solid var(--primary) !important;
                color: var(--text-dark) !important;
            }
            div[data-baseweb="select"] span,
            div[data-baseweb="select"] div {
                color: var(--text-dark) !important;
            }
            div[data-baseweb="popover"],
            ul[data-baseweb="menu"] {
                background-color: #ffffff !important;
            }
            li[data-baseweb="menu-item"] {
                background-color: #ffffff !important;
                color: var(--text-dark) !important;
            }
            li[data-baseweb="menu-item"]:hover,
            li[data-baseweb="menu-item"][aria-selected="true"] {
                background-color: var(--primary-light) !important;
                color: var(--text-dark) !important;
            }

            /* ---------- File uploader ---------- */
            [data-testid="stFileUploaderDropzone"] {
                background-color: var(--primary-light) !important;
                border: 1.5px dashed var(--primary) !important;
                border-radius: 12px !important;
            }
            [data-testid="stFileUploaderDropzone"] * {
                color: var(--text-dark) !important;
            }
            [data-testid="stFileUploaderDropzoneInstructions"] svg {
                fill: var(--primary) !important;
            }
            [data-testid="stBaseButton-secondary"] {
                background-color: #ffffff !important;
                color: var(--primary) !important;
                border: 1px solid var(--primary) !important;
            }

            /* ---------- Buttons ---------- */
            .stButton>button {
                background: linear-gradient(90deg, var(--primary), var(--primary-dark));
                color: #ffffff !important;
                border: none;
                border-radius: 12px;
                padding: 0.7rem 1.6rem;
                font-weight: 700;
                font-size: 1.05rem;
                box-shadow: 0 4px 14px rgba(15, 110, 92, 0.30);
                transition: transform 0.15s ease-in-out;
            }
            .stButton>button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 18px rgba(15, 110, 92, 0.40);
            }
            .stButton>button * {
                color: #ffffff !important;
            }

            /* ---------- Progress bar ---------- */
            [data-testid="stProgress"] > div > div > div {
                background-color: var(--primary) !important;
            }
            [data-testid="stProgress"] > div > div {
                background-color: var(--primary-light) !important;
            }

            /* ---------- Alerts (info / warning / success / error) ---------- */
            [data-testid="stAlert"] {
                border-radius: 12px !important;
            }

            /* ---------- Expander ---------- */
            [data-testid="stExpander"] {
                background-color: #ffffff !important;
                border: 1px solid var(--border) !important;
                border-radius: 12px !important;
            }
            [data-testid="stExpander"] summary {
                color: var(--primary-dark) !important;
            }

            /* ---------- Dataframe / table ---------- */
            [data-testid="stDataFrame"] {
                border: 1px solid var(--border) !important;
                border-radius: 10px !important;
            }

            /* ---------- Footer ---------- */
            .footer {
                text-align: center;
                padding: 1.2rem 0 0.5rem 0;
                color: var(--text-muted);
                font-size: 0.9rem;
                border-top: 1px solid var(--border);
                margin-top: 2rem;
            }

            /* ---------- Section titles ---------- */
            .section-title {
                font-size: 1.3rem;
                font-weight: 700;
                color: var(--primary-dark) !important;
                margin-bottom: 0.6rem;
            }
            .section-title * {
                color: var(--primary-dark) !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================================
# UTILITY / HELPER FUNCTIONS
# ==================================================================================

@st.cache_resource(show_spinner=False)
def load_class_names(filepath: str, mtime: float = 0.0):
    """
    Load class names from a text file (one class per line).
    Falls back to a default placeholder list if the file is missing.

    mtime is included purely as a cache-busting key: if the file is edited
    after the app already cached a result, passing the new modified-time
    forces a fresh read instead of silently returning the stale cached list.
    IMPORTANT: this parameter must NOT start with an underscore — Streamlit
    deliberately excludes underscore-prefixed parameters from the cache key,
    which would silently defeat this cache-busting entirely.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        if not classes:
            raise ValueError("class_names.txt is empty.")
        return classes
    except Exception:
        # Fallback so the app still runs even without the file present
        return ["Normal", "Pneumonia", "COVID-19", "Tuberculosis", "Lung Opacity"]


@st.cache_resource(show_spinner=False)
def load_model(model_path: str, mtime: float = 0.0):
    """
    Load and cache a Keras model from disk.
    Cached with @st.cache_resource so the model is loaded only once per
    (path, mtime) combination. Including the file's last-modified time in
    the cache key means that if a user deletes and re-uploads a model under
    the same path, the cache correctly invalidates instead of serving a
    stale model object.

    Args:
        model_path: path to the .keras / .h5 model file
        mtime: last-modified timestamp of the file, used only as a cache
            key. IMPORTANT: this must NOT start with an underscore —
            Streamlit deliberately excludes underscore-prefixed parameters
            from the cache key, which would silently defeat this entirely.

    Returns:
        model: The loaded TF/Keras model, or None if loading fails.
        error_message: str, empty if successful.
    """
    if not TF_AVAILABLE:
        return None, f"TensorFlow is not available: {TF_IMPORT_ERROR}"

    if not os.path.exists(model_path):
        return None, f"Model file not found at '{model_path}'."

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model, ""
    except Exception as e:
        return None, f"Failed to load model: {e}"


def _sanitize_filename(name: str) -> str:
    """Turn a user-supplied model name into a safe filename fragment."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    return safe or "model"


def init_custom_models_state():
    """Ensure the uploaded-models directory exists on disk."""
    os.makedirs(UPLOADED_MODELS_DIR, exist_ok=True)


def load_custom_models_registry():
    """
    Load the persisted registry of user-uploaded models from disk.
    This is what makes uploads permanent — the mapping of name -> model file
    is stored in a JSON file next to the uploaded models, not just in memory,
    so it survives page reloads and app restarts.

    Returns:
        dict: name -> {"path": str, "icon": str, "default_preprocessing": str}
    """
    if not os.path.exists(REGISTRY_FILE):
        return {}
    try:
        with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Drop any entries whose model file has gone missing on disk
        return {name: info for name, info in data.items() if os.path.exists(info.get("path", ""))}
    except Exception:
        return {}


def save_custom_models_registry(registry: dict):
    """Persist the uploaded-models registry to disk as JSON."""
    os.makedirs(UPLOADED_MODELS_DIR, exist_ok=True)
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)


def get_model_registry():
    """
    Merge the built-in models with every user-uploaded model persisted on disk.

    Returns:
        dict: name -> {"path": str, "icon": str, "default_preprocessing": str, "builtin": bool}
    """
    registry = {}
    for name, info in MODEL_OPTIONS.items():
        registry[name] = {**info, "builtin": True}
    for name, info in load_custom_models_registry().items():
        registry[name] = {**info, "builtin": False}
    return registry


def add_custom_model(name: str, uploaded_file, preprocessing_method: str):
    """
    Save an uploaded model file to disk and permanently register it under
    the given name (written to REGISTRY_FILE so it survives restarts).

    Args:
        name: user-chosen display name for the model (must be unique)
        uploaded_file: a Streamlit UploadedFile object (.keras or .h5)
        preprocessing_method: "rescale" or "resnet50"

    Returns:
        (success: bool, message: str)
    """
    name = name.strip()
    if not name:
        return False, "Please enter a name for the model."

    if name in MODEL_OPTIONS:
        return False, f"'{name}' is a reserved built-in model name. Please choose another name."

    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext not in (".keras", ".h5"):
        return False, "Unsupported file type. Please upload a .keras or .h5 model file."

    custom_registry = load_custom_models_registry()

    # If replacing a previously uploaded model with the same name, remove its old file first
    if name in custom_registry:
        old_path = custom_registry[name].get("path")
        if old_path and os.path.exists(old_path):
            try:
                os.remove(old_path)
            except OSError:
                pass  # non-fatal — the new file will still be saved under a fresh name

    safe_name = _sanitize_filename(name)
    unique_suffix = uuid.uuid4().hex[:8]
    save_path = os.path.join(UPLOADED_MODELS_DIR, f"{safe_name}_{unique_suffix}{ext}")

    try:
        os.makedirs(UPLOADED_MODELS_DIR, exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    except Exception as e:
        return False, f"Failed to save uploaded model: {e}"

    custom_registry[name] = {
        "path": save_path,
        "icon": "🧩",
        "default_preprocessing": preprocessing_method,
    }
    save_custom_models_registry(custom_registry)
    return True, f"'{name}' was added successfully."


def remove_custom_model(name: str):
    """Delete an uploaded model's file from disk and remove it from the persisted registry."""
    custom_registry = load_custom_models_registry()
    info = custom_registry.pop(name, None)
    if info and os.path.exists(info["path"]):
        try:
            os.remove(info["path"])
        except OSError:
            pass
    save_custom_models_registry(custom_registry)


def preprocess_image(image: Image.Image, target_size=IMAGE_SIZE, method: str = "rescale") -> np.ndarray:
    """
    Preprocess a PIL image for model inference.

    IMPORTANT: preprocessing MUST exactly match what was used during training,
    or the model will effectively see out-of-distribution input and can
    collapse to predicting one class regardless of the image. That is the
    most common cause of a model that "always predicts Normal".

    Steps:
      1. Convert to RGB (handles grayscale / RGBA X-rays)
      2. Resize to target_size (default 224x224)
      3. Apply the chosen normalization method
      4. Expand dimensions to create a batch of size 1

    Args:
        image: PIL.Image object (raw uploaded image)
        target_size: tuple(width, height) expected by the model
        method: "rescale"  -> divide by 255, giving pixel values in [0, 1]
                             (matches ImageDataGenerator(rescale=1./255))
                "resnet50" -> tf.keras.applications.resnet50.preprocess_input
                             (BGR channel order + ImageNet mean subtraction;
                             matches models fine-tuned with that helper)

    Returns:
        np.ndarray of shape (1, H, W, 3), dtype float32
    """
    image = image.convert("RGB")                     # Ensure 3-channel RGB
    image = image.resize(target_size)                 # Resize to model input size
    img_array = np.asarray(image).astype("float32")   # Convert to numpy array

    if method == "resnet50" and TF_AVAILABLE:
        # Applies the exact same transform Keras' ResNet50 was pretrained with:
        # converts RGB->BGR and subtracts per-channel ImageNet means.
        img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    else:
        # Default / fallback: simple min-max normalization to [0, 1]
        img_array = img_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)      # Add batch dimension
    return img_array


def predict_image(image: Image.Image, model, class_names, preprocessing_method: str = "rescale"):
    """
    Run inference on a single chest X-ray image.

    Args:
        image: PIL.Image, the raw uploaded X-ray image
        model: a loaded tf.keras.Model
        class_names: list[str], ordered class labels matching model output
        preprocessing_method: "rescale" or "resnet50" - MUST match training

    Returns:
        predicted_class (str): the class with the highest probability
        confidence (float): confidence percentage (0-100) of the top class
        probability_vector (np.ndarray): full probability distribution
        inference_time (float): time taken for prediction in milliseconds
        class_names (list[str]): the class names actually used (see mismatch_warning)
        mismatch_warning (str or None): set if class_names.txt's length didn't
            match this model's number of outputs, so the caller can surface
            it instead of silently showing "Class 0", "Class 1", etc.
    """
    # Preprocess the image into the format the model expects
    processed = preprocess_image(image, IMAGE_SIZE, method=preprocessing_method)

    # Time the raw model inference call
    start_time = time.perf_counter()
    predictions = model.predict(processed, verbose=0)
    end_time = time.perf_counter()

    inference_time = (end_time - start_time) * 1000.0  # convert to milliseconds

    probability_vector = predictions[0]

    # Guard against mismatched class_names length vs model output size.
    # Rather than silently falling back to "Class 0", "Class 1", ... this
    # returns a clear, specific warning explaining exactly what mismatched.
    num_classes = len(probability_vector)
    mismatch_warning = None
    if len(class_names) != num_classes:
        mismatch_warning = (
            f"class_names.txt has {len(class_names)} class name(s), but this "
            f"model outputs {num_classes} class(es). Showing generic labels "
            f"('Class 0', 'Class 1', ...) until class_names.txt has exactly "
            f"{num_classes} line(s) in the order this model was trained on."
        )
        class_names = [f"Class {i}" for i in range(num_classes)]

    predicted_index = int(np.argmax(probability_vector))
    predicted_class = class_names[predicted_index]
    confidence = float(probability_vector[predicted_index]) * 100.0

    return predicted_class, confidence, probability_vector, inference_time, class_names, mismatch_warning


def plot_probabilities(probability_vector, class_names, predicted_class):
    """
    Create a horizontal bar chart of class probabilities using matplotlib.
    The predicted class bar is highlighted in a distinct color.

    Returns:
        matplotlib Figure object
    """
    percentages = probability_vector * 100.0

    # Sort by probability for a cleaner visual (highest at top)
    order = np.argsort(percentages)
    sorted_classes = [class_names[i] for i in order]
    sorted_percentages = percentages[order]

    colors = [
        "#0f6e5c" if cls == predicted_class else "#bfe3da"
        for cls in sorted_classes
    ]

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(class_names) + 1.5))
    bars = ax.barh(sorted_classes, sorted_percentages, color=colors, height=0.55)

    # Add percentage labels at the end of each bar
    for bar, pct in zip(bars, sorted_percentages):
        ax.text(
            bar.get_width() + 1.0,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#1f2937",
        )

    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", fontsize=10, color="#1f2937")
    ax.set_title("Class Probability Distribution", fontsize=12, fontweight="bold", color="#0a4a3d")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=10, colors="#1f2937")
    ax.tick_params(axis="x", colors="#1f2937")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    fig.tight_layout()

    return fig


def average_probability_vectors(probability_vectors):
    """
    Combine multiple models' probability vectors into a single ensemble
    prediction by simple averaging (equal-weight soft voting).

    Args:
        probability_vectors: list of np.ndarray, all the same length

    Returns:
        np.ndarray: the mean probability vector
    """
    stacked = np.stack(probability_vectors, axis=0)
    return np.mean(stacked, axis=0)


def plot_model_comparison(results_by_model, class_names, highlight_key=None, highlight_color=None):
    """
    Create a grouped horizontal bar chart comparing probability distributions
    across multiple models (used in Ensemble / multi-model comparison mode).

    Args:
        results_by_model: dict mapping model_name -> probability_vector (np.ndarray)
        class_names: list[str], class labels shared by all models
        highlight_key: optional model_name to render in a distinct color
                       (used to make the combined ensemble result stand out)
        highlight_color: hex color to use for highlight_key's bars

    Returns:
        matplotlib Figure object
    """
    model_names = list(results_by_model.keys())
    num_models = len(model_names)
    num_classes = len(class_names)

    y = np.arange(num_classes)
    bar_height = 0.8 / max(num_models, 1)

    # Deep, muted tones that read well on white — cycles if there are many models
    palette = [
        "#0f6e5c", "#3f4a91", "#8a5a2b", "#a13f5c",
        "#2f6690", "#6b4e8e", "#4a7c59", "#8c5383",
    ]

    fig, ax = plt.subplots(figsize=(7.5, 0.6 * num_classes + 1.5))

    for i, model_name in enumerate(model_names):
        percentages = results_by_model[model_name] * 100.0
        offset = (i - (num_models - 1) / 2) * bar_height
        bar_color = (
            highlight_color if (highlight_key is not None and model_name == highlight_key)
            else palette[i % len(palette)]
        )
        bars = ax.barh(
            y + offset,
            percentages,
            height=bar_height,
            label=model_name,
            color=bar_color,
        )
        for bar, pct in zip(bars, percentages):
            ax.text(
                bar.get_width() + 1.0,
                bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="#1f2937",
            )

    ax.set_yticks(y)
    ax.set_yticklabels(class_names)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Probability (%)", fontsize=10, color="#1f2937")
    ax.set_title("Model Comparison", fontsize=12, fontweight="bold", color="#0a4a3d")
    ax.legend(loc="lower right", fontsize=9, frameon=False, labelcolor="#1f2937")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", colors="#1f2937")
    ax.tick_params(axis="x", colors="#1f2937")
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    fig.tight_layout()

    return fig


def run_single_model_prediction(model_name, image, class_names, registry=None):
    """
    Load the given model (by name, looked up in the model registry — built-in
    or user-uploaded) and run prediction on the provided image.

    Args:
        model_name: str, key into the model registry (an uploaded model's name)
        image: PIL.Image, the uploaded X-ray
        class_names: list[str], class labels
        registry: dict from get_model_registry(); computed if not provided

    Returns:
        dict with keys: success (bool), and on success: predicted_class,
        confidence, probability_vector, inference_time, class_names;
        on failure: error (str)
    """
    if registry is None:
        registry = get_model_registry()

    model_info = registry.get(model_name)
    if model_info is None:
        return {"success": False, "error": f"Model '{model_name}' was not found."}

    model_path = model_info["path"]
    mtime = os.path.getmtime(model_path) if os.path.exists(model_path) else 0.0
    model, error_message = load_model(model_path, mtime)

    if model is None:
        return {"success": False, "error": error_message}

    try:
        predicted_class, confidence, probability_vector, inference_time, resolved_class_names, mismatch_warning = (
            predict_image(
                image,
                model,
                class_names,
                preprocessing_method=model_info["default_preprocessing"],
            )
        )
        return {
            "success": True,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "probability_vector": probability_vector,
            "inference_time": inference_time,
            "class_names": resolved_class_names,
            "mismatch_warning": mismatch_warning,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def render_results_card(predicted_class, confidence, model_name, inference_time):
    """Render the results summary card with HTML/CSS styling."""
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-row">
                <span class="result-label">🔬 Prediction</span>
                <span class="result-value">{predicted_class}</span>
            </div>
            <div class="result-row">
                <span class="result-label">📊 Confidence</span>
                <span class="result-value">{confidence:.2f}%</span>
            </div>
            <div class="result-row">
                <span class="result-label">🤖 Model Used</span>
                <span class="result-value">{model_name}</span>
            </div>
            <div class="result-row">
                <span class="result-label">⏱️ Inference Time</span>
                <span class="result-value">{inference_time:.2f} ms</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================================================
# SIDEBAR
# ==================================================================================

def render_sidebar():
    """
    Render the sidebar:
      - Manage Custom Models: upload new models (unlimited), remove existing ones
      - Prediction mode: Single Model, or Ensemble (Multiple Models)

    Returns:
        (mode, selected) where mode is MODE_SINGLE / MODE_ENSEMBLE / None
        (None if no models are available yet), and selected is a model name
        (str) for single mode, a list of names for ensemble mode, or None.
    """
    init_custom_models_state()
    registry = get_model_registry()
    has_models = len(registry) > 0

    with st.sidebar:
        st.markdown("## 🫁 Control Panel")
        st.markdown("---")

        # ---------------- Manage Custom Models ----------------
        # Expanded by default until at least one model has been uploaded,
        # so a first-time user immediately sees where to add one.
        with st.expander("➕ Manage Custom Models", expanded=not has_models):
            st.caption("Upload your own trained .keras or .h5 model.")

            upload_key_suffix = st.session_state.get("upload_widget_version", 0)

            new_model_name = st.text_input(
                "Model name",
                key=f"new_model_name_{upload_key_suffix}",
                placeholder="e.g. My DenseNet121",
            )
            new_model_file = st.file_uploader(
                "Model file (.keras or .h5)",
                type=["keras", "h5"],
                key=f"new_model_file_{upload_key_suffix}",
            )
            new_model_preprocessing = st.selectbox(
                "Preprocessing used during training",
                options=list(PREPROCESSING_LABELS.keys()),
                format_func=lambda k: PREPROCESSING_LABELS[k],
                key=f"new_model_preproc_{upload_key_suffix}",
                help="This must match how you preprocessed images when training this model.",
            )

            if st.button("Add Model", use_container_width=True):
                if new_model_file is None:
                    st.warning("⚠️ Please choose a model file to upload.")
                else:
                    success, message = add_custom_model(
                        new_model_name, new_model_file, new_model_preprocessing
                    )
                    if success:
                        st.success(f"✅ {message}")
                        # Bump the widget key version so the form resets for the next upload
                        st.session_state.upload_widget_version = upload_key_suffix + 1
                        st.rerun()
                    else:
                        st.error(f"❌ {message}")

            custom_models = load_custom_models_registry()
            if custom_models:
                st.markdown("---")
                st.caption("Your uploaded models:")
                for name in list(custom_models.keys()):
                    col_name, col_delete = st.columns([4, 1])
                    with col_name:
                        st.markdown(f"🧩 {name}")
                    with col_delete:
                        if st.button("🗑️", key=f"delete_{name}", help=f"Remove {name}"):
                            remove_custom_model(name)
                            st.rerun()

        st.markdown("---")

        # ---------------- Prediction Mode ----------------
        if not has_models:
            st.info(
                "No models available yet. Upload one above under "
                "**'➕ Manage Custom Models'** to get started."
            )
            mode, selected = None, None
        else:
            all_model_names = list(registry.keys())

            st.markdown("### Prediction Mode")
            mode = st.radio(
                "Prediction mode:",
                options=[MODE_SINGLE, MODE_ENSEMBLE],
                label_visibility="collapsed",
            )

            if mode == MODE_SINGLE:
                selected = st.selectbox(
                    "Choose a model:",
                    options=all_model_names,
                    index=0,
                    label_visibility="collapsed",
                )
            else:
                default_selection = all_model_names[: min(2, len(all_model_names))]
                selected = st.multiselect(
                    "Choose models to combine:",
                    options=all_model_names,
                    default=default_selection,
                    label_visibility="collapsed",
                    help="Predictions from all selected models are averaged into one ensemble result.",
                )
                if len(selected) < 2:
                    st.caption("⚠️ Select at least 2 models to run an ensemble.")

        st.markdown("---")
        st.markdown(
            "<small>⚠️ For research/educational use only. "
            "Not a certified diagnostic tool.</small>",
            unsafe_allow_html=True,
        )

        return mode, selected



# ==================================================================================
# MAIN APP
# ==================================================================================

def main():
    inject_custom_css()
    init_custom_models_state()

    # ---------------- Header ----------------
    st.markdown(
        f"""
        <div class="app-header">
            <h1>🫁 {APP_TITLE}</h1>
            <p>{APP_SUBTITLE}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------- Sidebar ----------------
    mode, selected = render_sidebar()
    registry = get_model_registry()

    # ---------------- Load class names ----------------
    class_names_mtime = os.path.getmtime(CLASS_NAMES_FILE) if os.path.exists(CLASS_NAMES_FILE) else 0.0
    class_names = load_class_names(CLASS_NAMES_FILE, class_names_mtime)

    # ---------------- Layout: Upload (left) | Preview (right) ----------------
    left_col, right_col = st.columns([1, 1], gap="large")

    uploaded_file = None
    with left_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">📤 Upload Chest X-ray</div>', unsafe_allow_html=True)
            uploaded_file = st.file_uploader(
                "Drag and drop or browse a chest X-ray image",
                type=["jpg", "jpeg", "png"],
                help="Supported formats: JPG, JPEG, PNG",
            )
            predict_clicked = st.button("🔍 Predict", use_container_width=True)

    image = None
    with right_col:
        with st.container(border=True):
            st.markdown('<div class="section-title">🖼️ Image Preview</div>', unsafe_allow_html=True)
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file)
                    st.image(image, caption="Uploaded X-ray", use_container_width=True)
                except Exception as e:
                    st.error(f"Could not open the uploaded image: {e}")
            else:
                st.info("No image uploaded yet. Please upload a chest X-ray on the left.")

    # ---------------- Prediction Logic ----------------
    if predict_clicked:
        if uploaded_file is None or image is None:
            st.warning("⚠️ Please upload a chest X-ray image before predicting.")

        elif mode is None:
            st.warning(
                "⚠️ No models available yet. Please upload at least one model "
                "in the sidebar under 'Manage Custom Models' before predicting."
            )

        elif mode == MODE_SINGLE:
            if not selected:
                st.warning("⚠️ Please select a model in the sidebar.")
            else:
                # ----- Single model mode -----
                with st.spinner(f"Running inference with {selected}..."):
                    result = run_single_model_prediction(selected, image, class_names, registry)

                if not result["success"]:
                    st.error(f"❌ Error loading model: {result['error']}")
                else:
                    st.success("✅ Prediction complete!")
                    if result.get("mismatch_warning"):
                        st.warning(f"⚠️ {result['mismatch_warning']}")

                    predicted_class = result["predicted_class"]
                    confidence = result["confidence"]
                    probability_vector = result["probability_vector"]
                    inference_time = result["inference_time"]
                    resolved_class_names = result["class_names"]

                    # ---------------- Results Card ----------------
                    with st.container(border=True):
                        st.markdown('<div class="section-title">📋 Prediction Result</div>', unsafe_allow_html=True)
                        render_results_card(
                            predicted_class, confidence, selected, inference_time
                        )

                        # ---------------- Confidence Progress Bar ----------------
                        st.markdown("#### Confidence Level")
                        st.progress(min(int(confidence), 100))

                    # ---------------- Probability Chart ----------------
                    with st.container(border=True):
                        st.markdown('<div class="section-title">📊 Probability Breakdown</div>', unsafe_allow_html=True)
                        fig = plot_probabilities(probability_vector, resolved_class_names, predicted_class)
                        st.pyplot(fig, use_container_width=True)

                        # ---------------- Raw Probability Table ----------------
                        with st.expander("View raw probability table"):
                            df = pd.DataFrame(
                                {
                                    "Class": resolved_class_names,
                                    "Probability (%)": (probability_vector * 100).round(2),
                                }
                            ).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
                            st.dataframe(df, use_container_width=True)

        else:
            # ----- Ensemble mode: run every selected model, then average -----
            if len(selected) < 2:
                st.warning("⚠️ Please select at least 2 models in the sidebar to run an ensemble.")
            else:
                with st.spinner(f"Running inference with {len(selected)} models..."):
                    results = {}
                    errors = {}
                    for model_name in selected:
                        result = run_single_model_prediction(model_name, image, class_names, registry)
                        if result["success"]:
                            results[model_name] = result
                        else:
                            errors[model_name] = result["error"]

                for model_name, error_message in errors.items():
                    st.error(f"❌ {model_name} failed to load: {error_message}")

                if not results:
                    st.error("❌ None of the selected models could be loaded.")
                else:
                    # Only models whose output length matches the reference class list
                    # can be meaningfully averaged together.
                    reference_length = len(results[next(iter(results))]["class_names"])
                    valid_results = {
                        name: r for name, r in results.items()
                        if len(r["probability_vector"]) == reference_length
                    }
                    mismatched = set(results.keys()) - set(valid_results.keys())
                    if mismatched:
                        st.warning(
                            "⚠️ Excluded from the ensemble average because their number of "
                            f"output classes doesn't match the others: {', '.join(mismatched)}"
                        )

                    for model_name, result in results.items():
                        if result.get("mismatch_warning"):
                            st.warning(f"⚠️ {model_name}: {result['mismatch_warning']}")

                    st.success(f"✅ Prediction complete for {len(results)} model(s)!")

                    # ---------------- Individual Model Results ----------------
                    with st.container(border=True):
                        st.markdown('<div class="section-title">📋 Individual Model Results</div>', unsafe_allow_html=True)
                        result_cols = st.columns(len(results))
                        for col, (model_name, result) in zip(result_cols, results.items()):
                            with col:
                                render_results_card(
                                    result["predicted_class"],
                                    result["confidence"],
                                    model_name,
                                    result["inference_time"],
                                )

                    if len(valid_results) < 2:
                        st.error(
                            "❌ Need at least 2 models with a matching number of output "
                            "classes to compute an ensemble average."
                        )
                    else:
                        # ---------------- Ensemble Combined Result ----------------
                        resolved_class_names = next(iter(valid_results.values()))["class_names"]
                        ensemble_vector = average_probability_vectors(
                            [r["probability_vector"] for r in valid_results.values()]
                        )
                        ensemble_index = int(np.argmax(ensemble_vector))
                        ensemble_class = resolved_class_names[ensemble_index]
                        ensemble_confidence = float(ensemble_vector[ensemble_index]) * 100.0
                        ensemble_total_time = sum(r["inference_time"] for r in valid_results.values())

                        with st.container(border=True):
                            st.markdown(
                                f'<div class="section-title">🧬 {ENSEMBLE_RESULT_LABEL} — '
                                f'{len(valid_results)} Models Combined</div>',
                                unsafe_allow_html=True,
                            )
                            render_results_card(
                                ensemble_class,
                                ensemble_confidence,
                                f"{len(valid_results)} models (averaged)",
                                ensemble_total_time,
                            )
                            st.markdown("#### Ensemble Confidence Level")
                            st.progress(min(int(ensemble_confidence), 100))

                        # ---------------- Comparison Chart (per-model + ensemble) ----------------
                        with st.container(border=True):
                            st.markdown('<div class="section-title">📊 Probability Comparison</div>', unsafe_allow_html=True)
                            comparison_data = {
                                name: r["probability_vector"] for name, r in valid_results.items()
                            }
                            comparison_data[ENSEMBLE_RESULT_LABEL] = ensemble_vector
                            fig = plot_model_comparison(
                                comparison_data,
                                resolved_class_names,
                                highlight_key=ENSEMBLE_RESULT_LABEL,
                                highlight_color=ENSEMBLE_RESULT_COLOR,
                            )
                            st.pyplot(fig, use_container_width=True)

                            with st.expander("View raw ensemble probability table"):
                                df = pd.DataFrame(
                                    {
                                        "Class": resolved_class_names,
                                        "Probability (%)": (ensemble_vector * 100).round(2),
                                    }
                                ).sort_values("Probability (%)", ascending=False).reset_index(drop=True)
                                st.dataframe(df, use_container_width=True)

    # ---------------- Footer ----------------
    st.markdown(
        """
        <div class="footer">
            Developed by <b>Mahbubul Islam</b>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
