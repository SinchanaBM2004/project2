import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image, ImageOps
import streamlit.components.v1 as components


# =========================================================
# CONFIG
# =========================================================

MODEL_PATH = "char_digit_model.h5"
LABEL_PATH = "label_map.json"

IMG_SIZE = (64, 64)


# =========================================================
# PAGE
# =========================================================

st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="centered"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    return tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


# =========================================================
# LOAD LABELS
# =========================================================

@st.cache_data
def load_labels():

    with open(LABEL_PATH, "r") as file:
        labels = json.load(file)

    return {
        int(k): v
        for k, v in labels.items()
    }


# =========================================================
# CHECK FILES
# =========================================================

if not os.path.exists(MODEL_PATH):

    st.error(
        f"❌ {MODEL_PATH} not found."
    )

    st.stop()


if not os.path.exists(LABEL_PATH):

    st.error(
        f"❌ {LABEL_PATH} not found."
    )

    st.stop()


model = load_model()
label_map = load_labels()


# =========================================================
# PREPROCESS IMAGE
# =========================================================

def preprocess_image(image):

    # Convert to grayscale
    image = image.convert("L")

    # Resize exactly as training
    image = image.resize(
        IMG_SIZE,
        Image.Resampling.LANCZOS
    )

    # Convert to numpy
    image_array = np.array(
        image,
        dtype=np.float32
    )

    # Shape:
    # (64,64)
    # -> (64,64,1)
    # -> (1,64,64,1)

    image_array = np.expand_dims(
        image_array,
        axis=-1
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =========================================================
# BROWSER SPEECH
# =========================================================

def speak_browser(text):

    safe_text = (
        text
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", " ")
    )

    html = f"""
    <script>

    const text = '{safe_text}';

    const speech = new SpeechSynthesisUtterance(text);

    speech.rate = 0.9;
    speech.pitch = 1.0;
    speech.volume = 1.0;

    window.speechSynthesis.cancel();

    window.speechSynthesis.speak(speech);

    </script>
    """

    components.html(
        html,
        height=0
    )


# =========================================================
# TITLE
# =========================================================

st.title("✍️ Handwritten Character Recognition")

st.write(
    "Upload a handwritten digit or alphabet "
    "and let the CNN model recognize it."
)

st.divider()


# =========================================================
# UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "Upload handwritten image",
    type=["png", "jpg", "jpeg"]
)


# =========================================================
# PROCESS
# =========================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input image",
        width=250
    )


    # =====================================================
    # PREDICT
    # =====================================================

    if st.button(
        "🔍 Predict",
        use_container_width=True
    ):

        processed = preprocess_image(
            image
        )

        # Prediction
        predictions = model.predict(
            processed,
            verbose=0
        )[0]


        # =================================================
        # TOP 3 PREDICTIONS
        # =================================================

        top_indices = np.argsort(
            predictions
        )[-3:][::-1]


        predicted_index = int(
            top_indices[0]
        )

        predicted_character = label_map[
            predicted_index
        ]

        confidence = (
            float(predictions[predicted_index])
            * 100
        )


        # Save results in session
        st.session_state["prediction"] = (
            predicted_character
        )

        st.session_state["confidence"] = (
            confidence
        )

        st.session_state["predictions"] = [
            (
                label_map[int(i)],
                float(predictions[i]) * 100
            )
            for i in top_indices
        ]


# =========================================================
# DISPLAY RESULT
# =========================================================

if "prediction" in st.session_state:

    prediction = (
        st.session_state["prediction"]
    )

    confidence = (
        st.session_state["confidence"]
    )

    predictions = (
        st.session_state["predictions"]
    )


    st.divider()

    st.subheader("🎯 Prediction")

    st.success(
        f"Predicted Character: **{prediction}**"
    )

    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )


    # =====================================================
    # CONFIDENCE WARNING
    # =====================================================

    if confidence < 70:

        st.warning(
            "⚠️ The model is not very confident. "
            "Try uploading a clearer handwritten image."
        )

    elif confidence < 90:

        st.info(
            "The prediction is reasonable, "
            "but the model has some uncertainty."
        )

    else:

        st.success(
            "✅ High confidence prediction."
        )





    # =====================================================
    # SPEECH TEXT
    # =====================================================

    digit_words = {

        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine"

    }


    if prediction in digit_words:

        speech_text = (
            "The predicted number is "
            + digit_words[prediction]
        )

    else:

        speech_text = (
            "The predicted character is "
            + prediction
        )


    st.info(
        f"🔊 {speech_text}"
    )


    # =====================================================
    # SPEAK
    # =====================================================

    if st.button(
        "🔊 Speak Result",
        use_container_width=True
    ):

        speak_browser(
            speech_text
        )