
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
import keras

#set model settings for CNN
IMAGE_SIZE = 64
CLASS_NAMES = ["Healthy", "Crowned", "Root Canal"]
MODEL_PATH = Path(__file__).resolve().parent / "dental_xray_cnn.keras"


@st.cache_resource
def load_cnn_model():
    """load CNN model"""
    return keras.models.load_model(MODEL_PATH, compile=False)


def preprocess_image(image):
    """preproccess uploaded image"""

    #convert to grayscale
    image = image.convert("L")
    image_array = np.array(image)

    #crop black area
    mask = image_array > 5

    if mask.any():
        rows, columns = np.where(mask)

        image_array = image_array[
            rows.min():rows.max() + 1,
            columns.min():columns.max() + 1
        ]

    #resize
    image = Image.fromarray(image_array)
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE),Image.Resampling.LANCZOS)

    #convert between 0 and 1
    image_array = np.array(image, dtype=np.float32) / 255.0

    #change shape to (1, 64, 64, 1)
    image_array = image_array[np.newaxis, ..., np.newaxis]
    return image_array

st.set_page_config(page_title="Dental X-ray Classifier", layout="centered")
st.title("Dental X-ray CNN Classifier")
st.write("Upload a periapical dental X-ray to have a CNN model classify it as either Healthy, Crowned, or Root Canal.")
st.warning("This is a machine-learning project and is not intended for medical use.")

uploaded_file = st.file_uploader("Upload a dental X-ray",type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        uploaded_image = Image.open(uploaded_file)
        display_image = uploaded_image.convert("RGB")
        st.image(display_image, caption="Uploaded xray", use_container_width=True)

        if st.button("Classify X-ray", type="primary"):
            model = load_cnn_model()
            processed_image = preprocess_image(uploaded_image)

            with st.spinner("Analysing X-ray..."):
                probabilities = model.predict(processed_image, verbose=0)[0]

            predicted_number = int(np.argmax(probabilities))
            predicted_class = CLASS_NAMES[predicted_number]
            highest_probability = float(probabilities[predicted_number])

            st.success(f"Prediction: {predicted_class}")
            st.metric("Model probability score", f"{highest_probability:.1%}")
            st.subheader("Scores for each class")

            for class_name, probability in zip(CLASS_NAMES, probabilities):
                st.write(f"**{class_name}:** {probability:.1%}")
                st.progress(float(probability))

            st.caption("softmax scores describe the model's output, not medical advice.")

    except UnidentifiedImageError:
        st.error("The file you uploaded is not an image.")
    except Exception as error:
        st.error(f"The image could not be classified: {error}")
