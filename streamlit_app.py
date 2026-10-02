from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
import keras

#set model settings for CNN
IMAGE_SIZE = 64
CLASS_NAMES = ["Healthy", "Crowned", "Root Canal"]
APP_DIRECTORY = Path(__file__).resolve().parent
MODEL_PATH = APP_DIRECTORY / "dental_xray_cnn.keras"
EXAMPLE_IMAGES = {
    "Healthy": APP_DIRECTORY / "example_images" / "healthy.png",
    "Crowned": APP_DIRECTORY / "example_images" / "crowned.png",
    "Root Canal": APP_DIRECTORY / "example_images" / "root_canal.png",
}


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


st.set_page_config(page_title="Dental Xray Classifier", layout="centered")
st.title("Dental X-ray CNN Classifier")

st.write(
    "Try an example periapical dental Xray or upload your own. "
    "The CNN model classifies it as Healthy, Crowned, or Root Canal."
)

st.warning("This is a machine learning project and is not intended for medical use.")

image_source = st.radio(
    "Choose an image source",
    ["Try an example", "Upload your own"],
    horizontal=True,
)

image_input = None
image_caption = "Uploaded X-ray"

if image_source == "Try an example":
    example_label = st.selectbox(
        "Choose an example X-ray",
        list(EXAMPLE_IMAGES)
    )
    example_path = EXAMPLE_IMAGES[example_label]

    if example_path.is_file():
        image_input = example_path
        image_caption = f"Example X-ray — dataset label: {example_label}"
    else:
        st.error(
            "This example image is unavailable. "
            "You can still upload your own X-ray."
        )

    st.caption(
        "Examples from DentIRO by Md. Mehedi Hasan Shoib et al. "
        "([dataset](https://doi.org/10.6084/m9.figshare.32086377), "
        "[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). "
        "These images come from the project dataset and illustrate how the app works; "
        "they are not a separate evaluation of the model."
    )


else: 
    image_input = st.file_uploader(
        "Upload a dental X-ray",
        type=["png", "jpg", "jpeg"]
    )

if image_input is not None:
    try:
        with Image.open(image_input) as source_image:
            uploaded_image = source_image.copy()

        display_image = uploaded_image.convert("RGB")

        st.image(
            display_image,
            caption=image_caption,
            use_container_width=True
        )

        if st.button("Classify X-ray", type="primary"):
            processed_image = preprocess_image(uploaded_image)

            with st.spinner("Analysing X-ray..."):
                model = load_cnn_model()
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