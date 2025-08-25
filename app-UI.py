# app-2.py

import streamlit as st # type: ignore
import tensorflow as tf # type: ignore
import numpy as np # type: ignore
from PIL import Image # type: ignore
from lime import lime_image # type: ignore
from skimage.segmentation import mark_boundaries # type: ignore

# ------------------------------------------
# Configuration
# ------------------------------------------
class_names = ["glioma", "meningioma", "no_tumor", "pituitary_tumor"]
model = tf.keras.models.load_model("brain_tumor_classifier.h5")
explainer = lime_image.LimeImageExplainer()

# ------------------------------------------
# Utility Functions
# ------------------------------------------

def preprocess_image(image_file):
    image = Image.open(image_file).convert("RGB").resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict(image_file):
    image = preprocess_image(image_file)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    return predicted_class, predictions[0]

def get_lime_explanation(image_file, model):
    image = Image.open(image_file).convert("RGB").resize((224, 224))
    image_np = np.array(image) / 255.0
    image_for_lime = (image_np * 255).astype(np.uint8)

    def predict_fn(images):
        images = np.array(images) / 255.0
        return model.predict(images)

    explanation = explainer.explain_instance(
        image=image_for_lime,
        classifier_fn=predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )

    top_label = explanation.top_labels[0]
    temp, mask = explanation.get_image_and_mask(
        label=top_label,
        positive_only=True,
        hide_rest=False,
        num_features=10
    )

    lime_img = mark_boundaries(temp, mask)
    return lime_img, top_label

# ------------------------------------------
# Streamlit App
# ------------------------------------------

def main():
    st.set_page_config(page_title="Brain Tumor Classifier", layout="wide")
    st.title("Brain Tumor Classification with LIME Explainability")

    st.sidebar.header("Upload MRI Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if "predicted" not in st.session_state:
        st.session_state.predicted = None
        st.session_state.probs = None
        st.session_state.lime_image = None
        st.session_state.lime_class = None
        st.session_state.raw_image = None

    if uploaded_file:
        st.session_state.raw_image = uploaded_file

        if st.sidebar.button("Predict"):
            predicted_class, probs = predict(uploaded_file)
            st.session_state.predicted = predicted_class
            st.session_state.probs = probs
            st.session_state.lime_image = None  # Reset lime output

        if st.sidebar.button("Explain with LIME"):
            with st.spinner("Generating LIME explanation..."):
                lime_img, lime_class = get_lime_explanation(uploaded_file, model)
                st.session_state.lime_image = lime_img
                st.session_state.lime_class = lime_class

    if st.session_state.raw_image:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(st.session_state.raw_image, caption="Uploaded Image", use_container_width=True)

        with col2:
            if st.session_state.lime_image is not None:
                st.subheader("LIME Explanation")
                st.image(st.session_state.lime_image, caption=f"LIME for class: {class_names[st.session_state.lime_class]}", use_container_width=True)

        if st.session_state.predicted is not None:
            st.subheader("Prediction Results")
            st.markdown(f"**Predicted Class:** `{class_names[st.session_state.predicted]}`")
            st.markdown("**Class Probabilities:**")
            for i, prob in enumerate(st.session_state.probs):
                st.write(f"- **{class_names[i]}:** `{prob:.4f}`")

            confidence = np.max(st.session_state.probs) * 100
            st.markdown(f"**Confidence:** `{confidence:.2f}%`")
    else:
        st.info("Upload an image to get started.")

# ------------------------------------------
if __name__ == "__main__":
    main()
