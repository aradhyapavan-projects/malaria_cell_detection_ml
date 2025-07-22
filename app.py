import streamlit as st
import joblib
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog
import matplotlib.pyplot as plt
import os
import base64

# Asyncio patch for Streamlit+Python 3.11 compatibility
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Page configuration
st.set_page_config(
    page_title="Malaria Cell Detection",
    page_icon="🔬",
    layout="wide"
)

# Function to download example images
def get_binary_file_downloader_html(bin_file, file_label):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">{file_label}</a>'
    return href

# Load the model and label encoder
@st.cache_resource
def load_model():
    model_path = 'malaria_svm_hog_model.joblib'
    le_path = 'label_encoder.joblib'
    if not os.path.exists(model_path) or not os.path.exists(le_path):
        st.error("Model files not found. Please run the notebook to export the model first.")
        st.stop()
    model = joblib.load(model_path)
    label_encoder = joblib.load(le_path)
    return model, label_encoder

# Extract HOG features from image
def extract_hog_features(image):
    IMG_DIMS = (128, 64)  # Height x Width
    image = resize(image, IMG_DIMS, anti_aliasing=True)
    features, hog_image = hog(
        image, orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        visualize=True, 
        channel_axis=-1 if image.ndim == 3 else None
    )
    return features, hog_image

def main():
    model, label_encoder = load_model()
    st.title("🔬 Malaria Cell Detection")

    st.info("""
    ### Model Performance
    - **Training Accuracy**: 92.2%
    - **Validation Accuracy**: 80.4%
    ⚠️ **Disclaimer**: This model is for educational purposes and experimentation only. Please use with caution and do not use for actual medical diagnosis.
    """)

    with st.expander("📚 Documentation"):
        st.markdown("""
        ## Problem Statement
        Malaria remains one of the world's deadliest infectious diseases, affecting millions of people worldwide. Early and accurate diagnosis is crucial for effective treatment and reducing mortality rates. Traditional diagnosis involves manual microscopic examination of blood smears by trained professionals, which is time-consuming, requires expertise, and can be subject to human error.

        ## Project Approach
        This project uses machine learning to automate the detection of malaria parasites in blood smear images:

        1. **Data Collection**: Used a dataset of labeled blood cell images (parasitized and uninfected)
        2. **Feature Extraction**: Applied Histogram of Oriented Gradients (HOG) to extract meaningful features from the cell images
        3. **Model Training**: Trained a Support Vector Machine (SVM) classifier on the extracted features
        4. **Evaluation**: Validated the model on a separate test set to ensure generalizability

        ## How It Works
        The application follows these steps to analyze a blood cell image:

        1. **Image Upload**: User uploads a microscopic blood cell image
        2. **Preprocessing**: The image is resized and normalized
        3. **Feature Extraction**: HOG features are extracted to capture the cell's morphological characteristics
        4. **Classification**: The SVM model predicts whether the cell is infected with malaria parasites
        5. **Result Display**: The prediction result and confidence score are displayed along with visual aids

        ## Technical Details
        - **Feature Extraction**: HOG with 9 orientations, 8×8 pixels per cell, and 2×2 cells per block
        - **Classifier**: SVM with RBF kernel
        - **Image Size**: Processed at 128×64 pixels
        """)

    # Example images section
    st.markdown("### Example Images")
    example_folder = "examples"
    malaria_files = [
        "C50P11thinF_IMG_20150724_114951_cell_148.png",
        "C59P20thinF_IMG_20150803_111333_cell_144.png"
    ]
    healthy_files = [
        "C112P73ThinF_IMG_20150930_131659_cell_94.png",
        "C130P91ThinF_IMG_20151004_142709_cell_110.png"
    ]
    if os.path.exists(example_folder):
        st.success("Example images are available in the examples folder. You can use them to test the model.")
        col_examples1, col_examples2 = st.columns(2)
        with col_examples1:
            st.markdown("#### Parasitized (Malaria) Examples:")
            for file in malaria_files:
                file_path = os.path.join(example_folder, file)
                if os.path.exists(file_path):
                    try:
                        img = imread(file_path)
                        st.image(img, caption=file, width=250)
                        st.markdown(get_binary_file_downloader_html(file_path, f"Download {file}"), unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not display {file}: {e}")
        with col_examples2:
            st.markdown("#### Uninfected (Healthy) Examples:")
            for file in healthy_files:
                file_path = os.path.join(example_folder, file)
                if os.path.exists(file_path):
                    try:
                        img = imread(file_path)
                        st.image(img, caption=file, width=250)
                        st.markdown(get_binary_file_downloader_html(file_path, f"Download {file}"), unsafe_allow_html=True)
                    except Exception as e:
                        st.warning(f"Could not display {file}: {e}")
    else:
        st.warning("Examples folder not found. You can create an 'examples' folder with the following files:")
        col_examples1, col_examples2 = st.columns(2)
        with col_examples1:
            st.markdown("#### Parasitized (Malaria) Examples:")
            for file in malaria_files:
                st.markdown(f"- {file}")
        with col_examples2:
            st.markdown("#### Uninfected (Healthy) Examples:")
            for file in healthy_files:
                st.markdown(f"- {file}")

    st.markdown("### Upload Blood Cell Image")
    st.write("Upload an image of a blood cell to detect whether it's infected with malaria parasites.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    prediction_result = None
    confidence_score = None
    hog_img = None
    original_image = None

    if uploaded_file is not None:
        try:
            image = imread(uploaded_file)
            original_image = image.copy()
            features, hog_img = extract_hog_features(image)
            prediction = model.predict([features])[0]
            prediction_proba = model.decision_function([features])[0] if hasattr(model, "decision_function") else 0
            confidence_score = abs(prediction_proba)
            prediction_result = label_encoder.inverse_transform([prediction])[0]

            st.markdown("---")
            st.header("Prediction Results")
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown("### Uploaded Image")
                st.image(original_image, caption="Original Blood Cell Image", width=300)
                st.markdown("### Image Analysis")
                st.markdown(f"""
                - **Image Size**: {original_image.shape[0]} × {original_image.shape[1]} pixels
                - **Color Channels**: {original_image.shape[2] if len(original_image.shape) > 2 else "Grayscale"}
                - **Processing**: HOG feature extraction with 9 orientations
                """)
            with res_col2:
                st.markdown("### Detection Result")
                if prediction_result.lower() in ["parasitized", "malaria"]:
                    st.error(f"## Prediction: {prediction_result}")
                    st.markdown("⚠️ **Cell appears to be infected with malaria parasites**")
                else:
                    st.success(f"## Prediction: {prediction_result}")
                    st.markdown("✅ **Cell appears to be uninfected**")
                st.metric("Confidence Score", f"{confidence_score:.2f}")
                st.markdown("### HOG Feature Visualization")
                fig, ax = plt.subplots(figsize=(5, 8))
                ax.imshow(hog_img, cmap="gray")
                ax.set_title("HOG Features")
                st.pyplot(fig)

            st.markdown("---")
            st.markdown("""
            ### Understanding the Results

            - **HOG Features:** The **Histogram of Oriented Gradients (HOG)** visualization shows the dominant edge and gradient directions present in the cell, capturing the unique morphological patterns that help the model distinguish infected from uninfected cells.
            - **Confidence Score:** This reflects how strongly the SVM model separates this cell from its decision boundary; higher values indicate higher model certainty. Lower values suggest the prediction is less certain and should be interpreted with extra caution.
            - **Limitations:** This model has approximately 80% accuracy on validation data. Results may vary based on staining quality, microscope settings, and sample differences. **It is not a substitute for expert clinical diagnosis.**

            **How HOG Visualization Helps:**  
            HOG transforms complex microscopic images into structured gradient information, enabling the SVM to detect subtle shape and texture features typical of malaria-infected cells. The HOG image highlights regions with strong directionality—these are the patterns on which the machine learning model relies.

            **Clinical Disclaimer:**  
            All results are for illustrative and experimental purposes. No medical decisions should be made on the basis of this tool.
            """)
        except Exception as e:
            st.error(f"Error processing image: {e}")

    st.markdown("---")
    st.markdown("""
    ### Educational Purpose Only

    This application is designed for educational and experimental purposes only. It demonstrates the potential of machine learning in medical image analysis but is not intended for clinical use.

    **Important Notice**: Do not use this tool for actual medical diagnosis. Always consult qualified healthcare professionals for medical advice and diagnosis.
    """)

    st.markdown("""
    <div style="border: 1px solid rgba(128, 128, 128, 0.3); padding: 15px; text-align: center; margin-top: 20px; border-radius: 8px; background-color: rgba(240, 240, 240, 0.3);">
        <p style="margin: 0; font-weight: 500; font-size: 16px;">Malaria Detection Model | Created for Educational Purposes</p>
        <div style="margin: 8px 0; height: 1px; background: rgba(128, 128, 128, 0.2);"></div>
        <p style="margin: 0; font-size: 14px;">Designed & Developed by <b>Aradhya Pavan</b> | SVM with HOG Features</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
