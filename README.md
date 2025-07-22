

# Malaria Cell Detection 🚀

## Problem Statement
Malaria is a life-threatening disease caused by parasites transmitted through the bites of infected mosquitoes. Manual diagnosis from blood smear images is time-consuming and requires expert knowledge. This project aims to automate malaria cell detection using machine learning techniques applied to cell images, making diagnosis faster and more accessible.

## Approach

1. **Data Exploration**
   - Analyzed the dataset structure and checked class balance between infected and uninfected cells.

2. **Preprocessing**
   - Encoded categorical labels.
   - Resized images for uniformity.
   - Split data into training and validation sets.

3. **Feature Extraction**
   - **Raw Pixel Values:** Images resized to 40x40 and flattened.
   - **Histogram of Oriented Gradients (HOG):** Images resized to 128x64, then HOG features extracted for better texture representation.

4. **Model Training**
   - Trained both Logistic Regression and Linear SVM models on each feature type.
   - Compared performance to select the best approach.

5. **Evaluation**
   - Used accuracy metrics to compare models on training and validation sets.

6. **Model Saving**
   - Saved the best-performing model and the label encoder using `joblib` for easy reuse and deployment.

7. **Deployment**
   - Built a Streamlit app for user-friendly prediction and visualization.

## Results

| Feature Type   | Model                | Training Accuracy (%) | Validation Accuracy (%) |
|----------------|----------------------|----------------------|------------------------|
| Pixel Values   | Logistic Regression  | 99.7                 | 74.2                   |
| Pixel Values   | Linear SVM           | 99.7                 | 74.1                   |
| HOG Features   | Logistic Regression  | 85.2                 | 81.6                   |
| HOG Features   | Linear SVM           | 85.2                 | 81.7                   |

- **Best Model:** Linear SVM with HOG features (Validation Accuracy: 81.7%)

## Model and Streamlit App

- The trained SVM model (`malaria_svm_hog_model.joblib`) and label encoder (`label_encoder.joblib`) are saved for inference.
- The Streamlit app (`app.py`) loads these files to make predictions on new images and displays the result and confidence score.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repo-url>
cd <repo-folder>
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. (Optional) Extract Images

If you have a zipped dataset, extract it:

```bash
unzip cell_images.zip
```

### 4. Train the Model

You can train the model using the provided Jupyter notebook or script:

- Open and run `Malaria_Detection_SVM_HOG.ipynb` in Jupyter Notebook, **or**
- Use the provided scripts if available.

This will generate the model and label encoder files (`malaria_svm_hog_model.joblib`, `label_encoder.joblib`).

### 5. Run the Streamlit App

To launch the web interface locally (without Docker), run:

```bash
streamlit run app.py
```

The app will be available at [http://localhost:8501](http://localhost:8501) by default.

---

## Running with Docker

You can also run the app inside a Docker container for a fully isolated environment.

### 1. Build the Docker Image

```bash
docker build -t malaria-streamlit-app .
```

### 2. Run the Docker Container

```bash
docker run -p 8501:8501 malaria-streamlit-app
```

Then open [http://localhost:8501](http://localhost:8501) in your browser to use the app.

## Folder Structure

```
app.py
cell_images.zip
Dockerfile
extract.py
label_encoder.joblib
Malaria_Detection_SVM_HOG.ipynb
malaria_svm_hog_model.joblib
README.md
requirements.txt
start.sh
train.csv
examples/
    C112P73ThinF_IMG_20150930_131659_cell_94.png
    ...
Snapshot/
    1.image_samples.png
    2.pixel_features.png
    3.hog_feature_image.png
    4.hog_features.png
    5.streamlit_app.png
    6.Malaria_cell_infected_Result.png
    7.healthy_cell_prediction.png
```

## Snapshots

Below are some visualizations and results from the project, located in the `Snapshot` folder:

- **1.image_samples.png:** Example cell images from the dataset.
- **2.pixel_features.png:** Visualization of raw pixel features.
- **3.hog_feature_image.png:** HOG feature visualization for a sample image.
- **4.hog_features.png:** Distribution of HOG features.
- **5.streamlit_app.png:** Screenshot of the Streamlit app interface.
- **6.Malaria_cell_infected_Result.png:** Example of an infected cell prediction.
- **7.healthy_cell_prediction.png:** Example of a healthy cell prediction.

You can view these images in the `Snapshot` directory for a better understanding of the workflow and results.

---

**License:** MIT
