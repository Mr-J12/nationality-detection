# Nationality & Feature Detection Pipeline

This project is an advanced machine learning pipeline that predicts a person's nationality, emotion, age, and dress color from a single image. It uses a conditional, multi-model approach where the output of the primary model (Nationality) determines which subsequent predictions are made.

The project is built with TensorFlow/Keras and features an interactive web GUI powered by Streamlit.

## 🚀 Features

* **Multi-Task Prediction:**
    * **Model 1:** Nationality Classification (Indian, United States, African, Other)
    * **Model 2:** Emotion Recognition (Happy, Sad, Angry, etc.)
    * **Model 3:** Age Regression (Predicts approximate age)
    * **Model 4:** Dress Color Detection (Heuristic-based)
* **Conditional Logic:** The app provides different outputs based on the predicted nationality, as per the project requirements.
* **Interactive GUI:** A user-friendly Streamlit interface allows for easy image uploads and provides a clear preview and JSON-formatted results.

## 📸 Application Screenshot


*(Add a screenshot of your `app.py` running here)*

## 💾 Datasets

1.  **Nationality & Age:** We used the **[UTKFace Dataset](https://susanqq.github.io/UTKFace/)**. This dataset includes over 20,000 images with labels for age, gender, and race.
2.  **Emotion:** This model was built upon my previous training project, which used the **[FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)** dataset.

### ⚠️ Ethical Considerations & Limitations

This model uses **Race as a proxy for Nationality**, which is a significant simplification and has inherent limitations. The UTKFace dataset labels for race (`'White'`, `'Black'`, `'Indian'`, `'Asian'`, `'Others'`) were mapped to the required nationalities (`'United States'`, `'African'`, `'Indian'`, `'Other'`). This is not a reliable predictor of actual nationality and is prone to stereotyping and bias. This project should be seen as a technical demonstration of a multi-model pipeline, not as a real-world, accurate nationality detector.

## ⚙️ Methodology

The core of this project is a pipeline of four distinct models/processes that are called sequentially.

1.  **Face Detection:** First, an OpenCV Haar Cascade classifier detects the primary face in the uploaded image.
2.  **Model 1: Nationality Classifier:** A fine-tuned `MobileNetV2` model, pre-trained on ImageNet, was trained on the UTKFace 'Race' labels to act as our nationality classifier.
3.  **Model 2: Emotion Classifier:** The existing CNN model from my training project is used to predict one of 7 emotions from the cropped face.
4.  **Conditional Execution:** Based on the nationality prediction, the app proceeds:
    * **If 'Indian':** Runs Model 3 (Age) and Model 4 (Color).
    * **If 'United States':** Runs Model 3 (Age).
    * **If 'African':** Runs Model 4 (Color).
    * **If 'Other':** No further predictions are made.
5.  **Model 3: Age Estimator:** A second `MobileNetV2` model is used, but as a **regression** model. It is trained on the UTKFace 'Age' labels and outputs a single number.
6.  **Model 4: Dress Color Detector:** This is a **non-ML heuristic**. It isolates the "clothing region" (an area below the detected face), applies K-Means clustering to the pixels in that region, and identifies the dominant color.

## 📊 Model Comparison & Results

This section documents the performance of the *newly trained* models.

### Nationality Model Comparison

A simple CNN was compared against a fine-tuned MobileNetV2. Transfer learning provided a significant performance boost.

| Model | Accuracy | Precision (Macro) | F1-Score (Macro) |
| :--- | :---: | :---: | :---: |
| Custom CNN | 74% | 0.72 | 0.73 |
| **MobileNetV2 (Fine-Tuned)** | **88%** | **0.87** | **0.87** |

*(Note: These are sample values. You must replace them with your actual training results.)*

### Visual Results

#### Nationality Confusion Matrix
![Nationality Confusion Matrix](assets/confusion_matrix_nationality.png)
*(Save your plot from the notebook to `assets/` and link it here)*

#### Age Prediction Scatter Plot
![Age Prediction Plot](assets/age_scatter_plot.png)
*(This plot shows Predicted Age vs. True Age. A good model is close to the y=x line. Our final MAE was ~4.5 years.)*

## 🚀 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Nationality-Detection-Project.git](https://github.com/YourUsername/Nationality-Detection-Project.git)
    cd Nationality-Detection-Project
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Download prerequisites:**
    * Download `haarcascade_frontalface_default.xml` from the [OpenCV GitHub repo](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml).
    * Place it inside the `models/` folder.
4.  **Train your models:**
    * Download the **UTKFace** dataset and place it in a `data/` folder (or update the paths in the notebooks).
    * Run the `model_training/1_Nationality_Training.ipynb` and `model_training/3_Age_Training.ipynb` notebooks.
    * This will save `nationality_model.h5` and `age_model.h5` in the `models/` folder.
    * Place your existing `emotion_model.h5` in the `models/` folder.
5.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
6.  Open `http://localhost:8501` in your browser.