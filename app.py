import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
from sklearn.cluster import KMeans
import os

# --- Configuration ---
st.set_page_config(page_title="Nationality & Feature Detector", layout="wide")

# --- Model Loading ---
# We cache the model loading so it only runs once.
@st.cache_resource
def load_models():
    """Load all required models and classifiers."""
    try:
        nat_model = tf.keras.models.load_model('models/nationality_model.h5')
        age_model = tf.keras.models.load_model('models/age_model.h5')
        emo_model = tf.keras.models.load_model('models/emotion_model.h5')
        
        # Load the Haar Cascade for face detection
        face_cascade_path = 'models/haarcascade_frontalface_default.xml'
        if not os.path.exists(face_cascade_path):
            st.error(f"Missing Haar Cascade file: {face_cascade_path}. Please download it.")
            return None, None, None, None
            
        face_cascade = cv2.CascadeClassifier(face_cascade_path)
        
        return nat_model, age_model, emo_model, face_cascade
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.error("Please make sure all model files ('nationality_model.h5', 'age_model.h5', 'emotion_model.h5') are in the 'models/' directory.")
        return None, None, None, None

# Load all models
nat_model, age_model, emo_model, face_cascade = load_models()

# --- Model & Task Definitions ---
# Define the labels for your models.
# UPDATE THESE to match your models' training.
NATIONALITY_LABELS = {0: 'African', 1: 'Indian', 2: 'Other', 3: 'United States'}
EMOTION_LABELS = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

# --- Preprocessing Functions ---
def preprocess_for_nat_age(face_image):
    """Preprocesses a cropped face image for the Nationality and Age models."""
    # These models were trained on 224x224 RGB images
    img_resized = cv2.resize(face_image, (224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array) # Use the correct prep function
    return img_array

def preprocess_for_emotion(face_image):
    """Preprocesses a cropped face image for the Emotion model."""
    # This model was likely trained on 48x48 Grayscale images
    try:
        img_gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    except cv2.error:
        img_gray = face_image # Already grayscale
        
    img_resized = cv2.resize(img_gray, (48, 48))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

# --- Color Detection Function ---
def get_dominant_color(image, face_box):
    """
    Finds the dominant color in the clothing region below the face.
    """
    try:
        (x, y, w, h) = face_box
        
        # Define clothing region: 1.5x width, 1x height below the face
        ch = int(h * 1.0) # Clothing height
        cw = int(w * 1.5) # Clothing width
        cx_start = max(0, x - int((cw - w) / 2)) # Center the clothing box
        cy_start = y + h
        
        # Ensure the region is within image bounds
        clothing_region = image[cy_start:min(cy_start + ch, image.shape[0]), 
                                cx_start:min(cx_start + cw, image.shape[1])]

        if clothing_region.size < 50: # Not enough pixels
            return "Unknown"

        # Reshape for K-Means
        pixels = clothing_region.reshape(-1, 3)
        
        # Run K-Means
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        kmeans.fit(pixels)
        
        # Get the most dominant cluster
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_cluster = kmeans.cluster_centers_[unique[counts.argmax()]]
        
        return rgb_to_color_name(dominant_cluster)
        
    except Exception as e:
        print(f"Color detection error: {e}")
        return "Unknown"

def rgb_to_color_name(rgb_color):
    """Maps an RGB color to a simple color name."""
    # Simplified color mapping
    r, g, b = int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])
    
    # Check for grayscale
    if abs(r-g) < 20 and abs(r-b) < 20 and abs(g-b) < 20:
        if r > 200: return "White"
        if r < 50: return "Black"
        return "Gray"

    if r > 150 and r > g and r > b: return "Red"
    if g > 150 and g > r and g > b: return "Green"
    if b > 150 and b > r and b > g: return "Blue"
    if r > 200 and g > 200 and b < 100: return "Yellow"
    if r > 200 and g < 100 and b > 200: return "Pink"
    if r < 100 and g > 150 and b > 150: return "Cyan"
    if r > 120 and g > 50 and b < 50: return "Brown" # (Approx)
    
    return "Mixed" # Default


# --- Streamlit UI ---
st.title("👨‍👩‍👧‍👦 Nationality, Emotion, and Feature Detector")
st.write("Upload an image of a person to predict their features based on the project requirements.")

if nat_model is None:
    st.stop()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    image_cv = np.array(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # --- Face Detection ---
    image_gray = cv2.cvtColor(image_cv, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    if len(faces) == 0:
        st.error("No face detected. Please try a different image.")
    else:
        # Use the first (and likely largest) face found
        (x, y, w, h) = faces[0]
        face_box = (x, y, w, h)
        cropped_face_rgb = image_cv[y:y+h, x:x+w]
        
        # --- Run Predictions ---
        try:
            # 1. Preprocess for each model type
            face_nat_age = preprocess_for_nat_age(cropped_face_rgb)
            face_emotion = preprocess_for_emotion(cropped_face_rgb)
            
            # 2. Run core models (Nationality & Emotion)
            nat_pred_vec = nat_model.predict(face_nat_age)
            nat_index = np.argmax(nat_pred_vec)
            nationality = NATIONALITY_LABELS.get(nat_index, "Unknown")
            
            emo_pred_vec = emo_model.predict(face_emotion)
            emo_index = np.argmax(emo_pred_vec)
            emotion = EMOTION_LABELS.get(emo_index, "Unknown")
            
            # 3. Initialize results and run conditional models
            results = {
                "Nationality": nationality,
                "Emotion": emotion
            }
            
            if nationality == 'Indian':
                age_pred = age_model.predict(face_nat_age)[0][0]
                color = get_dominant_color(image_cv, face_box)
                results["Age (Approx)"] = f"{int(age_pred)} years"
                results["Dress Colour"] = color
            
            elif nationality == 'United States':
                age_pred = age_model.predict(face_nat_age)[0][0]
                results["Age (Approx)"] = f"{int(age_pred)} years"

            elif nationality == 'African':
                color = get_dominant_color(image_cv, face_box)
                results["Dress Colour"] = color
            
            # 4. Display results
            with col2:
                st.subheader("Analysis Results")
                st.json(results)
                st.success("Analysis complete!")
                
                # Draw bounding box on the image
                img_with_box = image_cv.copy()
                cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
                st.image(img_with_box, caption='Detected Face', use_container_width=True)

        except Exception as e:
            st.error(f"An error occurred during model prediction: {e}")