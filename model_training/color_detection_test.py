# %% [markdown]
# # 4. Dress Color Detection (Heuristic)
# 
# **Goal:** Develop a function to find the dominant color in the "clothing region" of an image.
# 
# **Methodology:**
# 1.  Use OpenCV's Haar Cascade to find the face.
# 2.  Define a "clothing region" just below the face's bounding box.
# 3.  Extract all pixels from this region.
# 4.  Use `sklearn.cluster.KMeans` to find the `k=3` dominant color clusters.
# 5.  Identify the center of the largest cluster as the dominant color.
# 6.  Map this RGB value to a simple color name.

# %% [code]
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image

# %% [code]
# --- Helper Functions (Copy-pasted from app.py) ---

def rgb_to_color_name(rgb_color):
    """Maps an RGB color to a simple color name."""
    r, g, b = int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2])
    
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
    if r > 120 and g > 50 and b < 50: return "Brown"
    
    return "Mixed"

def get_dominant_color(image, face_box):
    """Finds the dominant color in the clothing region."""
    try:
        (x, y, w, h) = face_box
        
        # Define clothing region
        ch = int(h * 1.0)
        cw = int(w * 1.5)
        cx_start = max(0, x - int((cw - w) / 2))
        cy_start = y + h
        
        clothing_region = image[cy_start:min(cy_start + ch, image.shape[0]), 
                                cx_start:min(cx_start + cw, image.shape[1])]

        if clothing_region.size < 50:
            return "Unknown", None, None

        # Reshape for K-Means
        pixels = clothing_region.reshape(-1, 3)
        
        # Run K-Means
        kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
        kmeans.fit(pixels)
        
        # Get the most dominant cluster
        unique, counts = np.unique(kmeans.labels_, return_counts=True)
        dominant_cluster_center = kmeans.cluster_centers_[unique[counts.argmax()]]
        
        color_name = rgb_to_color_name(dominant_cluster_center)
        
        return color_name, clothing_region, dominant_cluster_center.astype(int)
        
    except Exception as e:
        print(f"Color detection error: {e}")
        return "Unknown", None, None

# %% [code]
# --- Test the Function ---

# 1. Load Face Detector
face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

# 2. Load a test image (You must provide your own test image)
# Download a test image (e.g., from UTKFace) and place it here
TEST_IMAGE_PATH = 'data/UTKFace/32_0_0_20170117203115358.jpg.chip.jpg' # Example path
try:
    image_bgr = cv2.imread(TEST_IMAGE_PATH)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
except:
    print(f"Error: Could not load test image at {TEST_IMAGE_PATH}")
    print("Please download a test image and update the path.")

# 3. Detect Face
image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
faces = face_cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

if len(faces) > 0:
    face_box = faces[0]
    (x, y, w, h) = face_box
    
    # 4. Run Function
    color_name, region, rgb_val = get_dominant_color(image_rgb, face_box)
    
    print(f"Dominant Color Name: {color_name}")
    print(f"Dominant RGB Value: {rgb_val}")
    
    # 5. Visualize Results
    plt.figure(figsize=(15, 6))
    
    # Original Image with Face Box
    ax1 = plt.subplot(1, 3, 1)
    img_with_box = image_rgb.copy()
    cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 3)
    ax1.imshow(img_with_box)
    ax1.set_title('1. Face Detection')
    ax1.axis('off')
    
    # Clothing Region
    ax2 = plt.subplot(1, 3, 2)
    if region is not None:
        ax2.imshow(region)
    ax2.set_title('2. Isolated Clothing Region')
    ax2.axis('off')
    
    # Dominant Color
    ax3 = plt.subplot(1, 3, 3)
    if rgb_val is not None:
        color_swatch = np.zeros((100, 100, 3), dtype=np.uint8)
        color_swatch[:] = rgb_val
        ax3.imshow(color_swatch)
    ax3.set_title(f"3. Dominant Color: {color_name}")
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

else:
    print("No face detected in the test image.")