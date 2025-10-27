# %% [markdown]
# # 2. Emotion Recognition Model (Training Project)
# 
# **Goal:** Train a model to classify a facial image into one of 7 emotions.
# - 0: Angry
# - 1: Disgust
# - 2: Fear
# - 3: Happy
# - 4: Sad
# - 5: Surprise
# 
# **Dataset:** [FER2013 (Facial Expression Recognition)](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge)
# 
# **Methodology:** We will build a custom Convolutional Neural Network (CNN) tailored for this specific task. The input images are 48x48 grayscale.

# %% [code]
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, BatchNormalization, Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# %% [markdown]
# ## Step 1: Load and Preprocess Data
# 
# The FER2013 dataset is a CSV file. We need to parse the 'pixels' string into a 48x48 image array.

# %% [code]
# Define constants
IMG_ROWS, IMG_COLS = 48, 48
NUM_CLASSES = 7
# Try a couple of likely dataset locations (project-relative)
POSSIBLE_PATHS = [
    os.path.join('data', 'fer2013.csv'),
    os.path.join('..', 'data', 'fer2013.csv')
]

DATA_PATH = None
for p in POSSIBLE_PATHS:
    if os.path.exists(p):
        DATA_PATH = p
        break

if DATA_PATH is None:
    raise FileNotFoundError(
        "Dataset 'fer2013.csv' not found. Place it in 'data/' or '../data/' relative to the project root."
    )

data = pd.read_csv(DATA_PATH)
print(f"Loaded {len(data)} total samples from {DATA_PATH}.")
print(data.head())

# %% [code]
def preprocess_data(data):
    """
    Parses the FER2013 dataset.
    Returns X (images) and y (labels).
    """
    images = []
    labels = []
    
    for i in range(len(data)):
        # Parse pixel string
        pixels = [int(p) for p in data['pixels'].iloc[i].split(' ')]
        image = np.array(pixels, dtype='float32').reshape(IMG_ROWS, IMG_COLS, 1) # (48, 48, 1)
        
        images.append(image)
        labels.append(data['emotion'].iloc[i])
        
    # Convert to numpy arrays
    X = np.array(images)
    y = np.array(labels)
    
    # Normalize images
    X = X / 255.0
    
    # One-hot encode labels
    y = to_categorical(y, NUM_CLASSES)
    
    return X, y

# Split data based on the 'Usage' column
print("Preprocessing data...")
df_train = data[data['Usage'] == 'Training'].reset_index(drop=True)
df_val = data[data['Usage'] == 'PrivateTest'].reset_index(drop=True)
df_test = data[data['Usage'] == 'PublicTest'].reset_index(drop=True)

X_train, y_train = preprocess_data(df_train)
X_val, y_val = preprocess_data(df_val)
X_test, y_test = preprocess_data(df_test)

print(f"Training samples: {X_train.shape[0]}")
print(f"Validation samples: {X_val.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")

# %% [markdown]
# ## Step 2: Visualize Data & Define Labels

# %% [code]
EMOTION_LABELS = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}

plt.figure(figsize=(12, 6))
n_samples = min(10, X_train.shape[0])
for i in range(n_samples):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_train[i].reshape(IMG_ROWS, IMG_COLS), cmap='gray')
    plt.title(EMOTION_LABELS[int(np.argmax(y_train[i]))])
    plt.axis('off')
plt.suptitle('Sample Training Images')
plt.show()


# %% [code]
# --- NEW: Calculate Class Weights to address Imbalance ---
# This is crucial as the 'Disgust' class is severely under-represented
print("Calculating class weights...")
y_integers = np.argmax(y_train, axis=1)
computed_class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(y_integers),
    y=y_integers
)
class_weights_dict = dict(enumerate(computed_class_weights))
print("Class Weights:", class_weights_dict)

# %% [markdown]
# ## Step 3: Build the CNN Model
# 
# We'll use a standard "VGG-style" block architecture: Conv-Conv-Pool, with Batch Normalization and Dropout for regularization.

# %% [code]
def build_model(input_shape=(IMG_ROWS, IMG_COLS, 1)):
    model = Sequential()
    
    # Block 1
    model.add(Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    # Block 2
    model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    
    # Block 3
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flatten and Dense
    model.add(Flatten())
    model.add(Dense(512, kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    # Output Layer
    model.add(Dense(NUM_CLASSES, kernel_initializer='he_normal'))
    model.add(Activation('softmax'))
    
    # Compile
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

model = build_model()
model.summary()

# %% [markdown]
# ## Step 4: Data Augmentation
# 
# To prevent overfitting, we'll apply real-time data augmentation to the training images.

# %% [code]
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# No augmentation for validation/test data
val_datagen = ImageDataGenerator()

# %% [markdown]
# ## Step 5: Train the Model

# %% [code]
# Define callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]

# Define training parameters
BATCH_SIZE = 64
EPOCHS = 150
train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=len(X_train) // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(X_val, y_val), # Use un-augmented validation data
    callbacks=callbacks,
    class_weight=class_weights_dict
)

# %% [markdown]
# ## Step 6: Evaluate the Model

# %% [code]
# Plot training history
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True)
plt.show()

pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True)
plt.show()

# %% [code]
# Evaluate on the TEST set
print("\n--- Evaluating on Test Set ---")
score = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {score[0]:.4f}")
print(f"Test Accuracy: {score[1]:.4f}")

# %% [code]
# Get predictions
y_pred_vec = model.predict(X_test)
y_pred = np.argmax(y_pred_vec, axis=1)
y_true = np.argmax(y_test, axis=1) # Convert from one-hot

class_names = list(EMOTION_LABELS.values())

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (Test Set)')
plt.ylabel('True Label') 
plt.xlabel('Predicted Label')
plt.show()

# %% [markdown]
# ## Step 7: Save the Model
# 
# We save the best model for use in the Streamlit app.

# %% [code]
model.save('./models/emotion_model.h5')
print("Model saved to ./models/emotion_model.h5")