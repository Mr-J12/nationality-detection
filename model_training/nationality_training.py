# %% [markdown]
# # 1. Nationality Classification Model
# 
# **Goal:** Train a model to classify faces into one of four categories:
# - Indian
# - United States
# - African
# - Other
# 
# **Dataset:** [UTKFace](https://susanqq.github.io/UTKFace/)
# 
# **Methodology:** We will parse the filenames of the UTKFace dataset to extract the 'Race' label. We will then map these 'Race' labels to our target 'Nationality' labels. A `MobileNetV2` model pre-trained on ImageNet will be fine-tuned for this classification task.

# %% [code]
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# %% [markdown]
# ## Step 1: Load and Parse Data
# 
# We assume the UTKFace dataset has been downloaded and unzipped into a folder named `../data/UTKFace`.
# 
# The filenames are in the format: `[age]_[gender]_[race]_[date].jpg`

# %% [code]
DATA_DIR = 'data/UTKFace'
image_files = os.listdir(DATA_DIR)

# Parse filenames
data = []
for f in image_files:
    if f.endswith('.jpg'):
        parts = f.split('_')
        if len(parts) == 4:
            try:
                age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
                data.append({
                    'filename': os.path.join(DATA_DIR, f),
                    'age': age,
                    'gender': gender,
                    'race': race
                })
            except ValueError:
                pass  # Skip improperly named files

df = pd.DataFrame(data)
print(f"Loaded {len(df)} images.")
df.head()

# %% [markdown]
# ## Step 2: Map 'Race' to 'Nationality'
# 
# - `0: 'White'` -> `'United States'`
# - `1: 'Black'` -> `'African'`
# - `2: 'Asian'` -> `'Other'`
# - `3: 'Indian'` -> `'Indian'`
# - `4: 'Others'` -> `'Other'`

# %% [code]
def map_nationality(race_code):
    if race_code == 0:
        return 'United States'
    elif race_code == 1:
        return 'African'
    elif race_code == 3:
        return 'Indian'
    else:  # (2 and 4)
        return 'Other'

df['nationality'] = df['race'].apply(map_nationality)

# Check class distribution
print("Class Distribution:")
print(df['nationality'].value_counts())

# Plot distribution
plt.figure(figsize=(10, 5))
sns.countplot(x='nationality', data=df, order=df['nationality'].value_counts().index)
plt.title('Nationality Class Distribution')
plt.show()

# %% [markdown]
# ## Step 3: Split Data and Create Generators
# 
# We will split the data into training (80%) and validation (20%).
# We use `ImageDataGenerator` for preprocessing and augmentation.

# %% [code]
# Split data
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['nationality'], random_state=42)

print(f"Training samples: {len(train_df)}")
print(f"Validation samples: {len(val_df)}")

# Create Image Data Generators
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Use MobileNetV2 preprocessing
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='nationality',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filename',
    y_col='nationality',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Keep order for evaluation
)

# Save the class indices
class_labels = train_generator.class_indices
print("Class Indices:", class_labels)

# %% [markdown]
# ## Step 4: Build the Model (Transfer Learning)
# 
# We'll use a pre-trained `MobileNetV2` and add a custom classification head.

# %% [code]
# Load base model
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model
base_model.trainable = False

# Add custom head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(class_labels), activation='softmax')(x) # 4 classes

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# %% [markdown]
# ## Step 5: Train the Model

# %% [code]
# Add callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=15,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[early_stopping]  
)

# %% [markdown]
# ## Step 6: Evaluate the Model

# %% [code]
# Plot training history
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

# %% [code]
# Get predictions
val_generator.reset()
y_pred_vec = model.predict(val_generator, steps=len(val_generator))
y_pred = np.argmax(y_pred_vec, axis=1)
y_true = val_generator.classes

class_names = list(class_labels.keys())

# Classification Report
print("\n--- Classification Report ---")
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# %% [markdown]
# ## Step 7: Save the Model
# 
# We save the best model for use in the Streamlit app.

# %% [code]
model.save('models/nationality_model.h5')
print("Model saved to models/nationality_model.h5")