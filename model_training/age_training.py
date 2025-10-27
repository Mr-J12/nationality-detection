# %% [markdown]
# # 3. Age Regression Model
# 
# **Goal:** Train a model to predict the approximate age of a person.
# 
# **Dataset:** [UTKFace](https://susanqq.github.io/UTKFace/)
# 
# **Methodology:** We will use the same parsed UTKFace data, but this time, the target label will be the 'age' column. This is a **regression** task. We will again use a fine-tuned `MobileNetV2` but with a different regression head.

# %% [code]
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from keras.optimizers import Adam

# %% [markdown]
# ## Step 1: Load and Parse Data
# 
# We can re-use the parsing logic from the nationality notebook.

# %% [code]
DATA_DIR = './data/UTKFace'
image_files = os.listdir(DATA_DIR)

# Parse filenames
data = []
for f in image_files:
    if f.endswith('.jpg'):
        parts = f.split('_')
        if len(parts) == 4:
            try:
                age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
                # Filter out ages > 100 for more stable training
                if 1 < age <= 100:
                    data.append({
                        'filename': os.path.join(DATA_DIR, f),
                        'age': age
                    })
            except ValueError:
                pass  # Skip improperly named files

df = pd.DataFrame(data)
print(f"Loaded {len(df)} images.")
df.head()

# %% [markdown]
# ## Step 2: Split Data and Create Generators
# 
# For regression, `class_mode='raw'` is used.

# %% [code]
# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

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
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

# Note: class_mode='raw' for regression
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='filename',
    y_col='age',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw' 
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='filename',
    y_col='age',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw',
    shuffle=False
)

# %% [markdown]
# ## Step 3: Build the Model (Regression Head)
# 
# The final layer will have 1 neuron and a 'linear' activation.

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
x = Dropout(0.3)(x)
predictions = Dense(1, activation='linear')(x) # Single output neuron for age

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
# Use Mean Absolute Error (MAE) as the loss and metric
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mean_absolute_error',
    metrics=['mae']
)

model.summary()

# %% [markdown]
# ## Step 4: Train the Model

# %% [code]
# Add callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,
    validation_data=val_generator,
    validation_steps=len(val_generator),
    callbacks=[early_stopping]
)

# %% [markdown]
# ## Step 5: Evaluate the Model

# %% [code]
# Plot training history
pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
plt.title('Model MAE Loss')
plt.ylabel('Mean Absolute Error (Age)')
plt.xlabel('Epoch')
plt.show()

# %% [code]
# Get predictions on validation set
val_generator.reset()
y_pred = model.predict(val_generator, steps=len(val_generator)).flatten() # Flatten to 1D array
y_true = val_generator.labels

# Report metrics
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"\n--- Model Performance ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} years")
print(f"R-squared (R²): {r2:.2f}")

# Scatter Plot of True vs. Predicted Age
plt.figure(figsize=(8, 8))
plt.scatter(y_true, y_pred, alpha=0.3)
plt.plot([0, 100], [0, 100], color='red', linestyle='--') # y=x line
plt.title('True Age vs. Predicted Age')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid(True)
plt.show()

# %% [markdown]
# ## Step 6: Save the Model

# %% [code]
model.save('./models/age_model.h5')
print("Model saved to ./models/age_model.h5")