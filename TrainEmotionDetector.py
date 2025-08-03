import cv2
import os
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Prevent OpenCL errors from OpenCV
cv2.ocl.setUseOpenCL(False)

# Suppress TensorFlow unnecessary logs for demo
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Paths
train_dir = 'data/train'
val_dir = 'data/test'

# Check if folders exist
if not os.path.exists(train_dir) or not os.path.exists(val_dir):
    raise Exception("Training or validation folder not found! Check if 'data/train' and 'data/test' exist.")

# Image preprocessing
train_data_gen = ImageDataGenerator(rescale=1./255)
val_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=32,   # Smaller batch size to reduce memory load
    color_mode="grayscale",
    class_mode='categorical'
)

val_generator = val_data_gen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    batch_size=32,
    color_mode="grayscale",
    class_mode='categorical'
)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    Flatten(),
    Dense(512, activation='relu'),  # Reduced for faster training
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001, decay=1e-6),
    metrics=['accuracy']
)

# Train the model
print("Starting training...")
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=50,  #keeping 50 for now
    validation_data=val_generator,
    validation_steps=len(val_generator)
)
print("Training complete!")

# Save model
model.save_weights('emotion_model.h5')
with open("emotion_model.json", "w") as f:
    f.write(model.to_json())
print("Model saved!")

# Plot Accuracy
plt.figure(figsize=(8, 4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_plot.png')
plt.show()

# Plot Loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_plot.png')
plt.show()
