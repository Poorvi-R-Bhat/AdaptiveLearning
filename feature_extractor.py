from keras.models import model_from_json, Model
import numpy as np
import cv2

# Load your existing model
with open("emotion_model.json", "r") as f:
    model_json = f.read()

emotion_model = model_from_json(model_json)
emotion_model.load_weights("emotion_model.h5")

# Create feature extractor (outputs Layer 10: dense layer with 512 features)
feature_extractor = Model(
    inputs=emotion_model.input,
    outputs=emotion_model.layers[10].output  # The dense layer before final classification
)

print("Feature extractor created successfully!")
print(f"Feature extractor output shape: {feature_extractor.output_shape}")

# Test with a dummy input
test_input = np.random.random((1, 48, 48, 1))  # Same shape as your training data
test_features = feature_extractor.predict(test_input)
print(f"Test feature vector shape: {test_features.shape}")
print(f"Sample features: {test_features[0][:10]}")  # First 10 values
