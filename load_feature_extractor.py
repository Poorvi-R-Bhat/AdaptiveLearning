from keras.models import model_from_json, Model

# Load model architecture
with open("emotion_model.json", "r") as f:
    model_json = f.read()

emotion_model = model_from_json(model_json)
emotion_model.load_weights("emotion_model.h5")

print("Model loaded successfully!")

# Print layer info
print("\n Available layers in my CNN model:")
for i, layer in enumerate(emotion_model.layers):
    print(f"Layer {i}: {layer.name}, Output shape: {layer.output_shape}")
