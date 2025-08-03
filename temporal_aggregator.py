import numpy as np
from collections import deque
from keras.models import model_from_json, Model

class TemporalAggregator:
    """
    Collects CNN features over time windows for engagement estimation
    Following the base paper's methodology
    """
    
    def __init__(self, window_size=30):
        self.window_size = window_size  # ~1 second at 30 FPS
        self.features_buffer = deque(maxlen=window_size)
        
    def add_frame_features(self, features):
        """Add 512-dim features from one frame"""
        # Flatten in case features come as (1, 512)
        if features.ndim > 1:
            features = features.flatten()
        self.features_buffer.append(features)
        
    def is_ready_for_engagement_prediction(self):
        """Check if we have enough frames for stable engagement estimation"""
        return len(self.features_buffer) >= self.window_size
    
    def get_engagement_features(self):
        """
        Create aggregated feature vector for engagement regression
        Returns: 1024-dim vector (512 mean + 512 std)
        """
        if not self.is_ready_for_engagement_prediction():
            return None
            
        # Convert deque to numpy array
        features_array = np.array(list(self.features_buffer))
        
        # Statistical pooling (as in base paper)
        mean_features = np.mean(features_array, axis=0)  # 512 dims
        std_features = np.std(features_array, axis=0)    # 512 dims
        
        # Concatenate mean and std
        engagement_features = np.concatenate([mean_features, std_features])  # 1024 dims
        
        return engagement_features
    
    def get_buffer_status(self):
        """Get current buffer information"""
        return {
            'current_frames': len(self.features_buffer),
            'required_frames': self.window_size,
            'ready': self.is_ready_for_engagement_prediction()
        }

# Test the temporal aggregator
if __name__ == "__main__":
    print("Testing Temporal Aggregator...")
    
    # Load your feature extractor
    with open("emotion_model.json", "r") as f:
        model_json = f.read()
    
    emotion_model = model_from_json(model_json)
    emotion_model.load_weights("emotion_model.h5")
    
    feature_extractor = Model(
        inputs=emotion_model.input,
        outputs=emotion_model.layers[10].output
    )
    
    # Create aggregator
    aggregator = TemporalAggregator(window_size=30)
    
    # Simulate adding features from 35 frames
    print("\nSimulating feature collection over time:")
    for frame_num in range(35):
        # Simulate random face input
        dummy_face = np.random.random((1, 48, 48, 1))
        features = feature_extractor.predict(dummy_face, verbose=0)
        
        # Add to aggregator
        aggregator.add_frame_features(features)
        
        status = aggregator.get_buffer_status()
        print(f"Frame {frame_num+1:2d}: Buffer {status['current_frames']:2d}/{status['required_frames']} - Ready: {status['ready']}")
        
        # Try to get engagement features
        if status['ready']:
            eng_features = aggregator.get_engagement_features()
            print(f"    → Engagement features shape: {eng_features.shape}")
            print(f"    → Sample values: {eng_features[:5]}")
            break
