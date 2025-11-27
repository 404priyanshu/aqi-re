"""
Predictor Service - Loads trained model and makes predictions
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import json
from typing import Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import BEST_MODEL_PATH, MODEL_METRICS_PATH, FEATURE_COLUMNS, AQI_CATEGORIES


class AQIPredictor:
    """Handles AQI prediction using trained models"""
    
    CATEGORY_LABELS = [
        "Good",
        "Moderate",
        "Unhealthy for Sensitive Groups",
        "Unhealthy",
        "Very Unhealthy",
        "Hazardous"
    ]
    
    CATEGORY_COLORS = {
        "Good": "#00e400",
        "Moderate": "#ffff00",
        "Unhealthy for Sensitive Groups": "#ff7e00",
        "Unhealthy": "#ff0000",
        "Very Unhealthy": "#8f3f97",
        "Hazardous": "#7e0023"
    }
    
    def __init__(self, model_path: Path = BEST_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.model_info = None
        self._load_model()
    
    def _load_model(self) -> bool:
        """Load the trained model from disk"""
        try:
            if self.model_path.exists():
                self.model = joblib.load(self.model_path)
                
                # Load model metrics if available
                if MODEL_METRICS_PATH.exists():
                    with open(MODEL_METRICS_PATH, 'r') as f:
                        self.model_info = json.load(f)
                
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        
        return False
    
    def is_model_available(self) -> bool:
        """Check if model is loaded and ready"""
        return self.model is not None
    
    def predict(self, features: dict) -> Optional[Tuple[str, int, str]]:
        """
        Predict AQI category from features
        
        Returns:
            Tuple of (category_label, category_index, color_code) or None if error
        """
        if not self.is_model_available():
            return None
        
        try:
            # Prepare feature vector
            feature_vector = []
            for col in FEATURE_COLUMNS:
                feature_vector.append(features.get(col, 0))
            
            X = np.array([feature_vector])
            prediction = self.model.predict(X)[0]
            
            category_label = self.CATEGORY_LABELS[prediction]
            color_code = self.CATEGORY_COLORS.get(category_label, "#808080")
            
            return (category_label, int(prediction), color_code)
        except Exception as e:
            print(f"Error making prediction: {e}")
            return None
    
    def predict_proba(self, features: dict) -> Optional[dict]:
        """Get prediction probabilities for all categories"""
        if not self.is_model_available():
            return None
        
        try:
            # Prepare feature vector
            feature_vector = []
            for col in FEATURE_COLUMNS:
                feature_vector.append(features.get(col, 0))
            
            X = np.array([feature_vector])
            
            if hasattr(self.model, 'predict_proba'):
                probas = self.model.predict_proba(X)[0]
                return {
                    self.CATEGORY_LABELS[i]: float(prob) 
                    for i, prob in enumerate(probas) 
                    if i < len(self.CATEGORY_LABELS)
                }
        except Exception as e:
            print(f"Error getting prediction probabilities: {e}")
        
        return None
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if self.model_info:
            return self.model_info
        
        if self.model:
            return {
                "model_type": type(self.model).__name__,
                "status": "loaded"
            }
        
        return {"status": "no_model_loaded"}
    
    def reload_model(self) -> bool:
        """Reload the model from disk"""
        return self._load_model()


# Global predictor instance
_predictor = None

def get_predictor() -> AQIPredictor:
    """Get or create the global predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = AQIPredictor()
    return _predictor


# For testing
if __name__ == "__main__":
    predictor = AQIPredictor()
    
    if predictor.is_model_available():
        test_features = {
            "pm25": 75.0,
            "pm10": 120.0,
            "co": 2.5,
            "no2": 45.0,
            "so2": 20.0,
            "o3": 60.0,
            "temperature": 25.0,
            "humidity": 65.0
        }
        
        result = predictor.predict(test_features)
        print(f"Prediction: {result}")
        
        probas = predictor.predict_proba(test_features)
        print(f"Probabilities: {probas}")
    else:
        print("No model available. Please train a model first.")
