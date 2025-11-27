"""
Configuration settings for AQI Monitoring System
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Data directory
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Database settings
DATABASE_PATH = DATA_DIR / "aqi_data.db"

# API settings
AQICN_API_KEY = os.getenv("AQICN_API_KEY", "")
AQICN_BASE_URL = "https://api.waqi.info"

# Default city for AQI data
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "delhi")

# Scheduler settings (in seconds)
DATA_FETCH_INTERVAL = int(os.getenv("DATA_FETCH_INTERVAL", "300"))  # 5 minutes

# Model settings
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
MODEL_METRICS_PATH = MODELS_DIR / "model_metrics.json"

# AQI Categories
AQI_CATEGORIES = {
    (0, 50): "Good",
    (51, 100): "Moderate",
    (101, 150): "Unhealthy for Sensitive Groups",
    (151, 200): "Unhealthy",
    (201, 300): "Very Unhealthy",
    (301, 500): "Hazardous"
}

# Feature columns for ML models
FEATURE_COLUMNS = ["pm25", "pm10", "co", "no2", "so2", "o3", "temperature", "humidity"]

# Use mock data if no API key is provided
USE_MOCK_DATA = not bool(AQICN_API_KEY)
