"""
Data Preprocessor - Handles data cleaning, scaling, and feature engineering
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional
import sqlite3
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import DATABASE_PATH, FEATURE_COLUMNS, AQI_CATEGORIES, DATA_DIR


def get_aqi_category(aqi: int) -> str:
    """Convert AQI value to category label"""
    for (low, high), category in AQI_CATEGORIES.items():
        if low <= aqi <= high:
            return category
    return "Hazardous" if aqi > 500 else "Unknown"


def get_aqi_category_index(aqi: int) -> int:
    """Convert AQI value to category index for ML models"""
    if aqi <= 50:
        return 0  # Good
    elif aqi <= 100:
        return 1  # Moderate
    elif aqi <= 150:
        return 2  # Unhealthy for Sensitive Groups
    elif aqi <= 200:
        return 3  # Unhealthy
    elif aqi <= 300:
        return 4  # Very Unhealthy
    else:
        return 5  # Hazardous


class DatabaseManager:
    """Manages SQLite database operations"""
    
    def __init__(self, db_path: Path = DATABASE_PATH):
        self.db_path = db_path
        self._ensure_data_dir()
        self._init_database()
    
    def _ensure_data_dir(self):
        """Ensure data directory exists"""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def _init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create AQI readings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS aqi_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                city TEXT NOT NULL,
                aqi INTEGER,
                pm25 REAL,
                pm10 REAL,
                co REAL,
                no2 REAL,
                so2 REAL,
                o3 REAL,
                temperature REAL,
                humidity REAL,
                category TEXT,
                category_index INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create model metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                trained_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_reading(self, data: dict) -> bool:
        """Save a single AQI reading to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            aqi = data.get("aqi", 0)
            category = get_aqi_category(aqi)
            category_index = get_aqi_category_index(aqi)
            
            cursor.execute("""
                INSERT INTO aqi_readings 
                (timestamp, city, aqi, pm25, pm10, co, no2, so2, o3, temperature, humidity, category, category_index)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data.get("timestamp", datetime.utcnow().isoformat()),
                data.get("city", "Unknown"),
                aqi,
                data.get("pm25", 0),
                data.get("pm10", 0),
                data.get("co", 0),
                data.get("no2", 0),
                data.get("so2", 0),
                data.get("o3", 0),
                data.get("temperature", 0),
                data.get("humidity", 0),
                category,
                category_index
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving reading: {e}")
            return False
    
    def save_bulk_readings(self, data_list: list) -> int:
        """Save multiple AQI readings to database"""
        count = 0
        for data in data_list:
            if self.save_reading(data):
                count += 1
        return count
    
    def get_readings(self, limit: int = 1000, city: Optional[str] = None) -> pd.DataFrame:
        """Get AQI readings from database"""
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM aqi_readings"
        params = []
        
        if city:
            query += " WHERE city = ?"
            params.append(city)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    
    def get_readings_count(self) -> int:
        """Get total count of readings in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM aqi_readings")
        count = cursor.fetchone()[0]
        conn.close()
        return count
    
    def save_model_metrics(self, model_name: str, metrics: dict) -> bool:
        """Save model training metrics"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO model_metrics (model_name, accuracy, precision_score, recall, f1_score)
                VALUES (?, ?, ?, ?, ?)
            """, (
                model_name,
                metrics.get("accuracy", 0),
                metrics.get("precision", 0),
                metrics.get("recall", 0),
                metrics.get("f1_score", 0)
            ))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error saving model metrics: {e}")
            return False


class DataPreprocessor:
    """Handles data preprocessing for ML models"""
    
    def __init__(self):
        self.feature_columns = FEATURE_COLUMNS
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data"""
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Fill missing values with median
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Remove outliers using IQR method for numeric columns
        for col in self.feature_columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Prepare features and target for ML models"""
        # Ensure all feature columns exist
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        
        X = df[self.feature_columns].copy()
        
        # Get target if available
        y = None
        if "category_index" in df.columns:
            y = df["category_index"]
        elif "aqi" in df.columns:
            y = df["aqi"].apply(get_aqi_category_index)
        
        return X, y
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add engineered features"""
        df = df.copy()
        
        # Add time-based features if timestamp exists
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["hour"] = df["timestamp"].dt.hour
            df["day_of_week"] = df["timestamp"].dt.dayofweek
            df["month"] = df["timestamp"].dt.month
            df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        
        # Add ratio features
        if "pm25" in df.columns and "pm10" in df.columns:
            df["pm_ratio"] = df["pm25"] / (df["pm10"] + 1)
        
        return df


# For testing
if __name__ == "__main__":
    db = DatabaseManager()
    preprocessor = DataPreprocessor()
    
    # Test with mock data
    from backend.services.data_fetcher import AQIDataFetcher
    
    fetcher = AQIDataFetcher()
    historical_data = fetcher.get_historical_data(days=7)
    
    # Save to database
    saved_count = db.save_bulk_readings(historical_data)
    print(f"Saved {saved_count} records to database")
    
    # Retrieve and preprocess
    df = db.get_readings()
    print(f"Retrieved {len(df)} records from database")
    
    df_clean = preprocessor.clean_data(df)
    X, y = preprocessor.prepare_features(df_clean)
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
