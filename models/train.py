"""
Model Training Script - Train and evaluate multiple ML models for AQI prediction
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from config import BEST_MODEL_PATH, MODEL_METRICS_PATH, MODELS_DIR, FEATURE_COLUMNS
from backend.services.data_fetcher import AQIDataFetcher
from backend.services.preprocessor import DatabaseManager, DataPreprocessor, get_aqi_category_index


def filter_minority_classes(X, y, min_samples: int = 2):
    """
    Filter out classes with fewer than min_samples to allow stratified splitting.
    
    Args:
        X: Feature DataFrame or array
        y: Target Series or array
        min_samples: Minimum number of samples required per class
    
    Returns:
        Filtered X, y with only classes having enough samples
    """
    # Convert to pandas Series if numpy array
    if not isinstance(y, pd.Series):
        y = pd.Series(y)
    
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts >= min_samples].index.tolist()
    removed_classes = class_counts[class_counts < min_samples].index.tolist()
    
    if removed_classes:
        print(f"\nWarning: Removing classes with fewer than {min_samples} samples:")
        for cls in removed_classes:
            print(f"  - Class {cls}: {class_counts[cls]} sample(s)")
    
    mask = y.isin(valid_classes)
    
    # Handle both DataFrame and numpy array for X
    if hasattr(X, 'loc'):
        X_filtered = X[mask]
    else:
        X_filtered = X[mask.values]
    
    y_filtered = y[mask]
    
    print(f"\nFiltered dataset: {len(X_filtered)} samples ({len(X) - len(X_filtered)} removed)")
    print(f"Remaining classes: {sorted(valid_classes)}")
    
    return X_filtered, y_filtered


def get_models() -> dict:
    """Get dictionary of models to train"""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100, 
            max_depth=5, 
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='mlogloss'
        )
    }


def train_and_evaluate(X_train, X_test, y_train, y_test, scaler=None) -> dict:
    """Train all models and return their metrics"""
    models = get_models()
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Scale features if scaler provided
        if scaler:
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train
            X_test_scaled = X_test
        
        # Train
        model.fit(X_train_scaled, y_train)
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            "model": model,
            "scaler": scaler,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": conf_matrix.tolist()
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
    
    return results


def select_best_model(results: dict) -> tuple:
    """Select the best model based on F1 score"""
    best_name = None
    best_score = -1
    
    for name, metrics in results.items():
        if metrics["f1_score"] > best_score:
            best_score = metrics["f1_score"]
            best_name = name
    
    return best_name, results[best_name]


def save_confusion_matrices(results: dict, save_path: Path):
    """Save confusion matrix visualizations"""
    n_models = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    category_labels = ["Good", "Moderate", "USG", "Unhealthy", "V.Unhealthy", "Hazardous"]
    
    for idx, (name, metrics) in enumerate(results.items()):
        if idx < len(axes):
            cm = np.array(metrics["confusion_matrix"])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=category_labels[:cm.shape[1]], 
                       yticklabels=category_labels[:cm.shape[0]])
            axes[idx].set_title(f'{name}\nF1: {metrics["f1_score"]:.3f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
    
    # Hide empty subplots
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved to {save_path}")


def train_models(force_new_data: bool = False, days: int = 30, csv_path: str = None) -> dict:
    """Main training pipeline"""
    print("="*60)
    print("AQI Model Training Pipeline")
    print("="*60)
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    db = DatabaseManager()
    preprocessor = DataPreprocessor()
    fetcher = AQIDataFetcher()
    
    if csv_path:
        print(f"\nLoading data from CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} records from CSV")
        
        # Map CSV columns to expected column names
        column_mapping = {
            'pm2_5': 'pm25',
            'AQI_Label': 'category_index'
        }
        df = df.rename(columns=column_mapping)
    else:
        # Check if we need to generate data
        existing_count = db.get_readings_count()
        print(f"\nExisting records in database: {existing_count}")
        
        min_records_needed = 500
        if existing_count < min_records_needed or force_new_data:
            print(f"\nGenerating historical data for training...")
            historical_data = fetcher.get_historical_data(days=days)
            
            # Add city field to historical data
            for record in historical_data:
                record["city"] = "delhi"
            
            saved = db.save_bulk_readings(historical_data)
            print(f"Saved {saved} new records to database")
        
        # Load data from database
        df = db.get_readings(limit=10000)
        print(f"\nLoaded {len(df)} records for training")
        
        if len(df) < 100:
            print("Not enough data for training. Generating more...")
            historical_data = fetcher.get_historical_data(days=30)
            for record in historical_data:
                record["city"] = "delhi"
            db.save_bulk_readings(historical_data)
            df = db.get_readings(limit=10000)
    
    # Preprocess data
    print("\nPreprocessing data...")
    df_clean = preprocessor.clean_data(df)
    X, y = preprocessor.prepare_features(df_clean)
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts().sort_index()}")
    
    # Filter minority classes to allow stratified splitting
    X_filtered, y_filtered = filter_minority_classes(X, y, min_samples=2)
    
    if len(X_filtered) < 10:
        raise ValueError("Not enough data after filtering minority classes. Need at least 10 samples.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered
    )
    
    print(f"\nTraining set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    # Train and evaluate models
    scaler = StandardScaler()
    results = train_and_evaluate(X_train, X_test, y_train, y_test, scaler)
    
    # Select best model
    best_name, best_result = select_best_model(results)
    print(f"\n{'='*60}")
    print(f"Best Model: {best_name}")
    print(f"F1 Score: {best_result['f1_score']:.4f}")
    print(f"{'='*60}")
    
    # Save best model
    print(f"\nSaving best model to {BEST_MODEL_PATH}...")
    joblib.dump(best_result["model"], BEST_MODEL_PATH)
    
    # Save scaler
    scaler_path = MODELS_DIR / "scaler.joblib"
    if best_result["scaler"]:
        joblib.dump(best_result["scaler"], scaler_path)
    
    # Save metrics
    metrics_summary = {
        "best_model": best_name,
        "best_model_metrics": {
            "accuracy": best_result["accuracy"],
            "precision": best_result["precision"],
            "recall": best_result["recall"],
            "f1_score": best_result["f1_score"]
        },
        "all_models": {
            name: {
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1_score": m["f1_score"],
                "confusion_matrix": m["confusion_matrix"]
            }
            for name, m in results.items()
        }
    }
    
    with open(MODEL_METRICS_PATH, 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"Metrics saved to {MODEL_METRICS_PATH}")
    
    # Save confusion matrix visualization
    cm_path = MODELS_DIR / "confusion_matrices.png"
    save_confusion_matrices(results, cm_path)
    
    # Save model to database
    for name, metrics in results.items():
        db.save_model_metrics(name, {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"]
        })
    
    print("\nTraining complete!")
    return metrics_summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train AQI prediction models")
    parser.add_argument("--force", action="store_true", help="Force regenerate training data")
    parser.add_argument("--days", type=int, default=30, help="Days of historical data to generate")
    parser.add_argument("--csv", type=str, help="Path to CSV dataset file")
    
    args = parser.parse_args()
    
    train_models(force_new_data=args.force, days=args.days, csv_path=args.csv)
