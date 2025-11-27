"""
Model Evaluation Script - Evaluate and compare trained models
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from config import MODEL_METRICS_PATH, MODELS_DIR


def load_metrics() -> dict:
    """Load saved model metrics"""
    if not MODEL_METRICS_PATH.exists():
        print("No metrics file found. Please train models first.")
        return {}
    
    with open(MODEL_METRICS_PATH, 'r') as f:
        return json.load(f)


def print_comparison_table(metrics: dict):
    """Print a formatted comparison table of all models"""
    if not metrics or "all_models" not in metrics:
        print("No model metrics available")
        return
    
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"{'Model':<25} {'Accuracy':>12} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("-"*80)
    
    for name, m in metrics["all_models"].items():
        print(f"{name:<25} {m['accuracy']:>12.4f} {m['precision']:>12.4f} {m['recall']:>12.4f} {m['f1_score']:>12.4f}")
    
    print("-"*80)
    best = metrics.get("best_model", "Unknown")
    print(f"\nBest Model: {best}")
    print("="*80)


def plot_metrics_comparison(metrics: dict, save_path: Path = None):
    """Create a bar chart comparing model metrics"""
    if not metrics or "all_models" not in metrics:
        print("No model metrics available for plotting")
        return
    
    models = list(metrics["all_models"].keys())
    
    accuracy = [metrics["all_models"][m]["accuracy"] for m in models]
    precision = [metrics["all_models"][m]["precision"] for m in models]
    recall = [metrics["all_models"][m]["recall"] for m in models]
    f1 = [metrics["all_models"][m]["f1_score"] for m in models]
    
    x = np.arange(len(models))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', color='#2ecc71')
    bars2 = ax.bar(x - 0.5*width, precision, width, label='Precision', color='#3498db')
    bars3 = ax.bar(x + 0.5*width, recall, width, label='Recall', color='#e74c3c')
    bars4 = ax.bar(x + 1.5*width, f1, width, label='F1-Score', color='#9b59b6')
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    add_labels(bars4)  # Only add labels for F1 to avoid clutter
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison chart saved to {save_path}")
    
    plt.close()


def get_model_summary() -> dict:
    """Get a summary of model performance for API response"""
    metrics = load_metrics()
    
    if not metrics:
        return {"status": "no_models_trained"}
    
    summary = {
        "best_model": metrics.get("best_model"),
        "best_model_metrics": metrics.get("best_model_metrics"),
        "models": {}
    }
    
    for name, m in metrics.get("all_models", {}).items():
        summary["models"][name] = {
            "accuracy": round(m["accuracy"], 4),
            "precision": round(m["precision"], 4),
            "recall": round(m["recall"], 4),
            "f1_score": round(m["f1_score"], 4)
        }
    
    return summary


def evaluate_models():
    """Main evaluation function"""
    metrics = load_metrics()
    
    if not metrics:
        print("No trained models found. Run train.py first.")
        return
    
    print_comparison_table(metrics)
    
    # Generate comparison chart
    chart_path = MODELS_DIR / "metrics_comparison.png"
    plot_metrics_comparison(metrics, chart_path)
    
    return metrics


if __name__ == "__main__":
    evaluate_models()
