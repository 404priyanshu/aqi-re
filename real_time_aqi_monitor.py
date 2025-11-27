"""
Main Entry Point - Real-Time AQI Monitoring & Prediction System
"""
import sys
import os
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def train_models():
    """Train ML models"""
    print("Training ML models...")
    from models.train import train_models as run_training
    run_training(force_new_data=False, days=30)


def evaluate_models():
    """Evaluate trained models"""
    print("Evaluating models...")
    from models.evaluate import evaluate_models as run_evaluation
    run_evaluation()


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server"""
    import uvicorn
    print(f"Starting AQI Monitoring Server on {host}:{port}...")
    print(f"Dashboard: http://localhost:{port}")
    print(f"API Docs: http://localhost:{port}/docs")
    
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=False
    )


def run_dev_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server in development mode with auto-reload"""
    import uvicorn
    print(f"Starting AQI Monitoring Server (DEV MODE) on {host}:{port}...")
    print(f"Dashboard: http://localhost:{port}")
    print(f"API Docs: http://localhost:{port}/docs")
    
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=True
    )


def fetch_data():
    """Fetch current AQI data"""
    print("Fetching current AQI data...")
    from backend.services.data_fetcher import AQIDataFetcher
    from backend.services.preprocessor import DatabaseManager
    
    fetcher = AQIDataFetcher()
    db = DatabaseManager()
    
    data = fetcher.fetch_live_data_sync()
    if data:
        parsed = fetcher.parse_aqi_response(data)
        if parsed:
            db.save_reading(parsed)
            print(f"AQI: {parsed['aqi']} - City: {parsed['city']}")
            print(f"PM2.5: {parsed['pm25']}, PM10: {parsed['pm10']}")
            print(f"CO: {parsed['co']}, NO2: {parsed['no2']}")
            print(f"SO2: {parsed['so2']}, O3: {parsed['o3']}")
        else:
            print("Failed to parse AQI data")
    else:
        print("Failed to fetch AQI data")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Real-Time AQI Monitoring & Prediction System"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run the web server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    server_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    server_parser.add_argument("--dev", action="store_true", help="Run in development mode")
    
    # Train command
    subparsers.add_parser("train", help="Train ML models")
    
    # Evaluate command
    subparsers.add_parser("evaluate", help="Evaluate trained models")
    
    # Fetch command
    subparsers.add_parser("fetch", help="Fetch current AQI data")
    
    args = parser.parse_args()
    
    if args.command == "server":
        if args.dev:
            run_dev_server(args.host, args.port)
        else:
            run_server(args.host, args.port)
    elif args.command == "train":
        train_models()
    elif args.command == "evaluate":
        evaluate_models()
    elif args.command == "fetch":
        fetch_data()
    else:
        # Default: show help
        parser.print_help()
        print("\nQuick Start:")
        print("  1. Train models:  python real_time_aqi_monitor.py train")
        print("  2. Run server:    python real_time_aqi_monitor.py server")
        print("  3. Open browser:  http://localhost:8000")


if __name__ == "__main__":
    main()
