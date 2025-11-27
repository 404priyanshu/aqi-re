# Real-Time AQI Monitoring & Prediction System

A comprehensive Air Quality Index (AQI) monitoring and prediction system built with Python, FastAPI, and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🌟 Features

- **Real-time AQI Monitoring**: Fetches live air quality data from AQICN API or uses mock data for testing
- **ML-based Prediction**: Predicts AQI categories using trained machine learning models
- **Multiple ML Models**: Trains and compares 5 different models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - XGBoost
- **Interactive Dashboard**: Real-time visualization with Chart.js
- **Background Scheduler**: Automatic periodic data updates using APScheduler
- **SQLite Database**: Lightweight, portable data storage
- **Docker Support**: Easy deployment with Docker and docker-compose
- **RESTful API**: Well-documented FastAPI endpoints

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (HTML/CSS/JS)                    │
│                     ┌─────────────────────────┐                 │
│                     │   Dashboard (Chart.js)   │                 │
│                     └─────────────────────────┘                 │
└─────────────────────────────┬───────────────────────────────────┘
                              │ HTTP/REST
┌─────────────────────────────▼───────────────────────────────────┐
│                      FastAPI Backend                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  API Routes  │  │  Scheduler   │  │    ML Predictor      │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                       Services Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │ Data Fetcher │  │ Preprocessor │  │  Database Manager    │   │
│  └──────────────┘  └──────────────┘  └──────────────────────┘   │
└─────────────────────────────┬───────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  ┌──────────────────────┐     ┌────────────────────────────┐    │
│  │   SQLite Database    │     │   ML Models (joblib)       │    │
│  │   (aqi_data.db)      │     │   (best_model.joblib)      │    │
│  └──────────────────────┘     └────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
aqi-re/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # API endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── data_fetcher.py  # Data fetching & mock generation
│   │   ├── preprocessor.py  # Data cleaning & DB management
│   │   └── predictor.py     # ML prediction service
│   └── scheduler.py         # Background task scheduler
├── frontend/
│   ├── index.html           # Dashboard HTML
│   ├── css/
│   │   └── styles.css       # Dashboard styles
│   └── js/
│       └── app.js           # Dashboard JavaScript
├── models/
│   ├── __init__.py
│   ├── train.py             # Model training script
│   └── evaluate.py          # Model evaluation
├── data/
│   └── .gitkeep             # Data directory
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
├── requirements.txt         # Python dependencies
├── config.py                # Configuration settings
├── real_time_aqi_monitor.py # Main entry point
└── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- (Optional) Docker and Docker Compose

### Local Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/aqi-re.git
   cd aqi-re
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Train the ML models**
   ```bash
   python real_time_aqi_monitor.py train
   ```

5. **Run the server**
   ```bash
   python real_time_aqi_monitor.py server
   ```

6. **Open the dashboard**
   
   Navigate to [http://localhost:8000](http://localhost:8000)

### Docker Installation

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up -d
   ```

2. **Access the dashboard**
   
   Navigate to [http://localhost:8000](http://localhost:8000)

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `AQICN_API_KEY` | API key for AQICN (optional, uses mock data if not set) | "" |
| `DEFAULT_CITY` | Default city for AQI monitoring | "delhi" |
| `DATA_FETCH_INTERVAL` | Data fetch interval in seconds | 300 |

To use real AQI data, get a free API key from [AQICN](https://aqicn.org/data-platform/token/) and set it:

```bash
export AQICN_API_KEY="your-api-key"
```

## 📡 API Documentation

Once the server is running, access the interactive API documentation at:
- Swagger UI: [http://localhost:8000/docs](http://localhost:8000/docs)
- ReDoc: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/aqi/live` | Get current AQI data |
| GET | `/api/aqi/predict` | Get ML prediction for AQI category |
| POST | `/api/aqi/predict` | Get prediction with JSON body |
| GET | `/api/aqi/history` | Get historical AQI readings |
| POST | `/api/model/retrain` | Trigger model retraining |
| GET | `/api/model/status` | Get model status and info |
| GET | `/api/model/metrics` | Get model comparison metrics |
| GET | `/health` | Health check endpoint |

### Example API Calls

**Get Live AQI:**
```bash
curl http://localhost:8000/api/aqi/live?city=delhi
```

**Get Prediction:**
```bash
curl "http://localhost:8000/api/aqi/predict?pm25=75&pm10=120&co=2.5&no2=45"
```

**Trigger Retraining:**
```bash
curl -X POST http://localhost:8000/api/model/retrain
```

## 🧠 Machine Learning Models

The system trains and compares five different classification models:

1. **Logistic Regression** - Linear model for baseline
2. **Decision Tree** - Non-linear decision boundaries
3. **Random Forest** - Ensemble of decision trees
4. **Gradient Boosting** - Sequential ensemble method
5. **XGBoost** - Optimized gradient boosting

### Evaluation Metrics

Models are evaluated using:
- **Accuracy**: Overall correct predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

The best model (by F1-score) is automatically selected and saved.

### AQI Categories

| AQI Range | Category | Color |
|-----------|----------|-------|
| 0-50 | Good | Green |
| 51-100 | Moderate | Yellow |
| 101-150 | Unhealthy for Sensitive Groups | Orange |
| 151-200 | Unhealthy | Red |
| 201-300 | Very Unhealthy | Purple |
| 301-500 | Hazardous | Maroon |

## 📊 Dashboard Screenshots

*Dashboard with real-time AQI display and pollutant levels*

![Dashboard Placeholder](docs/dashboard-placeholder.png)

*Model performance comparison chart*

![Models Placeholder](docs/models-placeholder.png)

## 🛠️ Development

### Running in Development Mode

```bash
python real_time_aqi_monitor.py server --dev
```

This enables auto-reload for code changes.

### Running Tests

```bash
# Run model training with test data
python real_time_aqi_monitor.py train

# Evaluate models
python real_time_aqi_monitor.py evaluate
```

### CLI Commands

```bash
# Show help
python real_time_aqi_monitor.py --help

# Train models
python real_time_aqi_monitor.py train

# Evaluate models
python real_time_aqi_monitor.py evaluate

# Fetch current data
python real_time_aqi_monitor.py fetch

# Run server
python real_time_aqi_monitor.py server --host 0.0.0.0 --port 8000

# Run in dev mode
python real_time_aqi_monitor.py server --dev
```

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

Built with ❤️ using Python, FastAPI, and Machine Learning