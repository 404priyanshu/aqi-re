"""
API Routes - All API endpoints for the AQI Monitoring System
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Optional
import asyncio

from backend.services.data_fetcher import AQIDataFetcher
from backend.services.preprocessor import DatabaseManager, get_aqi_category
from backend.services.predictor import get_predictor, AQIPredictor
from models.evaluate import get_model_summary

# Create router
router = APIRouter()

# Initialize services
fetcher = AQIDataFetcher()
db = DatabaseManager()


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint"""
    pm25: float = 0
    pm10: float = 0
    co: float = 0
    no2: float = 0
    so2: float = 0
    o3: float = 0
    temperature: float = 25
    humidity: float = 50


class RetrainResponse(BaseModel):
    """Response for retrain endpoint"""
    status: str
    message: str


@router.get("/aqi/live")
async def get_live_aqi(city: str = "delhi"):
    """
    Get live AQI data for a city
    
    Args:
        city: City name (default: delhi)
    
    Returns:
        Current AQI data including all pollutant levels
    """
    try:
        raw_data = await fetcher.fetch_live_data(city)
        
        if not raw_data:
            raise HTTPException(status_code=503, detail="Unable to fetch AQI data")
        
        parsed_data = fetcher.parse_aqi_response(raw_data)
        
        if not parsed_data:
            raise HTTPException(status_code=503, detail="Error parsing AQI data")
        
        # Add category information
        aqi = parsed_data.get("aqi", 0)
        parsed_data["category"] = get_aqi_category(aqi)
        parsed_data["category_color"] = AQIPredictor.CATEGORY_COLORS.get(
            parsed_data["category"], "#808080"
        )
        
        # Save to database
        db.save_reading(parsed_data)
        
        return {
            "status": "success",
            "data": parsed_data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/aqi/predict")
async def get_prediction(
    pm25: float = 50,
    pm10: float = 80,
    co: float = 2.0,
    no2: float = 30,
    so2: float = 15,
    o3: float = 50,
    temperature: float = 25,
    humidity: float = 50
):
    """
    Get predicted AQI category based on pollutant levels
    
    Args:
        pm25: PM2.5 level
        pm10: PM10 level
        co: Carbon Monoxide level
        no2: Nitrogen Dioxide level
        so2: Sulfur Dioxide level
        o3: Ozone level
        temperature: Temperature in Celsius
        humidity: Humidity percentage
    
    Returns:
        Predicted AQI category and color code
    """
    predictor = get_predictor()
    
    if not predictor.is_model_available():
        # If no model, calculate category from PM2.5 directly
        aqi_estimate = int(pm25)
        category = get_aqi_category(aqi_estimate)
        return {
            "status": "success",
            "prediction": {
                "category": category,
                "category_index": None,
                "color": AQIPredictor.CATEGORY_COLORS.get(category, "#808080"),
                "confidence": None,
                "note": "Using direct PM2.5 estimation (no trained model available)"
            }
        }
    
    features = {
        "pm25": pm25,
        "pm10": pm10,
        "co": co,
        "no2": no2,
        "so2": so2,
        "o3": o3,
        "temperature": temperature,
        "humidity": humidity
    }
    
    result = predictor.predict(features)
    probabilities = predictor.predict_proba(features)
    
    if result is None:
        raise HTTPException(status_code=500, detail="Prediction failed")
    
    category, category_index, color = result
    
    return {
        "status": "success",
        "prediction": {
            "category": category,
            "category_index": category_index,
            "color": color,
            "confidence": max(probabilities.values()) if probabilities else None,
            "probabilities": probabilities
        },
        "input_features": features
    }


@router.post("/aqi/predict")
async def post_prediction(request: PredictionRequest):
    """
    Get predicted AQI category (POST version)
    
    Accepts JSON body with pollutant levels
    """
    return await get_prediction(
        pm25=request.pm25,
        pm10=request.pm10,
        co=request.co,
        no2=request.no2,
        so2=request.so2,
        o3=request.o3,
        temperature=request.temperature,
        humidity=request.humidity
    )


def run_training():
    """Run model training in background"""
    from models.train import train_models
    train_models(force_new_data=False)


@router.post("/model/retrain", response_model=RetrainResponse)
async def retrain_model(background_tasks: BackgroundTasks):
    """
    Trigger model retraining
    
    This endpoint triggers a background task to retrain all models
    and select the best one.
    """
    try:
        background_tasks.add_task(run_training)
        return RetrainResponse(
            status="accepted",
            message="Model retraining has been initiated. Check /api/model/status for updates."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/status")
async def get_model_status():
    """
    Get current model status and metrics
    
    Returns:
        Model information including accuracy metrics
    """
    predictor = get_predictor()
    model_info = predictor.get_model_info()
    model_summary = get_model_summary()
    
    return {
        "status": "success",
        "model_loaded": predictor.is_model_available(),
        "model_info": model_info,
        "comparison": model_summary
    }


@router.get("/aqi/history")
async def get_history(limit: int = 100, city: Optional[str] = None):
    """
    Get historical AQI readings
    
    Args:
        limit: Maximum number of records to return
        city: Optional city filter
    
    Returns:
        List of historical AQI readings
    """
    try:
        df = db.get_readings(limit=limit, city=city)
        records = df.to_dict(orient='records')
        
        return {
            "status": "success",
            "count": len(records),
            "data": records
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/metrics")
async def get_metrics():
    """
    Get detailed model comparison metrics
    
    Returns:
        Metrics for all trained models
    """
    summary = get_model_summary()
    return {
        "status": "success",
        "metrics": summary
    }
