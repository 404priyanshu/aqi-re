"""
Data Fetcher Service - Fetches real-time AQI data from APIs or generates mock data
"""
import asyncio
import random
import datetime
from typing import Optional
import httpx
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config import AQICN_API_KEY, AQICN_BASE_URL, DEFAULT_CITY, USE_MOCK_DATA


class MockDataGenerator:
    """Generates realistic mock AQI data for testing"""
    
    @staticmethod
    def generate_aqi_data(city: str = DEFAULT_CITY) -> dict:
        """Generate mock AQI data with realistic values"""
        base_aqi = random.randint(20, 250)
        
        # Generate correlated pollutant values
        pm25 = base_aqi * random.uniform(0.8, 1.2)
        pm10 = pm25 * random.uniform(1.2, 1.8)
        co = random.uniform(0.5, 10.0)
        no2 = random.uniform(10, 80)
        so2 = random.uniform(5, 50)
        o3 = random.uniform(20, 100)
        
        return {
            "status": "ok",
            "data": {
                "aqi": base_aqi,
                "city": {
                    "name": city.title(),
                    "geo": [28.6139, 77.2090] if city.lower() == "delhi" else [0, 0]
                },
                "dominentpol": "pm25",
                "iaqi": {
                    "pm25": {"v": round(pm25, 1)},
                    "pm10": {"v": round(pm10, 1)},
                    "co": {"v": round(co, 2)},
                    "no2": {"v": round(no2, 1)},
                    "so2": {"v": round(so2, 1)},
                    "o3": {"v": round(o3, 1)},
                    "t": {"v": round(random.uniform(15, 40), 1)},
                    "h": {"v": round(random.uniform(30, 90), 1)},
                    "p": {"v": round(random.uniform(990, 1020), 1)},
                    "w": {"v": round(random.uniform(1, 15), 1)}
                },
                "time": {
                    "iso": datetime.datetime.utcnow().isoformat() + "Z",
                    "tz": "+05:30"
                }
            }
        }
    
    @staticmethod
    def generate_historical_data(days: int = 30, samples_per_day: int = 24) -> list:
        """Generate historical mock data for training models"""
        data = []
        start_date = datetime.datetime.utcnow() - datetime.timedelta(days=days)
        
        for day in range(days):
            for hour in range(samples_per_day):
                timestamp = start_date + datetime.timedelta(days=day, hours=hour)
                
                # Add some seasonality and randomness
                base_aqi = 50 + 30 * abs(hour - 12) / 12  # Higher during morning/evening
                base_aqi += random.gauss(0, 20)  # Random noise
                base_aqi = max(10, min(500, base_aqi))  # Clamp values
                
                pm25 = base_aqi * random.uniform(0.8, 1.2)
                pm10 = pm25 * random.uniform(1.2, 1.8)
                
                record = {
                    "timestamp": timestamp.isoformat(),
                    "aqi": round(base_aqi),
                    "pm25": round(pm25, 1),
                    "pm10": round(pm10, 1),
                    "co": round(random.uniform(0.5, 10.0), 2),
                    "no2": round(random.uniform(10, 80), 1),
                    "so2": round(random.uniform(5, 50), 1),
                    "o3": round(random.uniform(20, 100), 1),
                    "temperature": round(random.uniform(15, 40), 1),
                    "humidity": round(random.uniform(30, 90), 1)
                }
                data.append(record)
        
        return data


class AQIDataFetcher:
    """Fetches AQI data from AQICN API or mock generator"""
    
    def __init__(self, api_key: str = AQICN_API_KEY, use_mock: bool = USE_MOCK_DATA):
        self.api_key = api_key
        self.use_mock = use_mock
        self.base_url = AQICN_BASE_URL
        self.mock_generator = MockDataGenerator()
    
    async def fetch_live_data(self, city: str = DEFAULT_CITY) -> Optional[dict]:
        """Fetch live AQI data for a city"""
        if self.use_mock:
            return self.mock_generator.generate_aqi_data(city)
        
        try:
            async with httpx.AsyncClient() as client:
                url = f"{self.base_url}/feed/{city}/?token={self.api_key}"
                response = await client.get(url, timeout=10.0)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            print(f"Error fetching live data: {e}")
            # Fallback to mock data
            return self.mock_generator.generate_aqi_data(city)
    
    def fetch_live_data_sync(self, city: str = DEFAULT_CITY) -> Optional[dict]:
        """Synchronous version of fetch_live_data"""
        return asyncio.run(self.fetch_live_data(city))
    
    def get_historical_data(self, days: int = 30) -> list:
        """Get historical data for model training"""
        return self.mock_generator.generate_historical_data(days)
    
    def parse_aqi_response(self, response: dict) -> Optional[dict]:
        """Parse and extract relevant data from API response"""
        if response.get("status") != "ok" or "data" not in response:
            return None
        
        data = response["data"]
        iaqi = data.get("iaqi", {})
        
        return {
            "aqi": data.get("aqi", 0),
            "city": data.get("city", {}).get("name", "Unknown"),
            "dominant_pollutant": data.get("dominentpol", "unknown"),
            "pm25": iaqi.get("pm25", {}).get("v", 0),
            "pm10": iaqi.get("pm10", {}).get("v", 0),
            "co": iaqi.get("co", {}).get("v", 0),
            "no2": iaqi.get("no2", {}).get("v", 0),
            "so2": iaqi.get("so2", {}).get("v", 0),
            "o3": iaqi.get("o3", {}).get("v", 0),
            "temperature": iaqi.get("t", {}).get("v", 0),
            "humidity": iaqi.get("h", {}).get("v", 0),
            "timestamp": data.get("time", {}).get("iso", datetime.datetime.utcnow().isoformat())
        }


# For testing
if __name__ == "__main__":
    fetcher = AQIDataFetcher()
    data = fetcher.fetch_live_data_sync()
    print("Live data:", data)
    
    historical = fetcher.get_historical_data(days=7)
    print(f"Generated {len(historical)} historical records")
