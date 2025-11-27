"""
Background Scheduler - Periodic data updates using APScheduler
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import asyncio

from config import DATA_FETCH_INTERVAL, DEFAULT_CITY
from backend.services.data_fetcher import AQIDataFetcher
from backend.services.preprocessor import DatabaseManager

# Global scheduler instance
scheduler = None


async def fetch_and_store_aqi():
    """Fetch current AQI data and store in database"""
    try:
        fetcher = AQIDataFetcher()
        db = DatabaseManager()
        
        raw_data = await fetcher.fetch_live_data(DEFAULT_CITY)
        
        if raw_data:
            parsed_data = fetcher.parse_aqi_response(raw_data)
            if parsed_data:
                db.save_reading(parsed_data)
                print(f"[Scheduler] Stored AQI reading: {parsed_data.get('aqi')} for {DEFAULT_CITY}")
    except Exception as e:
        print(f"[Scheduler] Error fetching AQI data: {e}")


def start_scheduler():
    """Start the background scheduler"""
    global scheduler
    
    if scheduler is not None:
        return
    
    scheduler = AsyncIOScheduler()
    
    # Add periodic job for fetching AQI data
    scheduler.add_job(
        fetch_and_store_aqi,
        trigger=IntervalTrigger(seconds=DATA_FETCH_INTERVAL),
        id="fetch_aqi_data",
        name="Fetch AQI Data",
        replace_existing=True
    )
    
    # Run immediately on startup
    scheduler.add_job(
        fetch_and_store_aqi,
        id="initial_fetch",
        name="Initial AQI Fetch"
    )
    
    scheduler.start()
    print(f"[Scheduler] Started. Fetching AQI data every {DATA_FETCH_INTERVAL} seconds")


def stop_scheduler():
    """Stop the background scheduler"""
    global scheduler
    
    if scheduler is not None:
        scheduler.shutdown()
        scheduler = None
        print("[Scheduler] Stopped")


def get_scheduler_status() -> dict:
    """Get current scheduler status"""
    global scheduler
    
    if scheduler is None:
        return {"running": False}
    
    jobs = []
    for job in scheduler.get_jobs():
        jobs.append({
            "id": job.id,
            "name": job.name,
            "next_run": str(job.next_run_time) if job.next_run_time else None
        })
    
    return {
        "running": scheduler.running,
        "jobs": jobs
    }


# For testing
if __name__ == "__main__":
    import time
    
    print("Testing scheduler...")
    start_scheduler()
    
    # Run for a short time
    try:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(asyncio.sleep(10))
    except KeyboardInterrupt:
        pass
    finally:
        stop_scheduler()
