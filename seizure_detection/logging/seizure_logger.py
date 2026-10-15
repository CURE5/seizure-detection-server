"""
Seizure event logger with enhanced error handling.
"""
import json
import aiofiles
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
from uuid import uuid4
from dateutil.relativedelta import relativedelta
from ..config import SEIZURE_LOGS_DIR
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class SeizureLogger:
    """Logs seizure events to monthly JSON files."""
    
    def __init__(self, log_dir: Path = SEIZURE_LOGS_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_monthly_filename(self, date: datetime) -> Path:
        """Get the filename for a given month's seizure log."""
        return self.log_dir / f"{date.strftime('%Y-%m')}.json"
    
    async def _load_seizures(self, filename: Path) -> List[Dict]:
        """Load seizures from a file."""
        if not filename.exists():
            return []
        
        try:
            async with aiofiles.open(filename, "r") as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.error(f"Error loading seizures from {filename}: {e}")
            return []
    
    async def _save_seizures(self, filename: Path, seizures: List[Dict]) -> bool:
        """Save seizures to a file."""
        try:
            async with aiofiles.open(filename, "w") as f:
                await f.write(json.dumps(seizures, indent=4))
            return True
        except Exception as e:
            logger.error(f"Error saving seizures to {filename}: {e}")
            return False
    
    async def start(self, seizure_type: str, intensity: str, notes: str = "", video: str = "") -> str:
        """
        Log the start of a seizure event.
        
        Args:
            seizure_type: Type of seizure
            intensity: Intensity level
            notes: Additional notes
            video: Associated video filename
            
        Returns:
            Event ID
        """
        start_time = datetime.now(timezone.utc)
        event_id = str(uuid4())
        
        event = {
            "id": event_id,
            "date": start_time.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "type": seizure_type,
            "intensity": intensity,
            "notes": notes,
            "video": video,
            "duration": "",
            "durationSeconds": 0,
        }
        
        filename = self._get_monthly_filename(start_time)
        seizures = await self._load_seizures(filename)
        seizures.append(event)
        
        await self._save_seizures(filename, seizures)
        logger.info(f"Seizure started: {event_id}, type: {seizure_type}, intensity: {intensity}")
        
        return event_id
    
    async def end(self, event_id: str) -> Optional[Dict]:
        """
        Mark a seizure event as ended.
        
        Args:
            event_id: ID of the seizure event
            
        Returns:
            Updated event, or None if not found
        """
        end_time = datetime.now(timezone.utc)
        filename = self._get_monthly_filename(end_time)
        seizures = await self._load_seizures(filename)
        
        for seizure in seizures:
            if seizure["id"] == event_id:
                start_time = datetime.fromisoformat(seizure["date"].replace("Z", "+00:00"))
                duration = end_time - start_time
                
                seizure["durationSeconds"] = int(duration.total_seconds())
                seizure["duration"] = str(timedelta(seconds=seizure["durationSeconds"]))
                
                await self._save_seizures(filename, seizures)
                logger.info(f"Seizure ended: {event_id}, duration: {seizure['duration']}")
                
                return seizure
        
        logger.warning(f"Seizure event not found: {event_id}")
        return None
    
    async def create(self, event_data: Dict[str, str]) -> Optional[Dict]:
        """Create a new seizure event."""
        event_data["id"] = str(uuid4())
        start_time = datetime.fromisoformat(event_data["date"].replace("Z", "+00:00"))
        
        filename = self._get_monthly_filename(start_time)
        seizures = await self._load_seizures(filename)
        seizures.append(event_data)
        
        if await self._save_seizures(filename, seizures):
            logger.info(f"Seizure created: {event_data['id']}")
            return event_data
        return None
    
    async def update(self, event_id: str, updates: Dict[str, Any]) -> Optional[Dict]:
        """Update a seizure event."""
        # Search all monthly files
        for filename in sorted(self.log_dir.glob("*.json")):
            seizures = await self._load_seizures(filename)
            
            for seizure in seizures:
                if seizure["id"] == event_id:
                    for key, value in updates.items():
                        if key in seizure:
                            seizure[key] = value
                    
                    if await self._save_seizures(filename, seizures):
                        logger.info(f"Seizure updated: {event_id}")
                        return seizure
        
        logger.warning(f"Seizure event not found for update: {event_id}")
        return None
    
    async def delete(self, event_id: str) -> bool:
        """Delete a seizure event."""
        for filename in sorted(self.log_dir.glob("*.json")):
            seizures = await self._load_seizures(filename)
            
            for i, seizure in enumerate(seizures):
                if seizure["id"] == event_id:
                    del seizures[i]
                    if await self._save_seizures(filename, seizures):
                        logger.info(f"Seizure deleted: {event_id}")
                        return True
        
        logger.warning(f"Seizure event not found for deletion: {event_id}")
        return False
    
    async def get_by_id(self, event_id: str) -> Optional[Dict]:
        """Find a seizure event by ID. Searches all monthly files."""
        for filename in sorted(self.log_dir.glob("*.json")):
            seizures = await self._load_seizures(filename)
            for seizure in seizures:
                if seizure.get("id") == event_id:
                    return seizure
        return None

    async def get_range(self, start_dt: datetime, end_dt: datetime) -> List[Dict]:
        """Get all seizure events in a date range."""
        seizures = []
        current_dt = datetime(start_dt.year, start_dt.month, 1, tzinfo=timezone.utc)
        
        while current_dt <= end_dt:
            filename = self._get_monthly_filename(current_dt)
            if filename.exists():
                month_seizures = await self._load_seizures(filename)
                for sz in month_seizures:
                    event_date = datetime.fromisoformat(sz["date"].replace("Z", "+00:00"))
                    if start_dt <= event_date <= end_dt:
                        seizures.append(sz)
            
            current_dt += relativedelta(months=1)
        
        logger.info(f"Retrieved {len(seizures)} seizure events")
        return seizures
