"""
Medication event logger with enhanced error handling.
"""
import json
import aiofiles
from datetime import datetime, timezone
from typing import Dict, List, Optional
from pathlib import Path
from uuid import uuid4
from dateutil.relativedelta import relativedelta
from ..config import MEDICATION_LOGS_DIR
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class MedicationLogger:
    """Logs medication events to monthly JSON files."""
    
    def __init__(self, log_dir: Path = MEDICATION_LOGS_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_monthly_filename(self, date: datetime) -> Path:
        """Get the filename for a given month's medication log."""
        return self.log_dir / f"{date.strftime('%Y-%m')}.json"
    
    async def _load_medications(self, filename: Path) -> List[Dict]:
        """Load medications from a file."""
        if not filename.exists():
            return []
        
        try:
            async with aiofiles.open(filename, "r") as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.error(f"Error loading medications from {filename}: {e}")
            return []
    
    async def _save_medications(self, filename: Path, medications: List[Dict]) -> bool:
        """Save medications to a file."""
        try:
            async with aiofiles.open(filename, "w") as f:
                await f.write(json.dumps(medications, indent=4))
            return True
        except Exception as e:
            logger.error(f"Error saving medications to {filename}: {e}")
            return False
    
    async def create(self, event_data: Dict[str, str]) -> Optional[Dict]:
        """Create a new medication event."""
        event_data["id"] = str(uuid4())
        start_time = datetime.fromisoformat(event_data["date"].replace("Z", "+00:00"))
        
        filename = self._get_monthly_filename(start_time)
        medications = await self._load_medications(filename)
        medications.append(event_data)
        
        if await self._save_medications(filename, medications):
            logger.info(f"Medication created: {event_data['id']}, name: {event_data.get('name')}")
            return event_data
        return None
    
    async def update(self, event_id: str, updates: Dict[str, str]) -> Optional[Dict]:
        """Update a medication event."""
        for filename in sorted(self.log_dir.glob("*.json")):
            medications = await self._load_medications(filename)
            
            for medication in medications:
                if medication["id"] == event_id:
                    for key, value in updates.items():
                        if key in medication:
                            medication[key] = value
                    
                    if await self._save_medications(filename, medications):
                        logger.info(f"Medication updated: {event_id}")
                        return medication
        
        logger.warning(f"Medication event not found for update: {event_id}")
        return None
    
    async def delete(self, event_id: str) -> bool:
        """Delete a medication event."""
        for filename in sorted(self.log_dir.glob("*.json")):
            medications = await self._load_medications(filename)
            
            for i, medication in enumerate(medications):
                if medication["id"] == event_id:
                    del medications[i]
                    if await self._save_medications(filename, medications):
                        logger.info(f"Medication deleted: {event_id}")
                        return True
        
        logger.warning(f"Medication event not found for deletion: {event_id}")
        return False
    
    async def get_range(self, start_dt: datetime, end_dt: datetime) -> List[Dict]:
        """Get all medication events in a date range."""
        medications = []
        current_dt = datetime(start_dt.year, start_dt.month, 1, tzinfo=timezone.utc)
        
        while current_dt <= end_dt:
            filename = self._get_monthly_filename(current_dt)
            if filename.exists():
                month_medications = await self._load_medications(filename)
                for med in month_medications:
                    event_date = datetime.fromisoformat(med["date"].replace("Z", "+00:00"))
                    if start_dt <= event_date <= end_dt:
                        medications.append(med)
            
            current_dt += relativedelta(months=1)
        
        logger.info(f"Retrieved {len(medications)} medication events")
        return medications
