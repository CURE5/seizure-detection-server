"""
Enhanced data logger with proper error handling.
"""
import json
import aiofiles
from datetime import datetime, timedelta
from typing import List, Dict
from pathlib import Path
from ..config import DATA_LOGS_DIR
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DataLogger:
    """Logs system data to JSON files organized by date and hour."""
    
    def __init__(self, log_dir: Path = DATA_LOGS_DIR):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.last_saved_data: Dict = {}
        
    def _get_log_path(self, timestamp: datetime) -> Path:
        """Get the log file path for a given timestamp."""
        date_dir = self.log_dir / timestamp.strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        return date_dir / f"{timestamp.strftime('%H')}.json"
    
    async def save(self, data: Dict) -> bool:
        """
        Save data to log file if values have changed.
        
        Args:
            data: Data dictionary to save
            
        Returns:
            True if saved successfully
        """
        # Skip if data hasn't changed
        if data == self.last_saved_data:
            return True
        
        try:
            now = datetime.now()
            log_file = self._get_log_path(now)
            
            # Create entry with timestamp
            entry = {
                "timestamp": now.strftime("%Y-%m-%dT%H:%M:%S"),
                "data": data
            }
            
            # Load existing logs
            logs = []
            if log_file.exists():
                try:
                    async with aiofiles.open(log_file, "r") as f:
                        content = await f.read()
                        logs = json.loads(content)
                except json.JSONDecodeError:
                    logger.error(f"Corrupted log file: {log_file}")
                    # Backup corrupted file
                    backup_file = log_file.with_suffix('.json.bak')
                    log_file.rename(backup_file)
                    logs = []
            
            # Append new entry
            logs.append(entry)
            
            # Save with compact format
            async with aiofiles.open(log_file, "w") as f:
                await f.write(json.dumps(logs, separators=(",", ":")))
            
            self.last_saved_data = data.copy()
            return True
            
        except Exception as e:
            logger.error(f"Error saving data log: {e}")
            return False
    
    async def read_logs(self, start_date: str, end_date: str) -> List[Dict]:
        """
        Read logs within a date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            List of log entries
        """
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            logger.error(f"Invalid date format: {e}")
            raise ValueError("Invalid date format. Use YYYY-MM-DD.")
        
        results = []
        current = start
        delta = timedelta(days=1)
        
        while current <= end:
            date_dir = self.log_dir / current.strftime("%Y-%m-%d")
            
            if date_dir.exists():
                for log_file in sorted(date_dir.glob("*.json")):
                    try:
                        async with aiofiles.open(log_file, "r") as f:
                            content = await f.read()
                            logs = json.loads(content)
                            results.extend(logs)
                    except Exception as e:
                        logger.error(f"Error reading log file {log_file}: {e}")
            
            current += delta
        
        logger.info(f"Retrieved {len(results)} log entries from {start_date} to {end_date}")
        return results
