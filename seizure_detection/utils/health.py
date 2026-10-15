"""
Health monitoring utilities.
"""
import psutil
from datetime import datetime, timedelta
from typing import Dict
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class HealthMonitor:
    """
    Monitors system health metrics.
    """
    
    def __init__(self):
        self.start_time = datetime.now()
        
    def get_system_info(self) -> Dict:
        """Get current system health information."""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            uptime = datetime.now() - self.start_time
            
            return {
                "status": "healthy",
                "uptime_seconds": int(uptime.total_seconds()),
                "uptime": str(timedelta(seconds=int(uptime.total_seconds()))),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available // (1024 * 1024),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free // (1024 * 1024 * 1024),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting system info: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def check_health(self) -> tuple[bool, str]:
        """
        Check if system is healthy.
        
        Returns:
            Tuple of (is_healthy, message)
        """
        try:
            info = self.get_system_info()
            
            if info["status"] == "error":
                return False, f"System error: {info.get('error')}"
            
            # Check thresholds
            if info["cpu_percent"] > 90:
                return False, f"High CPU usage: {info['cpu_percent']}%"
            
            if info["memory_percent"] > 90:
                return False, f"High memory usage: {info['memory_percent']}%"
            
            if info["disk_percent"] > 90:
                return False, f"Low disk space: {info['disk_percent']}% used"
            
            return True, "All systems operational"
            
        except Exception as e:
            logger.error(f"Error checking health: {e}")
            return False, f"Health check failed: {e}"
