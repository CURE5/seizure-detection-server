"""
Internet connectivity checker with improved reboot logic.
"""
import subprocess
import asyncio
from datetime import datetime
from ..config import ConnectivityConfig
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ConnectivityChecker:
    """
    Monitors internet connectivity and manages reboots safely.
    """
    
    def __init__(self, config: ConnectivityConfig = ConnectivityConfig()):
        self.config = config
        self.last_check_time: datetime = datetime.now()
        self.failed_attempts: int = 0
        
    def has_internet(self, host: str = None, timeout: int = None) -> bool:
        """
        Check if internet connection is available.
        
        Args:
            host: Host to ping (defaults to config)
            timeout: Timeout in seconds (defaults to config)
            
        Returns:
            True if internet is available
        """
        host = host or self.config.PING_HOST
        timeout = timeout or self.config.PING_TIMEOUT
        
        try:
            subprocess.run(
                ["ping", "-c", "1", "-W", str(timeout), host],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
                timeout=timeout + 1
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
    
    async def monitor(self, is_recording_func, is_seizing_func):
        """
        Monitor internet connectivity and reboot if necessary.
        
        Args:
            is_recording_func: Function that returns True if currently recording
            is_seizing_func: Function that returns True if seizure is active
        """
        # Initial delay after startup
        await asyncio.sleep(self.config.CHECK_INTERVAL)
        
        while True:
            try:
                if self.has_internet():
                    logger.debug("Internet connection verified")
                    self.failed_attempts = 0
                    await asyncio.sleep(self.config.CHECK_INTERVAL)
                    continue
                
                # Connection lost
                logger.warning(
                    f"Internet connection lost (attempt {self.failed_attempts + 1}/"
                    f"{self.config.RETRIES})"
                )
                self.failed_attempts += 1
                
                # Retry before giving up
                if self.failed_attempts < self.config.RETRIES:
                    await asyncio.sleep(self.config.RETRY_DELAY)
                    continue
                
                # Check if it's safe to reboot
                if is_recording_func() or is_seizing_func():
                    logger.warning(
                        "Internet down but recording/seizure in progress. "
                        "Delaying reboot."
                    )
                    self.failed_attempts = 0  # Reset to retry later
                    await asyncio.sleep(self.config.CHECK_INTERVAL)
                    continue
                
                # Perform reboot
                logger.critical("Internet connection lost. Initiating system reboot.")
                await asyncio.sleep(5)  # Give time for log to flush
                
                subprocess.run(["sudo", "reboot"], check=False)
                break  # Exit loop (system will reboot)
                
            except asyncio.CancelledError:
                logger.info("Connectivity monitoring cancelled")
                break
            except Exception as e:
                logger.exception(f"Error in connectivity monitor: {e}")
                await asyncio.sleep(60)  # Wait before retrying
