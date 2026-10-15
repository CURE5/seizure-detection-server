"""
Camera manager with state machine for reliable camera operations.
"""
import cv2
import platform
import asyncio
from typing import Optional
import numpy as np
from ..config import DEFAULT_RESOLUTION, DAY_CAMERA_NUM, NIGHT_CAMERA_NUM, VIDEO_PATH
from ..utils.logger import setup_logger
from ..utils.states import CaptureState, CameraMode

logger = setup_logger(__name__)


class CameraManager:
    """
    Manages camera capture with proper state management and error handling.
    """
    
    def __init__(self):
        self.state = CaptureState.STOPPED
        self.mode = CameraMode.DAY
        self.cam = None
        self.camera_src: Optional[str] = None
        self.resolution = DEFAULT_RESOLUTION
        self.platform_name = platform.system()
        self.camera_options = {}
        
        # Available camera controls for Pi
        self.camera_ctrl_info = [
            'AwbMode', 'AeEnable', 'AeFlickerPeriod', 'AfWindows', 'ColourGains',
            'Contrast', 'LensPosition', 'AfTrigger', 'AfMetering', 'AnalogueGain',
            'AfPause', 'AeMeteringMode', 'HdrMode', 'AwbEnable', 'AfSpeed', 'AfRange',
            'FrameDurationLimits', 'AfMode', 'ExposureValue', 'AeFlickerMode',
            'ExposureTime', 'AeExposureMode', 'Sharpness', 'AeConstraintMode',
            'ScalerCrop', 'NoiseReductionMode', 'StatsOutputEnable', 'Saturation',
            'CnnEnableInputTensor', 'Brightness'
        ]
    
    def _configure_pi_camera(self, camera_num: int):
        """Configure Raspberry Pi camera."""
        try:
            from picamera2 import Picamera2
            
            self.cam = Picamera2(camera_num=camera_num)
            self.cam.preview_configuration.main.size = self.resolution
            self.cam.preview_configuration.main.format = "RGB888"
            self.cam.preview_configuration.align()
            self.cam.configure("preview")
            self.cam.start()
            
            logger.info(f"Pi camera {camera_num} configured successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to configure Pi camera {camera_num}: {e}")
            return False
    
    def _configure_generic_camera(self, camera_id: int = 0):
        """Configure generic camera using OpenCV."""
        try:
            # Use AVFoundation backend on macOS for better compatibility
            if self.platform_name == "Darwin":
                self.cam = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
            else:
                self.cam = cv2.VideoCapture(camera_id)
            
            if not self.cam.isOpened():
                logger.error("Failed to open camera")
                return False
            
            # Set camera properties for better performance
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            self.cam.set(cv2.CAP_PROP_FPS, 30)
            
            # On macOS, read a few frames to let camera warm up
            if self.platform_name == "Darwin":
                logger.info("Warming up camera...")
                for _ in range(5):
                    self.cam.read()
                    import time
                    time.sleep(0.1)
            
            # Verify we can read a frame
            ret, frame = self.cam.read()
            if not ret or frame is None:
                logger.error("Camera opened but cannot read frames")
                return False
            
            logger.info(f"Generic camera {camera_id} opened successfully (frame size: {frame.shape})")
            return True
        except Exception as e:
            logger.error(f"Failed to configure generic camera: {e}")
            return False
    
    def _configure_video_file(self, video_path: str):
        """Configure video file as source."""
        try:
            self.cam = cv2.VideoCapture(video_path)
            if not self.cam.isOpened():
                logger.error(f"Failed to open video file: {video_path}")
                return False
            
            logger.info(f"Video file configured: {video_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to configure video file: {e}")
            return False
    
    async def start(self, video_path: Optional[str] = None, night_mode: bool = False) -> bool:
        """
        Start camera capture.
        
        Args:
            video_path: Optional path to video file (for testing/analysis)
            night_mode: Whether to use night camera (Pi only)
            
        Returns:
            True if started successfully
        """
        if self.state != CaptureState.STOPPED:
            logger.warning(f"Cannot start camera - current state: {self.state}")
            return False
        
        self.state = CaptureState.STARTING
        logger.info("Starting camera...")
        
        try:
            # Determine camera source
            if video_path or VIDEO_PATH:
                self.camera_src = "video"
                success = self._configure_video_file(video_path or VIDEO_PATH)
            elif self.platform_name == "Darwin":  # macOS
                self.camera_src = "mac_cam"
                success = self._configure_generic_camera(0)
            elif self.platform_name == "Linux":  # Raspberry Pi
                self.camera_src = "pi_cam"
                camera_num = NIGHT_CAMERA_NUM if night_mode else DAY_CAMERA_NUM
                self.mode = CameraMode.NIGHT if night_mode else CameraMode.DAY
                success = self._configure_pi_camera(camera_num)
            else:
                logger.error(f"Unsupported platform: {self.platform_name}")
                success = False
            
            if success:
                self.state = CaptureState.RUNNING
                logger.info(f"Camera started successfully (source: {self.camera_src})")
                return True
            else:
                self.state = CaptureState.STOPPED
                logger.error("Failed to start camera")
                return False
                
        except Exception as e:
            logger.exception(f"Error starting camera: {e}")
            self.state = CaptureState.STOPPED
            return False
    
    async def stop(self):
        """Stop camera capture."""
        if self.state == CaptureState.STOPPED:
            return
        
        self.state = CaptureState.STOPPING
        logger.info("Stopping camera...")
        
        try:
            if self.cam is not None:
                if self.camera_src == "pi_cam":
                    self.cam.stop()
                    self.cam.close()
                else:
                    self.cam.release()
                    cv2.destroyAllWindows()
                
                self.cam = None
            
            self.state = CaptureState.STOPPED
            self.camera_src = None
            logger.info("Camera stopped successfully")
            
        except Exception as e:
            logger.exception(f"Error stopping camera: {e}")
            self.state = CaptureState.STOPPED
    
    async def switch_mode(self, night_mode: bool) -> bool:
        """
        Switch between day and night camera (Pi only).
        
        Args:
            night_mode: True for night camera, False for day camera
            
        Returns:
            True if switched successfully
        """
        if self.camera_src != "pi_cam":
            logger.warning("Camera mode switching only available on Raspberry Pi")
            return False
        
        target_mode = CameraMode.NIGHT if night_mode else CameraMode.DAY
        if self.mode == target_mode:
            logger.info(f"Already in {target_mode.name} mode")
            return True
        
        logger.info(f"Switching camera mode to {target_mode.name}")
        self.mode = CameraMode.SWITCHING
        
        # Stop current camera
        await self.stop()
        
        # Small delay for camera release
        await asyncio.sleep(0.5)
        
        # Start with new mode
        success = await self.start(night_mode=night_mode)
        
        if success:
            logger.info(f"Successfully switched to {target_mode.name} mode")
        else:
            logger.error(f"Failed to switch to {target_mode.name} mode")
            self.mode = CameraMode.DAY  # Fallback
        
        return success
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Returns:
            Frame as numpy array, or None if capture failed
        """
        if self.state != CaptureState.RUNNING or self.cam is None:
            return None
        
        try:
            if self.camera_src == "pi_cam":
                frame = self.cam.capture_array()
            else:
                ret, frame = self.cam.read()
                if not ret:
                    if self.camera_src == "video":
                        # Loop video
                        self.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    return None
            
            # Resize to standard resolution
            if frame.shape[:2] != self.resolution[::-1]:  # height, width
                frame = cv2.resize(frame, self.resolution)
            
            return frame
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    async def set_controls(self, options: dict) -> bool:
        """
        Set camera controls (Pi only).
        
        Args:
            options: Dictionary of control options
            
        Returns:
            True if controls were set successfully
        """
        if self.camera_src != "pi_cam" or self.cam is None:
            logger.warning("Camera controls only available on Raspberry Pi")
            return False
        
        # Validate control keys
        for name in options.keys():
            if name not in self.camera_ctrl_info:
                logger.error(f"Invalid control key: {name}")
                return False
        
        try:
            self.cam.set_controls(options)
            self.camera_options.update(options)
            logger.info(f"Camera controls updated: {options}")
            return True
        except Exception as e:
            logger.error(f"Error setting camera controls: {e}")
            return False
    
    async def set_defaults(self):
        """Set default camera controls (Pi only)."""
        if self.camera_src != "pi_cam":
            return
        
        default_options = {
            "AfMode": 0,  # Manual focus
            "LensPosition": 0.0  # Infinity focus
        }
        
        await self.set_controls(default_options)
    
    def is_running(self) -> bool:
        """Check if camera is currently running."""
        return self.state == CaptureState.RUNNING
    
    def get_state(self) -> dict:
        """Get current camera state information."""
        return {
            "state": self.state.name,
            "mode": self.mode.name,
            "source": self.camera_src,
            "resolution": self.resolution,
            "options": self.camera_options
        }
