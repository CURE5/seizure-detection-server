"""
Video recorder with FFmpeg support and proper error handling.
"""
import subprocess
import asyncio
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional
import cv2
import numpy as np
from ..config import VIDEO_DIR, CODEC, VIDEO_CONTAINER, USE_FFMPEG, REENCODE_WITH_FFMPEG
from ..utils.logger import setup_logger
from ..utils.states import RecordingState

logger = setup_logger(__name__)

# Temporary directory for videos awaiting re-encoding
TMP_DIR = VIDEO_DIR / "tmp"
if REENCODE_WITH_FFMPEG:
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def find_ffmpeg() -> Optional[str]:
    """
    Find FFmpeg executable on the system.
    
    Checks common locations and system PATH.
    Returns full path to ffmpeg or None if not found.
    """
    # First try shutil.which() which searches system PATH
    ffmpeg_path = shutil.which('ffmpeg')
    if ffmpeg_path:
        return ffmpeg_path
    
    # Common installation locations
    common_paths = [
        '/usr/bin/ffmpeg',
        '/usr/local/bin/ffmpeg',
        '/opt/homebrew/bin/ffmpeg',  # Apple Silicon Macs
        '/opt/local/bin/ffmpeg',     # MacPorts
    ]
    
    for path in common_paths:
        if Path(path).exists():
            return path
    
    return None


class VideoRecorder:
    """
    Manages video recording with FFmpeg support and state management.
    """
    
    def __init__(self, output_dir: Path = VIDEO_DIR):
        self.state = RecordingState.IDLE
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_file: Optional[Path] = None
        self.ffmpeg_process: Optional[subprocess.Popen] = None
        self.cv_writer: Optional[cv2.VideoWriter] = None
        self.fps: float = 5.0
        self.frame_size: Optional[tuple] = None
        
    def _start_ffmpeg(self, output_path: Path, fps: float, width: int, height: int) -> subprocess.Popen:
        """Start FFmpeg process for video encoding."""
        ffmpeg_path = find_ffmpeg()
        if not ffmpeg_path:
            raise FileNotFoundError("FFmpeg not found on system. Please install FFmpeg.")
        
        command = [
            ffmpeg_path,
            '-y',  # Overwrite output files without asking
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{width}x{height}',
            '-r', str(fps),
            '-i', '-',  # Input from stdin
            '-an',  # No audio
            '-vcodec', 'libx264',
            '-preset', 'veryfast',
            '-crf', '23',
            '-profile:v', 'baseline',
            '-level', '3.1',
            '-movflags', 'faststart',
            str(output_path)
        ]
        
        try:
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            logger.info(f"FFmpeg process started for {output_path}")
            return process
        except Exception as e:
            logger.error(f"Failed to start FFmpeg: {e}")
            raise
    
    async def start(self, fps: float = 5.0, frame_size: tuple = (640, 360)) -> bool:
        """
        Start video recording.
        
        Args:
            fps: Frames per second
            frame_size: (width, height) tuple
            
        Returns:
            True if started successfully
        """
        if self.state != RecordingState.IDLE:
            logger.warning(f"Cannot start recording - current state: {self.state}")
            return False
        
        self.state = RecordingState.STARTING
        self.fps = fps
        self.frame_size = frame_size
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"video_{timestamp}.{VIDEO_CONTAINER}"
        
        # If re-encoding, save to tmp directory first
        if REENCODE_WITH_FFMPEG and not USE_FFMPEG:
            self.current_file = TMP_DIR / filename
        else:
            self.current_file = self.output_dir / filename
        
        try:
            if USE_FFMPEG:
                self.ffmpeg_process = self._start_ffmpeg(
                    self.current_file,
                    fps,
                    frame_size[0],
                    frame_size[1]
                )
            else:
                fourcc = cv2.VideoWriter_fourcc(*CODEC)
                self.cv_writer = cv2.VideoWriter(
                    str(self.current_file),
                    fourcc,
                    fps,
                    frame_size
                )
                
                if not self.cv_writer.isOpened():
                    logger.error(f"Failed to open VideoWriter for {self.current_file}")
                    self.state = RecordingState.IDLE
                    return False
            
            self.state = RecordingState.RECORDING
            logger.info(f"Recording started: {self.current_file}")
            return True
            
        except Exception as e:
            logger.exception(f"Error starting recording: {e}")
            self.state = RecordingState.IDLE
            return False
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write a frame to the recording.
        
        Args:
            frame: Frame to write (BGR format)
            
        Returns:
            True if frame was written successfully
        """
        if self.state != RecordingState.RECORDING:
            return False
        
        try:
            if USE_FFMPEG and self.ffmpeg_process:
                # Check if process is still alive
                if self.ffmpeg_process.poll() is not None:
                    logger.error("FFmpeg process died unexpectedly")
                    return False
                
                self.ffmpeg_process.stdin.write(frame.tobytes())
                return True
                
            elif self.cv_writer:
                self.cv_writer.write(frame)
                return True
            
            return False
            
        except BrokenPipeError:
            logger.error("FFmpeg pipe broken - process may have crashed")
            return False
        except Exception as e:
            logger.error(f"Error writing frame: {e}")
            return False
    
    async def stop(self) -> Optional[Path]:
        """
        Stop recording and finalize the video file.
        
        Returns:
            Path to the recorded file, or None if failed
        """
        if self.state != RecordingState.RECORDING:
            return None
        
        self.state = RecordingState.STOPPING
        logger.info("Stopping recording...")
        
        output_file = self.current_file
        
        try:
            if USE_FFMPEG and self.ffmpeg_process:
                # Close stdin to signal end of input
                if self.ffmpeg_process.stdin:
                    self.ffmpeg_process.stdin.close()
                
                # Wait for FFmpeg to finish (with timeout)
                try:
                    stdout, stderr = await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(self.ffmpeg_process.communicate)
                        ),
                        timeout=30.0
                    )
                    
                    if self.ffmpeg_process.returncode != 0:
                        logger.error(f"FFmpeg failed: {stderr.decode()}")
                        output_file = None
                    else:
                        logger.info(f"Recording saved: {output_file}")
                        
                except asyncio.TimeoutError:
                    logger.error("FFmpeg did not finish in time, killing process")
                    self.ffmpeg_process.kill()
                    output_file = None
                
                self.ffmpeg_process = None
                
            elif self.cv_writer:
                self.cv_writer.release()
                self.cv_writer = None
                logger.info(f"Recording saved: {output_file}")
                
                # If re-encoding is enabled, schedule re-encoding
                if REENCODE_WITH_FFMPEG and output_file:
                    filename = output_file.name
                    logger.info(f"Scheduling re-encode for {filename}")
                    # Re-encode in background (don't await to avoid blocking)
                    asyncio.create_task(reencode_video(filename))
                    # Return the final output path (not the tmp path)
                    output_file = self.output_dir / filename
            
            self.state = RecordingState.IDLE
            self.current_file = None
            
            return output_file
            
        except Exception as e:
            logger.exception(f"Error stopping recording: {e}")
            self.state = RecordingState.IDLE
            return None
    
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self.state == RecordingState.RECORDING
    
    def get_state(self) -> dict:
        """Get current recorder state information."""
        return {
            "state": self.state.name,
            "current_file": str(self.current_file) if self.current_file else None,
            "fps": self.fps,
            "frame_size": self.frame_size
        }


async def reencode_video(filename: str) -> bool:
    """
    Re-encode a video file using FFmpeg for web compatibility.
    
    This is useful when recording with OpenCV (which may produce files with
    compatibility issues) and then re-encoding with FFmpeg for maximum web support.
    
    Args:
        filename: Name of the video file to re-encode
        
    Returns:
        True if re-encoding was successful
    """
    input_path = TMP_DIR / filename
    output_path = VIDEO_DIR / filename
    
    if not input_path.exists():
        logger.error(f"Input file not found for re-encoding: {input_path}")
        return False
    
    # Find FFmpeg on system
    ffmpeg_path = find_ffmpeg()
    if not ffmpeg_path:
        logger.error("FFmpeg not found on system. Please install FFmpeg or set REENCODE_WITH_FFMPEG=False")
        return False
    
    ffmpeg_command = [
        ffmpeg_path,
        '-i', str(input_path),
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-crf', '23',               # Quality setting (lower = better quality, bigger file)
        '-profile:v', 'baseline',   # Baseline profile for mobile compatibility
        '-level', '3.1',            # H.264 level (common choice for wide device support)
        '-c:a', 'aac',
        '-movflags', 'faststart',   # Place 'moov' atom at beginning for fast web streaming
        str(output_path)
    ]
    
    logger.info(f"Starting re-encoding: {input_path} -> {output_path}")
    
    try:
        # Create subprocess
        process = await asyncio.create_subprocess_exec(
            *ffmpeg_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for completion
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            logger.info(f"Re-encoded video saved to {output_path}")
            # Delete the temporary file
            try:
                input_path.unlink()
                logger.info(f"Deleted temporary file: {input_path}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file: {e}")
            return True
        else:
            logger.error(f"FFmpeg failed with return code {process.returncode}")
            logger.error(f"stderr: {stderr.decode()}")
            return False
            
    except Exception as e:
        logger.exception(f"Error during re-encoding: {e}")
        return False
