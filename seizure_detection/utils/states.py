"""
State management enums and classes.
"""
from enum import Enum, auto


class CaptureState(Enum):
    """Camera capture state"""
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()


class RecordingState(Enum):
    """Video recording state"""
    IDLE = auto()
    STARTING = auto()
    RECORDING = auto()
    STOPPING = auto()


class MediaPipeState(Enum):
    """MediaPipe analysis state"""
    DISABLED = auto()
    STARTING = auto()
    ENABLED = auto()
    STOPPING = auto()


class CameraMode(Enum):
    """Camera mode (day/night)"""
    DAY = auto()
    NIGHT = auto()
    SWITCHING = auto()


class SleepMode(Enum):
    """Sleep mode states"""
    ACTIVE = auto()
    TRANSITIONING = auto()
    LIGHT_SLEEP = auto()
    DEEP_SLEEP = auto()
