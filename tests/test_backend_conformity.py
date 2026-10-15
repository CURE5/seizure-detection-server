#!/usr/bin/env python
"""Test script to verify multi-backend conformity."""

import sys
from pathlib import Path

# Add parent directory to path to import seizure_detection module
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from seizure_detection.analysis import pose_detection

# These are integration tests: they run the real pose backend and need the
# MediaPipe model files from download_models.sh. Skipped when absent (e.g. CI).
_MODELS = Path(__file__).parent.parent / "mediapipe_models" / "pose_landmarker.task"
pytestmark = pytest.mark.skipif(not _MODELS.exists(), reason="mediapipe_models/ not downloaded")

def test_backend_detection():
    """Test that backend detection works."""
    print(f"✓ Backend detected: {pose_detection.BACKEND}")
    print(f"✓ Pose detection available: {pose_detection.is_pose_detection_available}")
    return True

def test_setup_release():
    """Test setup and release functions."""
    print("\nTesting setup()...")
    pose_detection.setup()
    print("✓ setup() completed")
    
    print("\nTesting release()...")
    pose_detection.release()
    print("✓ release() completed")
    return True

def test_process_frame():
    """Test process_frame with a dummy frame."""
    print("\nTesting process_frame()...")
    
    # Create a dummy frame (480x640x3 BGR image)
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Call setup first
    pose_detection.setup()
    
    # Process the frame
    result_frame, data = pose_detection.process_frame(
        dummy_frame, do_draw=False, do_flip=False
    )
    
    print(f"✓ process_frame() returned frame shape: {result_frame.shape}")
    print(f"✓ Data keys: {data.keys()}")
    print(f"✓ Gaze: {data.get('gaze')}")
    print(f"✓ Pose: {type(data.get('pose'))}")
    
    # Verify expected structure
    assert 'gaze' in data, "Missing 'gaze' key"
    assert 'pose' in data, "Missing 'pose' key"
    
    # Check backend field if not "none"
    if pose_detection.BACKEND != "none":
        assert 'backend' in data, "Missing 'backend' key"
        print(f"✓ Backend field: {data.get('backend')}")
    
    # Cleanup
    pose_detection.release()
    
    print("✓ process_frame() conforms to expected interface")
    return True

def test_helper_functions():
    """Test that helper functions exist."""
    print("\nTesting helper functions exist...")
    
    assert hasattr(pose_detection, 'serialize_pose_landmarker_result'), \
        "Missing serialize_pose_landmarker_result"
    print("✓ serialize_pose_landmarker_result exists")
    
    assert hasattr(pose_detection, 'extract_eyes_and_hands'), \
        "Missing extract_eyes_and_hands"
    print("✓ extract_eyes_and_hands exists")
    
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Backend Conformity Test")
    print("=" * 60)
    
    try:
        test_backend_detection()
        test_setup_release()
        test_process_frame()
        test_helper_functions()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - Backend implementation conforms!")
        print("=" * 60)
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
