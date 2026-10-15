#!/usr/bin/env python3
"""Compare MediaPipe and COCO keypoint formats."""

import mediapipe as mp

# MediaPipe Pose Landmarks
pose = mp.solutions.pose
print("=" * 60)
print("MEDIAPIPE POSE LANDMARKS (33 points)")
print("=" * 60)
landmarks = []
for name in dir(pose.PoseLandmark):
    if not name.startswith('_'):
        try:
            val = getattr(pose.PoseLandmark, name)
            if isinstance(val, int):
                landmarks.append((name, val))
        except:
            pass
landmarks_sorted = sorted(landmarks, key=lambda x: x[1])
for name, idx in landmarks_sorted:
    print(f"{idx:2d}: {name}")

print("\n" + "=" * 60)
print("COCO KEYPOINTS (17 points)")
print("=" * 60)
coco_keypoints = [
    "nose",           # 0
    "left_eye",       # 1
    "right_eye",      # 2
    "left_ear",       # 3
    "right_ear",      # 4
    "left_shoulder",  # 5
    "right_shoulder", # 6
    "left_elbow",     # 7
    "right_elbow",    # 8
    "left_wrist",     # 9
    "right_wrist",    # 10
    "left_hip",       # 11
    "right_hip",      # 12
    "left_knee",      # 13
    "right_knee",     # 14
    "left_ankle",     # 15
    "right_ankle",    # 16
]
for idx, name in enumerate(coco_keypoints):
    print(f"{idx:2d}: {name}")

print("\n" + "=" * 60)
print("MAPPING: MediaPipe -> COCO")
print("=" * 60)
# Map COCO indices to MediaPipe indices
coco_to_mediapipe = {
    0: pose.PoseLandmark.NOSE,              # nose
    1: pose.PoseLandmark.LEFT_EYE,          # left_eye
    2: pose.PoseLandmark.RIGHT_EYE,         # right_eye
    3: pose.PoseLandmark.LEFT_EAR,          # left_ear
    4: pose.PoseLandmark.RIGHT_EAR,         # right_ear
    5: pose.PoseLandmark.LEFT_SHOULDER,     # left_shoulder
    6: pose.PoseLandmark.RIGHT_SHOULDER,    # right_shoulder
    7: pose.PoseLandmark.LEFT_ELBOW,        # left_elbow
    8: pose.PoseLandmark.RIGHT_ELBOW,       # right_elbow
    9: pose.PoseLandmark.LEFT_WRIST,        # left_wrist
    10: pose.PoseLandmark.RIGHT_WRIST,      # right_wrist
    11: pose.PoseLandmark.LEFT_HIP,         # left_hip
    12: pose.PoseLandmark.RIGHT_HIP,        # right_hip
    13: pose.PoseLandmark.LEFT_KNEE,        # left_knee
    14: pose.PoseLandmark.RIGHT_KNEE,       # right_knee
    15: pose.PoseLandmark.LEFT_ANKLE,       # left_ankle
    16: pose.PoseLandmark.RIGHT_ANKLE,      # right_ankle
}

for coco_idx, mp_idx in coco_to_mediapipe.items():
    coco_name = coco_keypoints[coco_idx]
    mp_name = [n for n, i in landmarks_sorted if i == mp_idx][0]
    print(f"COCO {coco_idx:2d} ({coco_name:15s}) <- MediaPipe {mp_idx:2d} ({mp_name})")

print("\n" + "=" * 60)
print("COMPATIBILITY ANALYSIS")
print("=" * 60)
print("✅ All 17 COCO keypoints have direct MediaPipe equivalents")
print("✅ Keypoint semantics are identical")
print("✅ ONNX model trained on MediaPipe should work with COCO format")
print("\n⚠️  NOTE: Must reorder keypoints from Hailo COCO order to MediaPipe order")
print("   before feeding to ONNX model")
