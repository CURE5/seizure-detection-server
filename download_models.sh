#!/bin/bash
# Download MediaPipe models for seizure detection
# Only downloads the models actually used by the application

echo "📥 Downloading MediaPipe models..."

# Create a folder for the models
mkdir -p mediapipe_models
cd mediapipe_models

# Download Pose Landmarker (outputs 33 landmarks for hypermotor detection)
echo "  - pose_landmarker.task (33 keypoints)..."
curl -o pose_landmarker.task -L https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task

# Download Face Landmarker (for gaze detection)
echo "  - face_landmarker.task (gaze tracking)..."
curl -o face_landmarker.task -L https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task

# Verify downloads
echo "✓ Downloaded models:"
ls -lh

echo "✓ Model download complete!"