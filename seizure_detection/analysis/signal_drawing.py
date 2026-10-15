"""Drawing overlay for signal pipeline telemetry."""
import cv2
import numpy as np
from typing import Dict, List

def draw_signal_overlay(frame: np.ndarray, telemetry: Dict, events: List[Dict]) -> np.ndarray:
    """Draw signal pipeline telemetry on frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    
    # Draw person/head boxes
    for det in telemetry.get("boxes", []):
        bbox = det.get("bbox")
        if bbox:
            x, y, bw, bh = bbox
            cv2.rectangle(overlay, (x, y), (x+bw, y+bh), (0, 255, 255), 2)  # Yellow person box
        
        head = det.get("head")
        if head:
            hx, hy, hw, hh = head
            cv2.rectangle(overlay, (hx, hy), (hx+hw, hy+hh), (255, 255, 0), 2)  # Cyan head box
    
    # Draw torso blob
    blobs = telemetry.get("blobs", {})
    torso = blobs.get("torso")
    if torso:
        tx, ty, tw, th = torso
        cv2.rectangle(overlay, (tx, ty), (tx+tw, ty+th), (0, 255, 0), 2)  # Green torso
    
    # Draw metrics text
    y_offset = 30
    motion = telemetry.get("motion", {})
    periodicity = telemetry.get("periodicity", {})
    breathing = telemetry.get("breathing", {})
    eyes = telemetry.get("eyes", {})
    
    metrics = [
        f"Motion: {motion.get('flowRMS', 0):.3f}",
        f"Period: {periodicity.get('peakHz') or 'N/A'} Hz",
        f"Band: {periodicity.get('bandPower', 0):.2f}",
        f"Breathing: {breathing.get('rrBpm') or 'N/A'} bpm",
        f"Eyes: {eyes.get('flutterHz') or 'N/A'} Hz"
    ]
    
    for metric in metrics:
        cv2.putText(overlay, metric, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
    
    # Draw events
    if events:
        y_offset += 10
        cv2.putText(overlay, f"EVENTS: {len(events)}", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        y_offset += 25
        for evt in events[:3]:  # Show max 3 events
            kind = evt.get("kind", "UNKNOWN")
            cv2.putText(overlay, f"  {kind}", (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            y_offset += 20
    
    return overlay
