"""Signal pipeline combining Hailo boxes and classical CV features."""
import cv2
import time
import numpy as np
from typing import Dict, List, Tuple
from .hailo_boxes import HailoBoxes
from .cv_features.motion_energy import MotionEnergy
from .cv_features.periodicity import Periodicity
from .cv_features.blobs import Blobs
from .cv_features.breathing import Breathing
from .cv_features.eyes import Eyes
from .signal_drawing import draw_signal_overlay

class SignalPipeline:
    def __init__(self, use_hailo=True, fps=20):
        self.boxes = HailoBoxes() if use_hailo else None
        self.motion = MotionEnergy()
        self.period = Periodicity(fs=fps)
        self.blobs = Blobs()
        self.breath = Breathing(fs=fps)
        self.eyes = Eyes(fs=fps)

    def process_frame(self, bgr: np.ndarray, do_draw: bool = False) -> Tuple[np.ndarray, Dict, List[Dict]]:
        ts = int(time.time()*1000)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 1) Motion features
        m = self.motion.update(gray)
        p = self.period.update(ts, m["flowRMS"])

        # 2) Person/head boxes
        if self.boxes and self.boxes.active:
            dets = self.boxes.detect(bgr)
        else:
            H, W = bgr.shape[:2]
            dets = [dict(label="person", score=0.5, bbox=(0,0,W,H), head=(0,0,W,int(H*0.3)))]
        
        det = max(dets, key=lambda d: d["bbox"][2]*d["bbox"][3]) if dets else None
        person = det["bbox"] if det else None
        head = det.get("head") if det else None

        # 3) Blobs & segmentation
        bl = self.blobs.update(gray, person_box=person, head_hint=head)

        # 4) Breathing & eyes
        br = self.breath.update(ts, gray, torso_box=bl.get("torso"))
        ey = self.eyes.update(ts, gray, head_box=bl.get("head"))

        telemetry = {
            "motion": m,
            "periodicity": p,
            "blobs": bl,
            "breathing": br,
            "eyes": ey,
            "boxes": dets
        }

        # Simple eventing
        events = []
        if p.get("bandPower", 0) > 0.45 and p.get("peakPower", 0) > 0.25 and p.get("peakHz"):
            events.append({
                "kind": "RHYTHMIC_SHAKE",
                "hz": p["peakHz"],
                "power": p["peakPower"],
                "ts": ts
            })
        if br.get("rrBpm") is not None and (br["rrBpm"] < 8 or br["rrBpm"] > 40):
            events.append({
                "kind": "APNEA_RISK",
                "rrBpm": br["rrBpm"],
                "ts": ts
            })
        if ey.get("flutterHz") and ey["flutterHz"] >= 10:
            events.append({
                "kind": "EYE_FLUTTER",
                "hz": ey["flutterHz"],
                "ts": ts
            })

        # Draw overlays if requested
        frame_out = draw_signal_overlay(bgr, telemetry, events) if do_draw else bgr
        return (frame_out, telemetry, events)
