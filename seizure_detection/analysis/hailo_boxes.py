"""Lightweight person detection wrapper using Hailo YOLOX model."""
import logging
import numpy as np
from typing import List, Dict

logger = logging.getLogger(__name__)

try:
    from hailo_platform import HEF, VDevice
    HAILO = True
except Exception:
    HAILO = False


class HailoBoxes:
    """
    Person detection using Hailo YOLOX object detection model.
    
    Tries YOLOX first (better NMS handling), falls back to pose model if needed.
    """

    def __init__(
        self, hef_path="/usr/share/hailo-models/yolox_s_leaky_h8l_rpi.hef"
    ):
        self.active = False
        self.use_pose = False
        if not HAILO:
            return
        try:
            logger.info(f"Attempting Hailo YOLOX detector: {hef_path}")
            self.hef = HEF(hef_path)
            self.device = VDevice()
            self.network_group = self._configure_network_group()
            self.active = True
            logger.info("✓ Hailo YOLOX object detector initialized")
        except Exception as e:
            logger.warning(f"YOLOX init failed ({e}), trying pose model fallback...")
            try:
                from .hailo_pose import HailoPoseDetector
                pose_path = "/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef"
                self.pose_detector = HailoPoseDetector(pose_path)
                self.active = True
                self.use_pose = True
                logger.info("✓ Using pose model fallback for person detection")
            except Exception as e2:
                logger.error(f"Both YOLOX and pose init failed: {e2}")
                self.active = False
    
    def _configure_network_group(self):
        """Configure network group with default params."""
        config = self.device.create_configure_params(self.hef)
        network_group = self.device.configure(self.hef, config)[0]
        return network_group
    
    def _nms(self, boxes, scores, iou_threshold=0.5):
        """Non-Maximum Suppression to remove overlapping boxes."""
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]
        areas = boxes[:, 2] * boxes[:, 3]
        
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep

    def detect(self, frame) -> List[Dict]:
        """
        Run person detection.
        """
        if not self.active:
            H, W = frame.shape[:2]
            fallback = {
                "label": "person",
                "score": 0.5,
                "bbox": (0, 0, W, H),
                "head": (0, 0, W, int(H * 0.3))
            }
            return [fallback]

        # Use pose fallback if YOLOX failed to init
        if self.use_pose:
            return self._detect_pose(frame)
        
        try:
            # Run YOLOX inference
            H, W = frame.shape[:2]
            input_name = self.network_group.get_input_vstream_infos()[0].name
            input_data = {input_name: np.expand_dims(frame, axis=0)}
            
            with self.network_group.activate():
                output = self.network_group.infer(input_data)
            
            # Parse YOLOX output
            output_name = list(output.keys())[0]
            detections = output[output_name][0]
            
            results = []
            for det in detections:
                if len(det) >= 7:
                    x1, y1, x2, y2, obj_conf, class_conf, class_id = det[:7]
                    
                    # Person class (0) with confidence > 0.3
                    if int(class_id) == 0 and obj_conf > 0.3:
                        x, y = int(x1), int(y1)
                        w, h = int(x2 - x1), int(y2 - y1)
                        
                        results.append({
                            "label": "person",
                            "score": float(obj_conf),
                            "bbox": (x, y, w, h),
                            "head": (x, y, w, int(h * 0.3))
                        })
            
            # Apply NMS and limit
            if len(results) > 1:
                boxes = np.array([r["bbox"] for r in results])
                scores = np.array([r["score"] for r in results])
                keep = self._nms(boxes, scores, iou_threshold=0.4)[:3]
                results = [results[i] for i in keep]
            
            logger.info(f"YOLOX: {len(detections)} raw → {len(results)} persons")
            return results

        except Exception as e:
            logger.error(f"YOLOX detection error: {e}", exc_info=True)
            return []
    
    def _detect_pose(self, frame) -> List[Dict]:
        """Fallback: use pose model for bounding boxes."""
        try:
            pose_detections = self.pose_detector.detect(frame)
            
            boxes = []
            scores = []
            for pose in pose_detections:
                bbox = pose.get("bbox")
                score = pose.get("confidence", 0.0)
                if bbox and score > 0.4:
                    boxes.append(bbox)
                    scores.append(score)
            
            if len(boxes) == 0:
                return []
            
            boxes = np.array(boxes)
            scores = np.array(scores)
            keep = self._nms(boxes, scores, iou_threshold=0.3)[:3]
            
            results = []
            for idx in keep:
                x, y, w, h = boxes[idx]
                results.append({
                    "label": "person",
                    "score": float(scores[idx]),
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "head": (int(x), int(y), int(w), int(h * 0.3))
                })
            
            logger.info(f"Pose fallback: {len(pose_detections)} → {len(results)} persons")
            return results
        except Exception as e:
            logger.error(f"Pose detection error: {e}")
            return []
