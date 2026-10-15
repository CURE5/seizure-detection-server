"""
Hailo AI hardware-accelerated pose detection for Raspberry Pi 5.
Direct frame-by-frame inference using https://github.com/DeGirum/hailo_examples/blob/main/user_guides/003_simplified_object_detection.md.
"""
import numpy as np
import cv2
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

try:
    from hailo_platform import (
        HEF,
        VDevice,
        HailoStreamInterface,
        InferVStreams,
        ConfigureParams,
        InputVStreamParams,
        OutputVStreamParams,
        FormatType,
    )

    HAILO_AVAILABLE = True
    logger.info("✓ Hailo platform available")
except ImportError as e:
    HAILO_AVAILABLE = False
    logger.warning(f"Hailo platform not available: {e}")


# COCO keypoint indices for pose estimation
COCO_KEYPOINTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}


class HailoPoseDetector:
    """
    Direct Hailo inference for pose detection using HailoRT Python API.
    Processes individual frames without GStreamer pipeline.
    """

    def __init__(
        self, hef_path="/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef"
    ):
        """Initialize Hailo pose detector."""
        if not HAILO_AVAILABLE:
            raise ImportError("Hailo platform not available")

        self.hef_path = hef_path
        self.device = None
        self.network_group = None
        self.input_vstreams = None
        self.output_vstreams = None
        self.input_shape = None
        self.output_shapes = None
        self._raw_debug_samples = 0
        self.pad_x = 0
        self.pad_y = 0
        self.scale = 1.0

        logger.info("Initializing Hailo pose detector with model: %s", hef_path)
        self._init_device()

    def _init_device(self):
        """Initialize Hailo device and load model."""
        try:
            self.device = VDevice()
            logger.info("✓ Hailo device created")

            hef = HEF(self.hef_path)
            logger.info("✓ Loaded HEF: %s", self.hef_path)

            configure_params = ConfigureParams.create_from_hef(
                hef, interface=HailoStreamInterface.PCIe
            )
            self.network_group = self.device.configure(hef, configure_params)[0]
            logger.info("✓ Network group configured")

            input_vstream_info = hef.get_input_vstream_infos()[0]
            output_vstream_infos = hef.get_output_vstream_infos()

            self.input_name = input_vstream_info.name
            self.input_shape = input_vstream_info.shape
            self.output_shapes = [info.shape for info in output_vstream_infos]
            logger.info(
                "  Input: %s, shape: %s", self.input_name, self.input_shape
            )
            logger.info("  %d output tensors", len(self.output_shapes))

            self.input_vstreams_params = InputVStreamParams.make_from_network_group(
                self.network_group,
                quantized=False,
                format_type=FormatType.FLOAT32,
            )
            self.output_vstreams_params = OutputVStreamParams.make_from_network_group(
                self.network_group,
                quantized=False,
                format_type=FormatType.FLOAT32,
            )

            self.network_group_params = self.network_group.create_params()
            logger.info("✓ Hailo pose detector initialized successfully")
            logger.info("  Model: yolov8s_pose (17 COCO keypoints)")
            logger.info("  Hardware: Hailo-8L AI accelerator")
        except Exception as exc:
            logger.error("Failed to initialize Hailo device: %s", exc)
            raise

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for Hailo inference with letterboxing.
        Maintains aspect ratio and stores padding info for coordinate mapping.

        Args:
            frame: BGR image from OpenCV

        Returns:
            Preprocessed image ready for inference
        """
        target_h, target_w = self.input_shape[0], self.input_shape[1]
        frame_h, frame_w = frame.shape[:2]

        scale = min(target_w / frame_w, target_h / frame_h)
        new_w = int(frame_w * scale)
        new_h = int(frame_h * scale)

        resized = cv2.resize(frame, (new_w, new_h))

        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized

        self.pad_x = pad_x
        self.pad_y = pad_y
        self.scale = scale

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        batched = np.expand_dims(normalized, axis=0)
        return batched

    def _sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def _softmax(self, x):
        """Softmax activation function."""
        return np.exp(x) / np.expand_dims(np.sum(np.exp(x), axis=-1), axis=-1)

    def _log_raw_kpts(self, scale_name: str, tensor: np.ndarray) -> None:
        """Log raw keypoint tensor stats before decoding."""
        if tensor is None:
            return

        try:
            flat = tensor.reshape(-1, 17, 3)
        except ValueError:
            logger.warning(
                "Could not reshape raw keypoints for scale %s: %s",
                scale_name,
                tensor.shape,
            )
            return

        x_vals = flat[..., 0]
        y_vals = flat[..., 1]
        conf_vals = flat[..., 2]

        logger.info(
            "DEBUG raw kpts scale %s x[%.4f,%.4f] y[%.4f,%.4f] c[%.4f,%.4f]",
            scale_name,
            float(x_vals.min()),
            float(x_vals.max()),
            float(y_vals.min()),
            float(y_vals.max()),
            float(conf_vals.min()),
            float(conf_vals.max()),
        )

        sample_x = float(x_vals.flat[0])
        sample_y = float(y_vals.flat[0])
        sample_conf = float(conf_vals.flat[0])
        logger.info(
            "DEBUG raw sample scale %s -> (x=%.4f, y=%.4f, c=%.4f)",
            scale_name,
            sample_x,
            sample_y,
            sample_conf,
        )

    def _xywh2xyxy(self, x):
        """Convert boxes from [x_center, y_center, w, h] to [x1, y1, x2, y2]."""
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        return y

    def _decode_kpts_from_head(self, head, stride, net_h, net_w):
        """
        Decode keypoints from a single output head following Hailo reference.
        Channel layout: x at 0::3, y at 1::3, conf at 2::3 (17 kp × 3 = 51 channels)
        
        Args:
            head: (1, Hs, Ws, 51) NHWC tensor
            stride: Stride for this head
            net_h, net_w: Network input dimensions
            
        Returns:
            kpts: (Ncells, 17, 3) decoded keypoints in net-pixel space
        """
        # Remove batch dimension
        head = head[0]  # (Hs, Ws, 51)
        Hs, Ws, _ = head.shape
        
        # Build grid (pixel centers in net space)
        xs = (np.arange(Ws) + 0.5)
        ys = (np.arange(Hs) + 0.5)
        gx, gy = np.meshgrid(xs, ys)  # (Hs, Ws)
        gx = gx * stride
        gy = gy * stride
        
        # Slice channels: x at 0::3, y at 1::3, conf at 2::3 (17 kp × 3)
        x_raw = head[:, :, 0::3]    # (Hs, Ws, 17)
        y_raw = head[:, :, 1::3]    # (Hs, Ws, 17)
        confraw = head[:, :, 2::3]  # (Hs, Ws, 17)
        
        # YOLOv8 pose formula
        x = (1.0 / (1.0 + np.exp(-x_raw))) * 2.0 - 0.5
        y = (1.0 / (1.0 + np.exp(-y_raw))) * 2.0 - 0.5
        x = x * stride + gx[:, :, None]  # broadcast grid
        y = y * stride + gy[:, :, None]
        
        c = 1.0 / (1.0 + np.exp(-confraw))
        
        # Flatten cells → (Ncells, 17, 3) in net-pixel space
        N = Hs * Ws
        kpts = np.empty((N, 17, 3), dtype=np.float32)
        kpts[:, :, 0] = x.reshape(N, 17)
        kpts[:, :, 1] = y.reshape(N, 17)
        kpts[:, :, 2] = c.reshape(N, 17)
        return kpts

    def _yolov8_decoding(
        self, raw_boxes, raw_kpts_heads, strides, image_dims, reg_max
    ):
        """Decode YOLOv8 raw outputs to boxes and keypoints."""
        boxes = None
        decoded_kpts = None
        net_h, net_w = image_dims

        for box_distribute, kpts_head, stride in zip(raw_boxes, raw_kpts_heads, strides):
            # Create grid
            shape = [int(x / stride) for x in image_dims]
            grid_x = np.arange(shape[1]) + 0.5
            grid_y = np.arange(shape[0]) + 0.5
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            ct_row = grid_y.flatten() * stride
            ct_col = grid_x.flatten() * stride
            center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

            # Box distribution to distance
            reg_range = np.arange(reg_max + 1)
            box_distribute = np.reshape(
                box_distribute,
                (
                    -1,
                    box_distribute.shape[1] * box_distribute.shape[2],
                    4,
                    reg_max + 1,
                ),
            )
            box_distance = self._softmax(box_distribute)
            box_distance = box_distance * np.reshape(reg_range, (1, 1, 1, -1))
            box_distance = np.sum(box_distance, axis=-1)
            box_distance = box_distance * stride

            # Decode box
            box_distance = np.concatenate(
                [box_distance[:, :, :2] * (-1), box_distance[:, :, 2:]],
                axis=-1,
            )
            decode_box = np.expand_dims(center, axis=0) + box_distance

            xmin = decode_box[:, :, 0]
            ymin = decode_box[:, :, 1]
            xmax = decode_box[:, :, 2]
            ymax = decode_box[:, :, 3]

            xywh_box = np.transpose(
                [
                    (xmin + xmax) / 2,
                    (ymin + ymax) / 2,
                    xmax - xmin,
                    ymax - ymin,
                ],
                [1, 2, 0],
            )
            boxes = (
                xywh_box
                if boxes is None
                else np.concatenate([boxes, xywh_box], axis=1)
            )

            # Decode keypoints using proper channel slicing (Hailo reference approach)
            kpts = self._decode_kpts_from_head(kpts_head, stride, net_h, net_w)
            decoded_kpts = (
                kpts
                if decoded_kpts is None
                else np.concatenate([decoded_kpts, kpts], axis=0)
            )

        return boxes, decoded_kpts

    def _nms_simple(self, boxes, scores, iou_threshold=0.45):
        """Simple NMS implementation without cython."""
        if len(boxes) == 0:
            return np.array([], dtype=np.int32)

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            iou = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return np.array(keep, dtype=np.int32)

    def _non_max_suppression(
        self,
        prediction,
        conf_thres=0.3,
        iou_thres=0.45,
        max_det=100,
        n_kpts=17,
    ):
        """Non-Maximum Suppression for YOLOv8-Pose."""
        nc = prediction.shape[2] - n_kpts * 3 - 4  # number of classes
        
        # DEBUG: Check confidence scores before filtering
        max_conf = float(prediction[..., 4].max())
        min_conf = float(prediction[..., 4].min())
        mean_conf = float(prediction[..., 4].mean())
        logger.info(f"Confidence scores - max: {max_conf:.4f}, min: {min_conf:.4f}, mean: {mean_conf:.4f}, threshold: {conf_thres}")
        
        xc = prediction[..., 4] > conf_thres  # candidates
        candidates_count = int(xc.sum())
        logger.info(f"Candidates passing {conf_thres} threshold: {candidates_count} out of {xc.size}")

        ki = 4 + nc  # keypoints start index
        output = []

        for xi, x in enumerate(prediction):
            x = x[xc[xi]]
            if not x.shape[0]:
                output.append(
                    {
                        "bboxes": np.zeros((0, 4)),
                        "keypoints": np.zeros((0, n_kpts, 3)),
                        "scores": np.zeros((0)),
                        "num_detections": 0,
                    }
                )
                continue

            # (center_x, center_y, width, height) to (x1, y1, x2, y2)
            boxes = self._xywh2xyxy(x[:, :4])
            kpts = x[:, ki:]

            conf = np.expand_dims(x[:, 4:ki].max(1), 1)
            j = np.expand_dims(x[:, 4:ki].argmax(1), 1).astype(np.float32)

            keep = np.squeeze(conf, 1) > conf_thres
            x = np.concatenate((boxes, conf, j, kpts), 1)[keep]

            # Sort by confidence
            x = x[x[:, 4].argsort()[::-1]]

            boxes = x[:, :4]
            conf = x[:, 4:5]
            preds = np.hstack(
                [boxes.astype(np.float32), conf.astype(np.float32)]
            )

            keep = self._nms_simple(preds[:, :4], preds[:, 4], iou_thres)
            if keep.shape[0] > max_det:
                keep = keep[:max_det]

            out = x[keep]
            scores = out[:, 4]
            boxes = out[:, :4]
            kpts = out[:, 6:]
            kpts = np.reshape(kpts, (-1, n_kpts, 3))

            out = {
                "bboxes": boxes,
                "keypoints": kpts,
                "scores": scores,
                "num_detections": int(scores.shape[0]),
            }
            output.append(out)

        return output

    def postprocess_output(
        self, outputs: Dict, frame_shape: Tuple[int, int]
    ) -> List[Dict]:
        """
        Postprocess Hailo YOLOv8-Pose output to extract pose keypoints.

        Args:
            outputs: Raw output from Hailo inference (9 tensors)
            frame_shape: Original frame (height, width)

        Returns:
            List of detected poses with keypoints
        """
        # YOLOv8-Pose outputs 9 tensors (3 scales × 3 outputs each):
        # Scale 1 (80×80): conv43 (bbox 64ch), conv45 (kpts 51ch), conv44 (obj 1ch)
        # Scale 2 (40×40): conv57 (bbox 64ch), conv59 (kpts 51ch), conv58 (obj 1ch)
        # Scale 3 (20×20): conv70 (bbox 64ch), conv72 (kpts 51ch), conv71 (obj 1ch)

        # Extract tensors by scale (sort by shape to get consistent ordering)
        sorted_outputs = sorted(
            outputs.items(), key=lambda x: x[1].shape[1], reverse=True
        )

        # Group by scale: [bbox, kpts, obj] for each scale
        scale_80 = [sorted_outputs[i][1] for i in [0, 1, 2]]  # 80×80
        scale_40 = [sorted_outputs[i][1] for i in [3, 4, 5]]  # 40×40
        scale_20 = [sorted_outputs[i][1] for i in [6, 7, 8]]  # 20×20

        # Identify which is bbox, kpts, obj by channel count
        def sort_by_channels(tensors):
            # bbox=64ch, kpts=51ch, obj=1ch
            sorted_t = sorted(tensors, key=lambda x: x.shape[3], reverse=True)
            return sorted_t[0], sorted_t[1], sorted_t[2]  # bbox, kpts, obj

        bbox_80, kpts_80, obj_80 = sort_by_channels(scale_80)
        bbox_40, kpts_40, obj_40 = sort_by_channels(scale_40)
        bbox_20, kpts_20, obj_20 = sort_by_channels(scale_20)

        # PROBE: Log raw head ranges to diagnose tensor layout
        def probe(name, t):
            tmin, tmax = float(t.min()), float(t.max())
            logger.info(f"PROBE {name} shape={t.shape} range=[{tmin:.3f},{tmax:.3f}]")
        
        if self._raw_debug_samples < 5:
            probe("kpts_80_head", kpts_80)
            probe("kpts_40_head", kpts_40)
            probe("kpts_20_head", kpts_20)
            probe("bbox_80_head", bbox_80)
            probe("bbox_40_head", bbox_40)
            probe("bbox_20_head", bbox_20)
            self._raw_debug_samples += 1

        # Prepare for decoding - pass raw heads, not reshaped
        raw_boxes = [bbox_20, bbox_40, bbox_80]  # Reverse order for strides
        raw_kpts_heads = [kpts_20, kpts_40, kpts_80]  # Pass raw (1, Hs, Ws, 51) heads

        # Combine objectness scores as "class scores" (person class only)
        # CRITICAL: Apply sigmoid to convert logits to probabilities [0,1]
        scores = [
            np.reshape(self._sigmoid(obj_20), (1, 20 * 20, 1)),
            np.reshape(self._sigmoid(obj_40), (1, 40 * 40, 1)),
            np.reshape(self._sigmoid(obj_80), (1, 80 * 80, 1)),
        ]
        scores = np.concatenate(scores, axis=1)

        # Decode boxes and keypoints
        # Use actual net size from self.input_shape (stored as HWC)
        net_h, net_w = int(self.input_shape[0]), int(self.input_shape[1])
        image_dims = (net_h, net_w)
        logger.info(f"DEBUG: Using network dimensions: {image_dims}")
        
        # Derive strides from actual output spatial dimensions
        def stride_from_shape(Hs):
            return net_h // Hs
        
        strides = [
            stride_from_shape(kpts_20.shape[1]),  # Should be 32
            stride_from_shape(kpts_40.shape[1]),  # Should be 16
            stride_from_shape(kpts_80.shape[1]),  # Should be 8
        ]
        logger.info(f"DEBUG: Derived strides from output shapes: {strides}")
        
        reg_max = 15  # YOLOv8 regression length

        decoded_boxes, decoded_kpts = self._yolov8_decoding(
            raw_boxes, raw_kpts_heads, strides, image_dims, reg_max
        )
        
        # SANITY CHECK: Decoded keypoint range (should be ~0..net_w, 0..net_h)
        if decoded_kpts is not None and decoded_kpts.size > 0:
            kpt_x_min = decoded_kpts[..., 0].min()
            kpt_x_max = decoded_kpts[..., 0].max()
            kpt_y_min = decoded_kpts[..., 1].min()
            kpt_y_max = decoded_kpts[..., 1].max()
            kpt_conf_min = decoded_kpts[..., 2].min()
            kpt_conf_max = decoded_kpts[..., 2].max()
            
            logger.info(f"SANITY: Decoded kpts X:[{kpt_x_min:.1f}..{kpt_x_max:.1f}] "
                       f"Y:[{kpt_y_min:.1f}..{kpt_y_max:.1f}] "
                       f"Conf:[{kpt_conf_min:.3f}..{kpt_conf_max:.3f}]")
            logger.info(f"SANITY: Expected range X:[0..{net_w}] Y:[0..{net_h}]")
            
            # Sample first keypoint (nose) from first anchor - decoded_kpts is now (Ncells, 17, 3)
            if decoded_kpts.shape[0] > 0:
                sample_kp = decoded_kpts[0, 0, :]  # [anchor, keypoint, xyz]
                logger.info(f"SANITY: Sample nose kpt: x={float(sample_kp[0]):.1f}, y={float(sample_kp[1]):.1f}, conf={float(sample_kp[2]):.3f}")

        # Reshape for NMS - decoded_kpts is now (Ncells, 17, 3), add batch dim
        batch_size = 1
        decoded_kpts = np.expand_dims(decoded_kpts, axis=0)  # (1, Ncells, 17, 3)
        decoded_kpts = np.reshape(decoded_kpts, (batch_size, -1, 51))  # (1, Ncells, 51)
        predictions = np.concatenate(
            [decoded_boxes, scores, decoded_kpts], axis=2
        )

        # Apply NMS
        nms_res = self._non_max_suppression(
            predictions, conf_thres=0.3, iou_thres=0.45, max_det=100
        )

        # Debug: Log NMS results
        total_before_nms = predictions.shape[1] if len(predictions.shape) > 1 else 0
        total_after_nms = sum(r["num_detections"] for r in nms_res)
        if total_before_nms > 0 or total_after_nms > 0:
            logger.info(f"NMS: {total_before_nms} candidates → {total_after_nms} detections")

        # Convert to our format
        detections = []
        for result in nms_res:
            if result["num_detections"] == 0:
                continue

            for bbox, kpts, score in zip(
                result["bboxes"], result["keypoints"], result["scores"]
            ):
                # Convert bbox: undo letterboxing
                # Bbox is in 640x640 letterboxed space
                x1, y1, x2, y2 = bbox
                
                # Remove padding
                x1_unpad = x1 - self.pad_x
                y1_unpad = y1 - self.pad_y
                x2_unpad = x2 - self.pad_x
                y2_unpad = y2 - self.pad_y
                
                # Scale back to original frame
                bbox_x = int(x1_unpad / self.scale)
                bbox_y = int(y1_unpad / self.scale)
                bbox_w = int((x2_unpad - x1_unpad) / self.scale)
                bbox_h = int((y2_unpad - y1_unpad) / self.scale)

                # Convert keypoints: undo letterboxing then map to frame
                # Keypoints are in 640x640 letterboxed space
                # Need to remove padding and scale back to original frame
                keypoints = []
                raw_keypoints = []
                raw_xs = []
                raw_ys = []
                
                for idx, kp in enumerate(kpts):
                    # 1. Keypoints are in 640x640 letterboxed space
                    kp_x_letterbox = kp[0]
                    kp_y_letterbox = kp[1]
                    
                    # 2. Remove letterbox padding
                    kp_x_unpadded = kp_x_letterbox - self.pad_x
                    kp_y_unpadded = kp_y_letterbox - self.pad_y
                    
                    # 3. Scale back to original frame dimensions
                    # self.scale is how much we resized the frame
                    pixel_x = int(kp_x_unpadded / self.scale)
                    pixel_y = int(kp_y_unpadded / self.scale)
                    
                    # Clamp to frame boundaries
                    pixel_x = max(0, min(pixel_x, frame_shape[1] - 1))
                    pixel_y = max(0, min(pixel_y, frame_shape[0] - 1))
                    
                    kp_conf = float(self._sigmoid(kp[2]))
                    
                    # DEBUG: Log first keypoint
                    if idx == 0:
                        msg = f"DEBUG: kp letterbox=({kp_x_letterbox:.1f},"
                        msg += f"{kp_y_letterbox:.1f})"
                        msg += f" unpad=({kp_x_unpadded:.1f},"
                        msg += f"{kp_y_unpadded:.1f})"
                        msg += f" pad=({self.pad_x},{self.pad_y})"
                        msg += f" scale={self.scale:.3f}"
                        msg += f" final=({pixel_x},{pixel_y})"
                        logger.info(msg)

                    # Collect raw keypoints for debug overlay
                    raw_keypoints.append((kp_x_letterbox, kp_y_letterbox))
                    raw_xs.append(kp_x_letterbox)
                    raw_ys.append(kp_y_letterbox)
                    keypoints.append(
                        {"x": pixel_x, "y": pixel_y, "confidence": kp_conf}
                    )

                # DEBUG: Log raw decoded keypoint range for this detection
                if raw_xs and raw_ys:
                    logger.info(f"DEBUG: Detection raw keypoints X:[{min(raw_xs):.1f}..{max(raw_xs):.1f}] Y:[{min(raw_ys):.1f}..{max(raw_ys):.1f}]")

                detections.append(
                    {
                        "bbox": (bbox_x, bbox_y, bbox_w, bbox_h),
                        "confidence": float(score),
                        "keypoints": keypoints,
                        "raw_keypoints": raw_keypoints,  # DEBUG: add raw keypoints for overlay
                    }
                )

        logger.info(
            f"Hailo post-processing: {len(detections)} detections found"
        )
        if len(detections) > 0:
            logger.info(
                f"  First detection confidence: {detections[0]['confidence']:.3f}"
            )
        return detections

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run pose detection on a single frame.

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of detected poses with keypoints
        """
        try:
            # Preprocess
            input_data = self.preprocess_frame(frame)

            # Prepare input dict
            input_dict = {self.input_name: input_data}

            # Run inference with context managers (required by Hailo API)
            with InferVStreams(
                self.network_group,
                self.input_vstreams_params,
                self.output_vstreams_params,
            ) as infer_pipeline:
                with self.network_group.activate(self.network_group_params):
                    outputs = infer_pipeline.infer(input_dict)

            # Postprocess
            detections = self.postprocess_output(outputs, frame.shape[:2])

            return detections

        except Exception as e:
            import traceback

            logger.error(f"Hailo inference error: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def release(self):
        """Release Hailo resources."""
        try:
            if self.network_group:
                self.network_group = None
            if self.device:
                self.device = None
            logger.info("✓ Hailo resources released")
        except Exception as e:
            logger.error(f"Error releasing Hailo resources: {e}")


def draw_pose(frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """
    Draw pose keypoints and skeleton on frame.

    Args:
        frame: Image to draw on
        detections: List of pose detections

    Returns:
        Frame with poses drawn
    """
    # Define skeleton connections
    skeleton = [
        (5, 6),
        (5, 7),
        (7, 9),
        (6, 8),
        (8, 10),  # Arms
        (5, 11),
        (6, 12),
        (11, 12),  # Torso
        (11, 13),
        (13, 15),
        (12, 14),
        (14, 16),  # Legs
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),  # Head
    ]

    for detection in detections:
        keypoints = detection["keypoints"]
        raw_keypoints = detection.get("raw_keypoints", [])  # DEBUG: overlay raw decoded keypoints

        # Draw keypoints (final, after unpadding/scaling)
        for kp in keypoints:
            if kp["confidence"] > 0.5:
                cv2.circle(frame, (kp["x"], kp["y"]), 3, (0, 255, 0), -1)

        # Draw raw decoded keypoints (letterbox space, before unpadding/scaling) in RED for debug
        for raw_kp in raw_keypoints:
            rx, ry = int(raw_kp[0]), int(raw_kp[1])
            cv2.circle(frame, (rx, ry), 2, (0, 0, 255), -1)

        # Draw skeleton
        for start_idx, end_idx in skeleton:
            if (
                start_idx < len(keypoints)
                and end_idx < len(keypoints)
                and keypoints[start_idx]["confidence"] > 0.5
                and keypoints[end_idx]["confidence"] > 0.5
            ):
                pt1 = (keypoints[start_idx]["x"], keypoints[start_idx]["y"])
                pt2 = (keypoints[end_idx]["x"], keypoints[end_idx]["y"])
                cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Draw bounding box
        bbox = detection["bbox"]
        cv2.rectangle(
            frame,
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
            (255, 0, 0),
            2,
        )

        # Draw confidence
        conf_text = f"{detection['confidence']:.2f}"
        cv2.putText(
            frame,
            conf_text,
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
        )

    return frame
