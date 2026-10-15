"""
Debug visualization for motion-based seizure detection.

Provides visual debugging tools to understand motion patterns and model behavior.
"""
import cv2
import numpy as np
from typing import Dict, Optional
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class MotionVisualizer:
    """
    Visualizes motion features for debugging and analysis.
    
    Provides multiple visualization modes:
    - Optical flow vectors
    - Motion heatmap
    - Motion density graph
    - Statistics overlay
    """
    
    def __init__(self, display_width: int = 640, display_height: int = 480):
        """
        Initialize motion visualizer.
        
        Args:
            display_width: Width of visualization output
            display_height: Height of visualization output
        """
        self.display_width = display_width
        self.display_height = display_height
        
        # Color schemes
        self.colors = {
            'flow_vector': (0, 255, 0),      # Green arrows
            'motion_hot': (0, 0, 255),       # Red for high motion
            'motion_medium': (0, 165, 255),  # Orange for medium
            'motion_low': (0, 255, 255),     # Yellow for low
            'text_bg': (0, 0, 0),            # Black background for text
            'text_fg': (255, 255, 255),      # White text
            'graph_line': (0, 255, 0),       # Green line
            'threshold_line': (0, 0, 255),   # Red threshold line
        }
        
    def draw_motion_overlay(self, 
                           frame: np.ndarray,
                           motion_features: Dict[str, np.ndarray],
                           motion_history: list = None,
                           mode: str = 'heatmap') -> np.ndarray:
        """
        Draw motion visualization overlay on frame.
        
        Args:
            frame: Original frame (BGR)
            motion_features: Dictionary from MotionFeatureExtractor.process_frame()
            motion_history: List of motion magnitude values over time
            mode: Visualization mode ('heatmap', 'vectors', 'mask', 'all')
            
        Returns:
            Frame with motion overlay
        """
        result = frame.copy()
        
        if mode in ['heatmap', 'all']:
            result = self._draw_motion_heatmap(result, motion_features)
        
        if mode in ['vectors', 'all']:
            result = self._draw_flow_vectors(result, motion_features)
        
        if mode in ['mask', 'all']:
            result = self._draw_motion_mask(result, motion_features)
        
        # Always draw stats
        # result = self._draw_stats(result, motion_features)
        
        # Draw mini graph at bottom if history provided
        if motion_history and len(motion_history) > 1:
            result = self._draw_mini_graph(result, motion_history)
        
        return result
    
    def _draw_motion_heatmap(self, 
                            frame: np.ndarray,
                            motion_features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Overlay motion magnitude as colored heatmap.
        """
        magnitude = motion_features['flow_magnitude']
        
        # Normalize magnitude to 0-255
        mag_normalized = np.clip(magnitude * 50, 0, 255).astype(np.uint8)
        
        # Apply colormap (RAINBOW: blue=no motion, red=high motion)
        heatmap = cv2.applyColorMap(mag_normalized, cv2.COLORMAP_JET)
        
        # Resize heatmap to match frame size if needed
        if heatmap.shape[:2] != frame.shape[:2]:
            heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        
        # Blend with original frame (30% heatmap, 70% original)
        result = cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
        
        return result
    
    def _draw_flow_vectors(self,
                          frame: np.ndarray,
                          motion_features: Dict[str, np.ndarray],
                          step: int = 16) -> np.ndarray:
        """
        Draw optical flow as arrows.
        
        Args:
            frame: Frame to draw on
            motion_features: Motion features
            step: Spacing between arrows (pixels)
        """
        flow_x = motion_features['flow_x']
        flow_y = motion_features['flow_y']
        magnitude = motion_features['flow_magnitude']
        
        h, w = flow_x.shape
        
        # Draw arrows on a grid
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx = flow_x[y, x]
                fy = flow_y[y, x]
                mag = magnitude[y, x]
                
                # Only draw if motion is significant
                if mag > 0.5:
                    # Scale arrow length
                    scale = 2.0
                    end_x = int(x + fx * scale)
                    end_y = int(y + fy * scale)
                    
                    # Color by magnitude
                    if mag > 3.0:
                        color = self.colors['motion_hot']
                    elif mag > 1.5:
                        color = self.colors['motion_medium']
                    else:
                        color = self.colors['motion_low']
                    
                    cv2.arrowedLine(frame, (x, y), (end_x, end_y), 
                                  color, 1, tipLength=0.3)
        
        return frame
    
    def _draw_motion_mask(self,
                         frame: np.ndarray,
                         motion_features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Highlight moving regions with overlay.
        """
        motion_mask = motion_features['motion_mask']
        
        # Resize mask to match frame if needed
        if motion_mask.shape[:2] != frame.shape[:2]:
            motion_mask = cv2.resize(motion_mask, (frame.shape[1], frame.shape[0]))
        
        # Create colored overlay (semi-transparent green)
        overlay = frame.copy()
        overlay[motion_mask > 0] = [0, 255, 0]  # Green
        
        # Blend with original
        result = cv2.addWeighted(frame, 0.85, overlay, 0.15, 0)
        
        return result
    
    def _draw_stats(self,
                   frame: np.ndarray,
                   motion_features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Draw motion statistics as text overlay.
        """
        stats = motion_features['stats']
        
        # Prepare text lines
        lines = [
            f"Motion: {stats['avg_magnitude']:.2f} avg, {stats['max_magnitude']:.2f} max",
            f"Density: {stats['motion_density']*100:.1f}% ({stats['motion_area']} px)",
        ]
        
        # Add temporal stats if available
        if 'magnitude_std' in stats:
            lines.append(f"Variability: {stats['magnitude_std']:.2f}")
        if 'magnitude_trend' in stats:
            trend_str = "↑ increasing" if stats['magnitude_trend'] > 0.01 else \
                       "↓ decreasing" if stats['magnitude_trend'] < -0.01 else \
                       "→ stable"
            lines.append(f"Trend: {trend_str}")
        
        # Draw semi-transparent background
        text_height = 25
        padding = 10
        bg_height = len(lines) * text_height + padding * 2
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 10 + bg_height),
                     self.colors['text_bg'], -1)
        frame = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
        
        # Draw text lines
        y_offset = 30
        for line in lines:
            cv2.putText(frame, line, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                       self.colors['text_fg'], 2)
            y_offset += text_height
        
        return frame
    
    def _draw_mini_graph(self,
                        frame: np.ndarray,
                        motion_history: list) -> np.ndarray:
        """
        Draw a small motion magnitude graph at the bottom of the frame.
        
        Args:
            frame: Frame to draw on
            motion_history: List of recent motion magnitude values
            
        Returns:
            Frame with mini graph
        """
        h, w = frame.shape[:2]
        graph_height = 40
        graph_y_start = h - graph_height
        
        # Create semi-transparent background for graph
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, graph_y_start), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Prepare data (use last 100 values or frame width, whichever is less)
        max_points = min(len(motion_history), w)
        if max_points < 2:
            return frame
        
        values = list(motion_history[-max_points:])
        
        # Normalize values to graph height (leave 5px padding top/bottom)
        max_val = max(max(values), 1.0)  # Minimum scale of 1
        padding = 5
        scale_height = graph_height - (2 * padding)
        
        # Draw graph line
        points = []
        step = w / max(max_points - 1, 1)
        for i, val in enumerate(values):
            x = int(i * step)
            # Invert y (higher values = higher on screen)
            y = int(graph_y_start + graph_height - padding - (val / max_val) * scale_height)
            points.append((x, y))
        
        # Draw lines connecting points
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i+1], self.colors['graph_line'], 1, cv2.LINE_AA)
        
        # Draw scale indicator (max value)
        cv2.putText(frame, f"{max_val:.1f}", (w - 40, graph_y_start + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def draw_hr_overlay(self,
                        frame: np.ndarray,
                        hr_state: Dict) -> np.ndarray:
        """
        Draw a 40-px heart-rate strip directly above the motion mini-graph.

        Stacks at h-80..h-40 so it sits above `_draw_mini_graph` (h-40..h).
        Plots BPM history, baseline and threshold reference lines, current
        BPM/baseline numeric label, and a sustain-progress bar on the left
        edge. Color-coded by detector state.

        Args:
            frame: Frame to draw on.
            hr_state: dict with keys: enabled, connected, stale, bpm,
                baseline_bpm, above_baseline, above_absolute, sustained_sec,
                recent (list of (timestamp, bpm)), delta_threshold,
                absolute_threshold, sustain_sec.
        """
        if not hr_state or not hr_state.get('enabled'):
            return frame

        h, w = frame.shape[:2]
        strip_height = 40
        # Sit directly above the motion mini-graph (which lives at h-40..h).
        y0 = h - 80
        y1 = h - 40

        # Fixed BPM display range; clip out-of-range.
        bpm_min, bpm_max = 50, 130
        padding = 4
        plot_h = strip_height - 2 * padding

        # Semi-transparent black band; tinted red while alerting.
        overlay = frame.copy()
        bg_color = (0, 0, 0)
        if (hr_state.get('above_baseline') or hr_state.get('above_absolute')) \
                and hr_state.get('sustained_sec', 0.0) >= hr_state.get('sustain_sec', 20):
            bg_color = (0, 0, 60)  # dark red wash
        cv2.rectangle(overlay, (0, y0), (w, y1), bg_color, -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        def bpm_to_y(bpm: float) -> int:
            clipped = max(bpm_min, min(bpm_max, bpm))
            frac = (clipped - bpm_min) / (bpm_max - bpm_min)
            return int(y1 - padding - frac * plot_h)

        # Reference lines (only when baseline known).
        baseline = hr_state.get('baseline_bpm')
        delta_th = hr_state.get('delta_threshold', 15)
        abs_th = hr_state.get('absolute_threshold', 100)

        # Absolute floor — dashed red.
        y_abs = bpm_to_y(abs_th)
        for x in range(0, w, 8):
            cv2.line(frame, (x, y_abs), (x + 4, y_abs), (60, 60, 200), 1)

        if baseline is not None:
            # Baseline — solid faint white.
            y_base = bpm_to_y(baseline)
            cv2.line(frame, (0, y_base), (w, y_base), (180, 180, 180), 1)
            # Baseline + delta — dashed amber.
            y_delta = bpm_to_y(baseline + delta_th)
            for x in range(0, w, 8):
                cv2.line(frame, (x, y_delta), (x + 4, y_delta), (0, 165, 220), 1)

        # Pick line color by state.
        connected = hr_state.get('connected') and not hr_state.get('stale')
        if not connected:
            line_color = (160, 160, 160)  # grey
        elif hr_state.get('sustained_sec', 0.0) >= hr_state.get('sustain_sec', 20):
            line_color = (50, 50, 220)    # red — alerting
        elif hr_state.get('above_baseline') or hr_state.get('above_absolute'):
            line_color = (0, 165, 255)    # amber — elevated, not yet sustained
        else:
            line_color = (80, 200, 80)    # green — normal

        # Plot BPM history.
        recent = hr_state.get('recent') or []
        if len(recent) >= 2:
            n = len(recent)
            step = w / max(n - 1, 1)
            points = [
                (int(i * step), bpm_to_y(bpm))
                for i, (_, bpm) in enumerate(recent)
            ]
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i + 1], line_color, 1, cv2.LINE_AA)

        # Sustain progress bar on the left edge (only when elevated).
        sustained = hr_state.get('sustained_sec', 0.0)
        sustain_target = hr_state.get('sustain_sec', 20)
        if sustained > 0 and sustain_target > 0:
            frac = min(sustained / sustain_target, 1.0)
            bar_h = int(plot_h * frac)
            cv2.rectangle(
                frame,
                (0, y1 - padding - bar_h),
                (3, y1 - padding),
                line_color,
                -1,
            )

        # Numeric label top-left.
        if hr_state.get('bpm') is None or not connected:
            label = "HR --"
        else:
            label = f"HR {hr_state['bpm']}"
            if baseline is not None:
                label += f"  base {baseline:.0f}"
        cv2.putText(frame, label, (8, y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        return frame

    def draw_motion_graph(self,
                         motion_history: list,
                         width: int = 640,
                         height: int = 150,
                         threshold: Optional[float] = None) -> np.ndarray:
        """
        Draw motion magnitude over time as a graph.
        
        Args:
            motion_history: List of motion magnitude values
            width: Graph width
            height: Graph height
            threshold: Optional threshold line to draw
            
        Returns:
            Graph image
        """
        # Create blank graph
        graph = np.zeros((height, width, 3), dtype=np.uint8)
        
        if len(motion_history) < 2:
            return graph
        
        # Normalize values to graph height
        values = np.array(motion_history)
        max_val = max(np.max(values), 5.0)  # Minimum scale of 5
        normalized = (values / max_val) * (height - 20) + 10
        
        # Draw grid lines
        for i in range(0, height, 30):
            cv2.line(graph, (0, i), (width, i), (40, 40, 40), 1)
        
        # Draw graph line
        points = []
        step = width / max(len(motion_history) - 1, 1)
        for i, val in enumerate(normalized):
            x = int(i * step)
            y = int(height - val)
            points.append((x, y))
        
        for i in range(len(points) - 1):
            cv2.line(graph, points[i], points[i+1], 
                    self.colors['graph_line'], 2)
        
        # Draw threshold line if provided
        if threshold is not None:
            threshold_y = int(height - (threshold / max_val) * (height - 20) - 10)
            cv2.line(graph, (0, threshold_y), (width, threshold_y),
                    self.colors['threshold_line'], 2, cv2.LINE_AA)
            cv2.putText(graph, f"Threshold: {threshold:.1f}", (10, threshold_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['threshold_line'], 1)
        
        # Draw axis labels
        cv2.putText(graph, "Motion Magnitude Over Time", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(graph, "0", (5, height - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(graph, f"{max_val:.1f}", (5, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return graph
    
    def create_debug_view(self,
                         frame: np.ndarray,
                         motion_features: Dict[str, np.ndarray],
                         motion_history: list,
                         detection_result: Optional[Dict] = None) -> np.ndarray:
        """
        Create comprehensive debug view with multiple visualizations.
        
        Args:
            frame: Original frame
            motion_features: Current frame motion features
            motion_history: Historical motion magnitude values
            detection_result: Optional detection results to display
            
        Returns:
            Debug view image combining multiple visualizations
        """
        # Main view with heatmap overlay
        main_view = self.draw_motion_overlay(frame, motion_features, mode='heatmap')
        main_view = cv2.resize(main_view, (self.display_width, self.display_height))
        
        # Motion graph
        graph = self.draw_motion_graph(motion_history, width=self.display_width, height=150)
        
        # Combine views vertically
        debug_view = np.vstack([main_view, graph])
        
        # Add detection result if provided
        if detection_result:
            self._draw_detection_result(debug_view, detection_result)
        
        return debug_view
    
    def _draw_detection_result(self,
                              frame: np.ndarray,
                              detection_result: Dict):
        """Draw detection result overlay."""
        probability = detection_result.get('probability', 0.0)
        is_seizure = detection_result.get('is_seizure', False)
        
        # Draw result box
        h = frame.shape[0]
        box_y = h - 200
        
        overlay = frame.copy()
        color = (0, 0, 255) if is_seizure else (0, 255, 0)
        cv2.rectangle(overlay, (10, box_y), (300, h - 10), color, -1)
        frame[:] = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        # Draw text
        status = "⚠️ SEIZURE DETECTED" if is_seizure else "✓ Normal"
        cv2.putText(frame, status, (20, box_y + 40),
                   cv2.FONT_HERSHEY_BOLD, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Confidence: {probability*100:.1f}%", (20, box_y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
