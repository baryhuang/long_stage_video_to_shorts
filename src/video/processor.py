#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video processing module for manipulating video frames and applying effects.
"""

import os
import cv2
import numpy as np
import logging
from typing import Tuple, List, Dict, Any, Optional, Union
from dataclasses import dataclass

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.face_detector import FaceDetector, FaceDetectorResult
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import ImageClip, CompositeVideoClip, clips_array, vfx

logger = logging.getLogger(__name__)

# Initialize MediaPipe components
mp_face_detection = mp.solutions.face_detection
mp_pose = mp.solutions.pose

@dataclass
class DetectionBox:
    """Class to store detection box coordinates."""
    x: float
    y: float
    width: float
    height: float
    score: float = 0.0
    
    @property
    def center_x(self) -> float:
        return self.x + self.width / 2
    
    @property
    def center_y(self) -> float:
        return self.y + self.height / 2
    
    @property
    def area(self) -> float:
        return self.width * self.height

def get_video_info(video_path: str) -> Dict[str, Any]:
    """
    Get basic information about a video file.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Dictionary with video info (width, height, fps, duration, etc.)
    """
    try:
        video = VideoFileClip(video_path)
        info = {
            'width': int(video.w),
            'height': int(video.h),
            'fps': video.fps,
            'duration': video.duration,
            'audio': video.audio is not None,
            'rotation': getattr(video, 'rotation', 0),
            'aspect_ratio': video.w / video.h if video.h > 0 else 0
        }
        video.close()
        return info
    except Exception as e:
        logger.error(f"Failed to get video info: {e}")
        raise

def create_face_detector():
    """
    Create a MediaPipe face detector object using the legacy solution API.
    
    Returns:
        MediaPipe face detector
    """
    # Using the legacy solution API which is more stable
    return mp_face_detection.FaceDetection(
        min_detection_confidence=0.5,
        model_selection=0  # 0 for short-range model, best for faces within 2 meters
    )

def detect_faces_in_frame(detector, frame: np.ndarray) -> List[DetectionBox]:
    """
    Detect faces in a single frame using MediaPipe.
    
    Args:
        detector: MediaPipe face detector
        frame: Video frame as numpy array (BGR format)
    
    Returns:
        List of DetectionBox objects
    """
    # Verify frame is a valid numpy array
    if frame is None:
        logger.warning("Frame is None in detect_faces_in_frame")
        return []
    
    if not isinstance(frame, np.ndarray):
        logger.warning(f"Frame is not a numpy array, type: {type(frame)}")
        return []
    
    # Check if frame has shape attribute
    try:
        h, w = frame.shape[:2]
        logger.debug(f"Frame shape: {frame.shape}")
    except (AttributeError, ValueError, IndexError) as e:
        logger.error(f"Invalid frame shape: {e}")
        return []
    
    # Convert BGR to RGB
    try:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.error(f"Error converting frame color: {e}")
        return []
    
    # Process the frame
    try:
        results = detector.process(rgb_frame)
    except Exception as e:
        logger.error(f"Error processing frame for face detection: {e}")
        return []
    
    # Convert detections to our format
    boxes = []
    
    if results.detections:
        for detection in results.detections:
            # Get bounding box
            bbox = detection.location_data.relative_bounding_box
            
            # Convert to our format
            box = DetectionBox(
                x=bbox.xmin,
                y=bbox.ymin,
                width=bbox.width,
                height=bbox.height,
                score=detection.score[0]
            )
            boxes.append(box)
    
    return boxes

def calculate_zoom_for_faces(faces: List[DetectionBox], 
                           min_zoom: float = 1.0, 
                           max_zoom: float = 1.5,
                           padding_percentage: float = 0.2) -> Tuple[float, Tuple[float, float]]:
    """
    Calculate optimal zoom factor and center point based on detected faces.
    
    Args:
        faces: List of face detections
        min_zoom: Minimum zoom factor
        max_zoom: Maximum zoom factor
        padding_percentage: Extra padding around faces as percentage of frame
    
    Returns:
        Tuple of (zoom_factor, (center_x, center_y))
    """
    if not faces:
        # No faces detected, return default values
        return min_zoom, (0.5, 0.5)
    
    # Find the centroid of all faces
    total_weight = sum(face.score for face in faces)
    if total_weight == 0:
        # If no valid scores, treat all faces equally
        center_x = sum(face.center_x for face in faces) / len(faces)
        center_y = sum(face.center_y for face in faces) / len(faces)
    else:
        # Weighted average based on detection confidence
        center_x = sum(face.center_x * face.score for face in faces) / total_weight
        center_y = sum(face.center_y * face.score for face in faces) / total_weight
    
    # Find the maximum distance from the centroid to any face edge, with padding
    max_distance_x = max(
        max(abs(face.x - center_x), abs(face.x + face.width - center_x))
        for face in faces
    ) * (1 + padding_percentage)
    
    max_distance_y = max(
        max(abs(face.y - center_y), abs(face.y + face.height - center_y))
        for face in faces
    ) * (1 + padding_percentage)
    
    # Calculate required zoom to fit all faces
    zoom_x = 0.5 / max_distance_x if max_distance_x > 0 else max_zoom
    zoom_y = 0.5 / max_distance_y if max_distance_y > 0 else max_zoom
    
    # Take the minimum zoom that fits in both dimensions
    zoom = min(zoom_x, zoom_y)
    
    # Clamp the zoom factor
    zoom = max(min_zoom, min(zoom, max_zoom))
    
    return zoom, (center_x, center_y)

def apply_zoomed_pan_effect(clip: VideoFileClip, 
                           zoom_factor: float = 1.3, 
                           center_x: float = 0.5, 
                           center_y: float = 0.5) -> VideoFileClip:
    """
    Apply a zoom and pan effect to a video clip.
    
    Args:
        clip: Original video clip
        zoom_factor: How much to zoom in (1.0 = no zoom)
        center_x: Horizontal center point (0-1, where 0.5 is center)
        center_y: Vertical center point (0-1, where 0.5 is center)
    
    Returns:
        Transformed video clip
    """
    # Make sure center is in range 0-1
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    
    # Convert from normalized coordinates to pixel offsets
    w, h = clip.size
    x_offset = w * (0.5 - center_x)
    y_offset = h * (0.5 - center_y)
    
    # Apply the zoom and pan effect
    zoomed_clip = clip.resize(zoom_factor)
    
    # Calculate the position to place the zoomed clip
    position = (int(x_offset * zoom_factor), int(y_offset * zoom_factor))
    
    # Create a composite clip with the zoomed clip positioned correctly
    composite = CompositeVideoClip([zoomed_clip.set_position(position)], size=clip.size)
    composite = composite.set_duration(clip.duration)
    
    # Preserve audio from original clip
    if clip.audio is not None:
        composite = composite.set_audio(clip.audio)
    
    return composite

def track_and_zoom_video(clip: VideoFileClip, 
                        min_zoom: float = 1.0, 
                        max_zoom: float = 1.5,
                        smoothing_window: int = 15) -> VideoFileClip:
    """
    Track faces in a video and apply smooth zoom and pan effects.
    
    Args:
        clip: Original video clip
        min_zoom: Minimum zoom factor
        max_zoom: Maximum zoom factor
        smoothing_window: Number of frames to average for smoothing
    
    Returns:
        Processed video clip with smart zooming
    """
    # Create a face detector
    detector = create_face_detector()
    
    # Initialize caches
    track_cache = {
        'frames_processed': 0,
        'zoom_history': [],
        'center_x_history': [],
        'center_y_history': []
    }
    
    # Pre-initialize the history with default values
    default_zoom = (min_zoom + max_zoom) / 2
    track_cache['zoom_history'] = [default_zoom] * smoothing_window
    track_cache['center_x_history'] = [0.5] * smoothing_window
    track_cache['center_y_history'] = [0.5] * smoothing_window
    
    # Define the simple processing function for each frame that only applies zoom and tracking
    # without face detection, using the parameters directly
    def simple_process_frame(frame):
        try:
            h, w = frame.shape[:2]
            
            # Use vertical_position_ratio of 0.5 (center) when not detecting faces
            center_x = 0.5  # Center horizontally
            center_y = 0.5  # Center vertically
            
            # Calculate crop region
            crop_w = w / min_zoom
            crop_h = h / min_zoom
            
            # Convert center from 0-1 range to pixel coordinates
            center_x_px = int(w * center_x)
            center_y_px = int(h * center_y)
            
            # Calculate crop region coordinates
            x1 = max(0, int(center_x_px - crop_w / 2))
            y1 = max(0, int(center_y_px - crop_h / 2))
            x2 = min(w, int(x1 + crop_w))
            y2 = min(h, int(y1 + crop_h))
            
            # Adjust x1/y1 if x2/y2 are constrained
            if x2 == w:
                x1 = max(0, int(w - crop_w))
            if y2 == h:
                y1 = max(0, int(h - crop_h))
            
            # Crop the frame
            cropped = frame[y1:y2, x1:x2]
            
            # Resize back to original size
            result = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
            
            return result
        except Exception as e:
            logger.error(f"Error in simple_process_frame: {e}")
            return frame
    
    # Apply the simple processing function to the clip with error handling
    try:
        # Use fl_image for simple frame-by-frame processing without time parameter
        processed_clip = clip.fl_image(simple_process_frame)
        
        # Ensure audio is preserved
        if clip.audio is not None:
            processed_clip = processed_clip.set_audio(clip.audio)
            
        return processed_clip
    except Exception as e:
        logger.error(f"Error in track_and_zoom_video: {e}")
        logger.info("Returning original clip due to processing error")
        return clip 