#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video preview module for generating preview images for video effects.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from moviepy.video.io.VideoFileClip import VideoFileClip

from src.models.font import FontManager
from src.video.processor import track_and_zoom_video

logger = logging.getLogger(__name__)

def create_zoom_preview(
    input_path: str,
    timestamp: float,
    zoom_factor: float,
    vertical_position_ratio: float,
    output_preview_path: str
) -> bool:
    """
    Create a preview image showing the zoom and vertical crop settings.
    Shows both the original frame and the processed frame side by side.
    
    Args:
        input_path: Path to input video
        timestamp: Timestamp to use for preview frame
        zoom_factor: Zoom factor to apply
        vertical_position_ratio: Ratio for vertical positioning
        output_preview_path: Path to save the preview image
    
    Returns:
        bool: True if user confirms the settings, False otherwise
    """
    logger.info(f"Creating zoom preview using frame at {timestamp:.2f}s")
    
    # Initialize font manager
    font_manager = FontManager()
    font_path = font_manager.get_font_path()
    
    # Load video and extract frame
    video = VideoFileClip(input_path)
    frame = video.get_frame(timestamp)
    video.close()
    
    # Create a temporary clip for the single frame
    frame_clip = VideoFileClip(input_path).subclip(timestamp, timestamp + 0.1)
    
    # Process the frame with zoom and tracking
    processed_clip = track_and_zoom_video(
        frame_clip,
        min_zoom=zoom_factor,
        max_zoom=zoom_factor,
        smoothing_window=15,
        vertical_position_ratio=vertical_position_ratio
    )
    processed_frame = processed_clip.get_frame(0)
    processed_clip.close()
    frame_clip.close()
    
    # Convert frames to PIL Images
    original_image = Image.fromarray(frame)
    processed_image = Image.fromarray(processed_frame)
    
    # Calculate dimensions for side-by-side comparison
    max_height = 800  # Maximum height for preview
    scale_factor = max_height / max(original_image.height, processed_image.height)
    
    # Resize images while maintaining aspect ratio
    new_width_orig = int(original_image.width * scale_factor)
    new_height_orig = int(original_image.height * scale_factor)
    new_width_proc = int(processed_image.width * scale_factor)
    new_height_proc = int(processed_image.height * scale_factor)
    
    original_image = original_image.resize((new_width_orig, new_height_orig), Image.Resampling.LANCZOS)
    processed_image = processed_image.resize((new_width_proc, new_height_proc), Image.Resampling.LANCZOS)
    
    # Create a new image with both frames side by side
    margin = 20  # Margin between images and for text
    text_height = 40  # Height for text
    info_height = 80  # Height for additional information text
    preview_width = new_width_orig + new_width_proc + 3 * margin
    preview_height = max(new_height_orig, new_height_proc) + 2 * margin + text_height + info_height
    
    preview = Image.new('RGB', (preview_width, preview_height), 'black')
    
    # Paste original frame on the left
    x_offset = margin
    y_offset = margin + text_height
    preview.paste(original_image, (x_offset, y_offset))
    
    # Paste processed frame on the right
    x_offset = margin * 2 + new_width_orig
    preview.paste(processed_image, (x_offset, y_offset))
    
    # Add text labels
    draw = ImageDraw.Draw(preview)
    font = ImageFont.truetype(font_path, 20)  # Use our custom font
    small_font = ImageFont.truetype(font_path, 16)  # Smaller font for additional info
    
    # Add labels
    draw.text((margin, margin), "原始影像", fill='white', font=font)  # "Original Frame" in Traditional Chinese
    draw.text((margin * 2 + new_width_orig, margin), 
             f"處理後影像 (縮放: {zoom_factor}x, 垂直位置: {vertical_position_ratio:.2f})",  # "Processed Frame" in Traditional Chinese
             fill='white', font=font)
    
    # Add information about no person detection behavior
    info_y = margin + text_height + max(new_height_orig, new_height_proc) + margin
    info_text = "注意: 當畫面中沒有檢測到人時，將自動取消縮放效果，恢復原始畫面。"  # Note about no person detection in Traditional Chinese
    draw.text((preview_width // 2, info_y), info_text, fill='#FFFF00', font=font, anchor='mm')
    
    # Add additional explanation
    info_y += 30
    info_text2 = "系統會在人物消失後約半秒內平滑過渡到原始畫面，人物重新出現時會恢復縮放效果。"  # Explanation in Traditional Chinese
    draw.text((preview_width // 2, info_y), info_text2, fill='#FFFF00', font=small_font, anchor='mm')
    
    # Save preview
    preview.save(output_preview_path, 'JPEG', quality=95)
    logger.info(f"Saved zoom preview to {output_preview_path}")
    
    # Show preview and get confirmation
    print("\n縮放和裁剪預覽:")  # "Zoom and Crop Preview" in Traditional Chinese
    print("-" * 50)
    print(f"預覽圖片已保存至: {output_preview_path}")  # "Preview image has been saved to" in Traditional Chinese
    print(f"縮放倍率: {zoom_factor}x")  # "Zoom factor" in Traditional Chinese
    print(f"垂直位置比例: {vertical_position_ratio:.2f}")  # "Vertical position ratio" in Traditional Chinese
    print("\n注意: 當畫面中沒有檢測到人時，將自動取消縮放效果，恢復原始畫面。")  # Note about no person detection
    print("系統會在人物消失後約半秒內平滑過渡到原始畫面，人物重新出現時會恢復縮放效果。")  # Explanation
    print("\n請檢查預覽圖片，確認是否要使用這些設置繼續。")  # "Please check the preview image..." in Traditional Chinese
    
    while True:
        choice = input("\n您想要:\n1. 使用這些設置繼續\n2. 退出\n請輸入您的選擇 (1/2): ").strip()  # Menu options in Traditional Chinese
        if choice in ['1', '2']:
            break
        print("無效的選擇。請輸入 1 或 2。")  # "Invalid choice" in Traditional Chinese
    
    return choice == '1' 