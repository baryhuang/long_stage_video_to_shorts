#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Highlight video generator that extracts engaging segments from longer videos.
The output is in 9:16 portrait format with traditional Chinese titles and subtitles.
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
from dotenv import load_dotenv
import assemblyai as aai
from anthropic import Anthropic
import mediapipe as mp  # Add MediaPipe import
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.VideoClip import TextClip, ColorClip, ImageClip
from moviepy.video.compositing.CompositeVideoClip import CompositeVideoClip
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.VideoClip import VideoClip
from moviepy.audio.io.AudioFileClip import AudioFileClip  # Add this import
from tqdm import tqdm
import zhconv  # For converting between Chinese variants
import requests
import openai
from PIL import Image, ImageDraw, ImageFont
import jieba  # Add this to the imports at the top

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API keys from environment
ASSEMBLY_API_KEY = os.getenv("ASSEMBLY_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not ASSEMBLY_API_KEY:
    logger.error("AssemblyAI API key not found. Please set the ASSEMBLY_API_KEY environment variable.")
    sys.exit(1)

if not ANTHROPIC_API_KEY:
    logger.error("Anthropic API key not found. Please set the ANTHROPIC_API_KEY environment variable.")
    sys.exit(1)

if not OPENAI_API_KEY:
    logger.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

# Initialize API clients
aai.settings.api_key = ASSEMBLY_API_KEY
anthropic_client = Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = openai.Client(api_key=OPENAI_API_KEY)

class TranscriptionSegment:
    """Represents a transcription segment with timing information and word-level timestamps"""
    def __init__(self, text: str, start: float, end: float, words: Optional[List[Dict[str, Any]]] = None, speaker: Optional[str] = None):
        self.text = text
        self.start = start
        self.end = end
        self.duration = end - start
        self.words = words or []  # List of word timestamps {text, start, end}
        self.speaker = speaker  # Add speaker field

    def __str__(self):
        speaker_info = f"[Speaker {self.speaker}] " if self.speaker else ""
        return f"{self.start:.2f}s - {self.end:.2f}s: {speaker_info}{self.text}"

class HighlightClip:
    """Represents a potential highlight clip with score and timing information"""
    def __init__(self, start: float, end: float, score: float, title: str):
        self.start = start
        self.end = end
        self.duration = end - start
        self.score = score
        self.title = title

    def __str__(self):
        return f"Highlight: {self.start:.2f}s - {self.end:.2f}s ({self.duration:.2f}s) | Score: {self.score} | Title: {self.title}"

class FontManager:
    def __init__(self):
        self.font_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        self.font_path = os.path.join(self.font_dir, 'NotoSansTC-Regular.otf')
        self._ensure_font_exists()
    
    def _ensure_font_exists(self):
        """Ensure the Chinese font exists, download if not present."""
        if not os.path.exists(self.font_dir):
            os.makedirs(self.font_dir)
        
        if not os.path.exists(self.font_path):
            logger.info("Downloading Chinese font...")
            font_url = 'https://fonts.gstatic.com/s/notosanstc/v26/-nF7OG829Oofr2wohFbTp9iFOSsLA_ZJ1g.otf'
            try:
                response = requests.get(font_url)
                response.raise_for_status()
                with open(self.font_path, 'wb') as f:
                    f.write(response.content)
                # Verify font file
                try:
                    ImageFont.truetype(self.font_path, 16)
                    logger.info("Font downloaded and verified successfully!")
                except Exception as e:
                    logger.error(f"Downloaded font file is invalid: {e}")
                    if os.path.exists(self.font_path):
                        os.remove(self.font_path)
                    raise
            except Exception as e:
                logger.error(f"Failed to download font: {e}")
                raise
    
    def get_font_path(self):
        """Get the path to the Chinese font."""
        return self.font_path

def transcribe_video(video_path: str, language_code: str = "zh") -> List[TranscriptionSegment]:
    """
    Transcribe the video or audio file using AssemblyAI and return segments with timing information.
    First extracts audio from video if needed, then transcribes the audio.
    Uses word-level timestamps for more precise timing information.
    If a transcript file already exists, load it instead of re-transcribing.
    
    Args:
        video_path: Path to the video or audio file
        language_code: Language code (zh for Traditional Chinese)
    
    Returns:
        List of TranscriptionSegment objects
    """
    # Create transcript file path
    video_path_obj = Path(video_path)
    transcript_path = video_path_obj.with_suffix('.transcript.json')
    
    # Check if transcript file exists
    if transcript_path.exists():
        logger.info(f"Loading existing transcript from {transcript_path}")
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                transcript_data = json.load(f)
            
            # Reconstruct segments from saved data
            segments = []
            for seg_data in transcript_data:
                segments.append(TranscriptionSegment(
                    text=seg_data['text'],
                    start=seg_data['start'],
                    end=seg_data['end'],
                    words=seg_data.get('words', []),  # Include word-level timestamps
                    speaker=seg_data.get('speaker')  # Load speaker information
                ))
            logger.info(f"Loaded {len(segments)} segments from existing transcript")
            return segments
        except Exception as e:
            logger.warning(f"Failed to load existing transcript: {e}. Will re-transcribe.")
    
    logger.info(f"Processing file: {video_path}")
    
    # Extract audio to temporary file
    temp_audio_path = video_path_obj.with_suffix('.temp.wav')
    try:
        # Check if input is video or audio
        is_video = video_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.webm'))
        
        if is_video:
            logger.info("Input is video, extracting audio...")
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(str(temp_audio_path), codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
            video.close()
        else:
            logger.info("Input is audio, converting to WAV format...")
            audio = AudioFileClip(video_path)
            audio.write_audiofile(str(temp_audio_path), codec='pcm_s16le', ffmpeg_params=["-ac", "1"])
            audio.close()
        
        logger.info(f"Transcribing audio with speaker diarization...")
        
        # Create the transcriber with configuration
        config = aai.TranscriptionConfig(
            language_code=language_code,
            speaker_labels=True,  # Enable speaker diarization
            punctuate=True,
            format_text=True
        )
        transcriber = aai.Transcriber()
        
        # Start the transcription with audio file
        transcript = transcriber.transcribe(str(temp_audio_path), config=config)
        
        # Show progress bar while waiting for transcription
        with tqdm(total=100, desc="Transcribing") as pbar:
            last_progress = 0
            while True:
                status = transcript.status
                
                # Calculate progress percentage based on status
                if status == "queued":
                    progress = 5
                elif status == "processing":
                    progress = 50
                elif status == "completed":
                    progress = 100
                    break
                elif status == "error":
                    raise Exception(f"Transcription failed with status: {status}")
                
                # Update progress bar
                if progress > last_progress:
                    pbar.update(progress - last_progress)
                    last_progress = progress
                
                time.sleep(3)  # Wait 3 seconds before checking again
        
        # Process utterances with speaker information
        segments = []
        
        # First, process utterances to get speaker information
        logger.info("Processing speaker diarization results...")
        for utterance in transcript.utterances:
            # Convert text to traditional Chinese
            text = zhconv.convert(utterance.text, 'zh-hant')
            
            # Create segment with speaker information
            segment = TranscriptionSegment(
                text=text,
                start=utterance.start / 1000,  # Convert ms to seconds
                end=utterance.end / 1000,      # Convert ms to seconds
                speaker=utterance.speaker,
                words=[{
                    'text': word.text,
                    'start': word.start / 1000,
                    'end': word.end / 1000
                } for word in utterance.words] if hasattr(utterance, 'words') else []
            )
            segments.append(segment)
        
        logger.info(f"Found {len(segments)} segments with {len(set(seg.speaker for seg in segments))} unique speakers")
        
        # Save transcript to file with speaker information
        logger.info(f"Saving transcript to {transcript_path}")
        transcript_data = [
            {
                'text': seg.text,
                'start': seg.start,
                'end': seg.end,
                'speaker': seg.speaker,
                'words': seg.words
            }
            for seg in segments
        ]
        with open(transcript_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Transcription complete: {len(segments)} segments")
        return segments
    
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise
    
    finally:
        # Clean up temporary audio file
        if temp_audio_path.exists():
            temp_audio_path.unlink()
            logger.info("Cleaned up temporary audio file")

def identify_highlights(segments: List[TranscriptionSegment], max_duration: float = 120.0) -> List[Dict[str, Any]]:
    """
    Two-step approach:
    1. Use OpenAI o3-mini to identify one continuous engaging segment of ~120 seconds
    2. Use Claude to format the response into proper JSON
    
    Args:
        segments: List of transcription segments
        max_duration: Target duration for the highlight clip in seconds (default: 120s)
    
    Returns:
        List of dictionaries containing segment information
    """
    logger.info("Step 1: Identifying one continuous engaging segment using OpenAI o3-mini")
    
    # Format the transcription for OpenAI
    transcript_text = "\n".join([f"{i+1}. {seg.start:.2f}s - {seg.end:.2f}s: {seg.text}" for i, seg in enumerate(segments)])
    
    # Save transcript text for debugging
    debug_file = Path("openai_debug.log")
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write("=== Transcript Text ===\n")
        f.write(transcript_text)
        f.write("\n\n")
    
    try:
        # Log OpenAI request
        logger.debug("Sending request to OpenAI API")
        logger.debug("Request payload:")
        request_payload = {
            "model": "o3-mini",
            "messages": [
                {
                    "role": "developer",
                    "content": "You are a video editing expert. Your task is to analyze video transcripts and identify one continuous engaging segment that is approximately 120 seconds long. Focus on finding a segment that is self-contained, has natural start/end points, and maintains narrative coherence. Do not include double quotes in the response."
                },
                {
                    "role": "system",
                    "content": "Analyze the transcript and identify one continuous engaging segment that is approximately 120 seconds long. Ensure the segment has clear start/end times and maintains context."
                },
                {
                    "role": "user",
                    "content": f"""以下是視頻的完整逐字稿（包含時間戳）：

{transcript_text}

請仔細分析這些內容，然後：
1. 先看整個視頻，了解整個視頻的內容和主題
2. 找出一個大約120秒長的連續精彩片段
3. 片段的開始和結束要在自然的開始和結束，不要牽強或者突然地開始和結束

請列出：
1. 整個視頻的主題
2. 精彩片段的：
   - 開始時間（秒）
   - 結束時間（秒）
   - 內容摘要
   - 選擇原因"""
                }
            ],
        }
        logger.debug(json.dumps(request_payload, ensure_ascii=False, indent=2))
        
        # Append request to debug file
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write("=== OpenAI Request ===\n")
            f.write(json.dumps(request_payload, ensure_ascii=False, indent=2))
            f.write("\n\n")
        
        # Call OpenAI API
        openai_response = openai_client.chat.completions.create(**request_payload)
        
        # Log OpenAI response
        logger.debug("OpenAI API Response:")
        logger.debug(str(openai_response))
        
        # Append response to debug file
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write("=== OpenAI Response ===\n")
            f.write(str(openai_response))
            f.write("\n\n")
        
        # Get OpenAI's analysis
        openai_analysis = openai_response.choices[0].message.content
        logger.debug("OpenAI Analysis:")
        logger.debug(openai_analysis)
        
        # Append analysis to debug file
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write("=== OpenAI Analysis ===\n")
            f.write(openai_analysis)
            f.write("\n\n")
        
        logger.info("Step 2: Formatting response using Claude")
        
        # Prepare prompt for Claude to format the response
        claude_prompt = f"""Please convert the following video analysis into a properly formatted JSON structure. Escape all double quotes in the JSON.

Analysis from OpenAI:
{openai_analysis}

Please format it as:
```json
{{
  "segments": [
    {{
      "start_time": start_time_in_seconds,
      "end_time": end_time_in_seconds,
      "content": "content_summary",
      "reason": "selection_reason"
    }}
  ],
  "theme": "video_theme"
}}
```

Ensure all times are in seconds (floating point numbers) and the content and reasons are in Traditional Chinese."""

        # Log Claude prompt
        logger.debug("Claude Prompt:")
        logger.debug(claude_prompt)
        
        # Append Claude prompt to debug file
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write("=== Claude Prompt ===\n")
            f.write(claude_prompt)
            f.write("\n\n")
        
        # Get response from Claude
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            temperature=0.2,
            max_tokens=1000,
            system="You are a helpful assistant that converts video analysis into properly formatted JSON. Always maintain Traditional Chinese text in the output.",
            messages=[{"role": "user", "content": claude_prompt}]
        )
        
        # Log Claude response
        response_text = response.content[0].text
        logger.debug("Claude Response:")
        logger.debug(response_text)
        
        # Append Claude response to debug file
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write("=== Claude Response ===\n")
            f.write(response_text)
            f.write("\n\n")
        
        # Extract JSON from the response
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            
            # Log final formatted data
            logger.debug("Final Formatted Data:")
            logger.debug(json.dumps(data, ensure_ascii=False, indent=2))
            
            # Append final data to debug file
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write("=== Final Formatted Data ===\n")
                f.write(json.dumps(data, ensure_ascii=False, indent=2))
                f.write("\n\n")
            
            return data
        else:
            error_msg = "Failed to extract JSON from Claude response"
            logger.error(error_msg)
            with open(debug_file, 'a', encoding='utf-8') as f:
                f.write("=== Error ===\n")
                f.write(error_msg)
                f.write("\n\n")
            return None
    
    except Exception as e:
        error_msg = f"Error identifying segments: {str(e)}"
        logger.error(error_msg)
        with open(debug_file, 'a', encoding='utf-8') as f:
            f.write("=== Error ===\n")
            f.write(error_msg)
            f.write("\n\n")
        return None

def generate_titles(segments_data: Dict[str, Any]) -> Tuple[str, List[HighlightClip]]:
    """
    Use Claude 3.5 to generate engaging titles for the segments
    
    Args:
        segments_data: Dictionary containing segment information and theme
    
    Returns:
        Tuple of (main_title, List[HighlightClip])
    """
    logger.info("Generating segment title using Claude")
    
    # Prepare prompt for Claude
    segments_text = "\n".join([
        f"{i+1}. {seg['start_time']:.2f}s - {seg['end_time']:.2f}s\n內容：{seg['content']}\n原因：{seg['reason']}"
        for i, seg in enumerate(segments_data['segments'])
    ])
    
    prompt = f"""你是影片標題專家，需要為一個視頻片段創作吸引人的標題。要直接、簡潔、有力，不要使用過多的修飾詞。要严格符合圣经教义.

視頻主題：{segments_data['theme']}

精彩片段內容：
{segments_text}

請為片段創作標題，格式如下：
```json
{{
  "segments": [
    {{
      "title": "片段標題（15字以內，繁體中文）",
      "score": 吸引力評分（0-100）
    }}
  ]
}}
```

要求：
1. 片段標題要吸引人，突出該片段最精彩的部分
2. 所有標題使用繁體中文
3. 標題要簡潔有力，容易理解"""

    try:
        # Get response from Claude
        response = anthropic_client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=1000,
            system="You are a creative title generator specializing in engaging video titles. Always respond in Traditional Chinese.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Extract JSON from the response
        response_text = response.content[0].text
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = response_text[json_start:json_end]
            data = json.loads(json_str)
            
            # Create HighlightClip objects
            highlights = []
            for seg_data, title_data in zip(segments_data['segments'], data['segments']):
                highlight = HighlightClip(
                    start=float(seg_data['start_time']),
                    end=float(seg_data['end_time']),
                    score=float(title_data['score']),
                    title=title_data['title']
                )
                highlights.append(highlight)
            
            # Return empty string for main_title since we're not generating it
            return "", highlights
        else:
            logger.error("Failed to extract JSON from Claude response")
            return "", []
    
    except Exception as e:
        logger.error(f"Error generating titles: {e}")
        return "", []

def create_layout_preview(
    input_path: str,
    timestamp: float,
    main_title: str,
    output_preview_path: str,
    logo_path: Optional[str] = None,
    segments: Optional[List[TranscriptionSegment]] = None,  # Add segments parameter
    highlight: Optional[HighlightClip] = None  # Add highlight parameter
) -> bool:
    """
    Create a preview image of the video layout
    
    Args:
        input_path: Path to input video
        timestamp: Timestamp to use for preview frame
        main_title: Main title of the video
        output_preview_path: Path to save the preview image
        logo_path: Path to logo image file (optional)
        segments: List of transcription segments (optional)
        highlight: HighlightClip object (optional)
    
    Returns:
        bool: True if user confirms the layout, False otherwise
    """
    logger.info(f"Creating layout preview using frame at {timestamp:.2f}s")
    
    # Initialize font manager
    font_manager = FontManager()
    font_path = font_manager.get_font_path()
    logger.info(f"Using font: {font_path}")
    
    # Load video and extract frame
    video = VideoFileClip(input_path)
    frame = video.get_frame(timestamp)
    video.close()
    
    # Debug original video dimensions
    logger.info(f"Original video dimensions: {frame.shape[1]}x{frame.shape[0]} (width x height)")
    
    # Create PIL Image for preview
    frame_image = Image.fromarray(frame)
    
    # Calculate dimensions for 9:16 format
    output_width = 1080
    output_height = 1920
    logger.info(f"Output dimensions (9:16): {output_width}x{output_height} (width x height)")
    
    # Create black background
    preview = Image.new('RGB', (output_width, output_height), 'black')
    
    # Resize frame while maintaining aspect ratio
    frame_width = output_width
    frame_height = int(frame_image.height * (output_width / frame_image.width))
    frame_resized = frame_image.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
    logger.info(f"Resized video frame dimensions: {frame_width}x{frame_height} (width x height)")
    
    # Calculate position to center the frame
    frame_y = (output_height - frame_height) // 2
    logger.info(f"Video frame Y position: {frame_y} (distance from top)")
    preview.paste(frame_resized, (0, frame_y))
    
    # Create draw object
    draw = ImageDraw.Draw(preview)
    
    # Load fonts with proper Chinese support
    title_font = ImageFont.truetype(font_path, 80)
    church_font = ImageFont.truetype(font_path, 50)
    service_font = ImageFont.truetype(font_path, 40)
    logger.info("Using Noto Sans TC font with sizes: title=80px, church=50px, service=40px")
    
    # Define text areas with increased space for logo
    title_margin = 40
    church_name_height = 130  # Increased from 100 to 200 for larger logo
    main_title_height = 200  # Increased from 150 to 200 for longer titles
    service_info_height = 80
    
    # Debug text area dimensions
    logger.info("Text area dimensions:")
    logger.info(f"- Title margin: {title_margin}px")
    logger.info(f"- Church name height: {church_name_height}px")
    logger.info(f"- Main title height: {main_title_height}px")
    logger.info(f"- Service info height: {service_info_height}px")
    
    # Calculate and debug text positions
    church_name_y = title_margin
    main_title_y = title_margin + church_name_height + main_title_height//2
    service_info_y = title_margin + church_name_height + main_title_height + service_info_height//2
    
    logger.info("Text Y positions from top:")
    logger.info(f"- Church name area: {church_name_y}px")
    logger.info(f"- Main title: {main_title_y}px")
    logger.info(f"- Service info: {service_info_y}px")
    
    # Add logo for church name with increased size
    if logo_path and os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path)
            # Calculate logo size to fit church name height while maintaining aspect ratio
            logo_height = church_name_height
            logo_width = int(logo.width * (logo_height / logo.height))
            
            # Allow logo to be wider, up to 90% of output width
            max_logo_width = int(output_width * 0.9)  # Increased from fixed margin to 90% of width
            if logo_width > max_logo_width:
                logo_width = max_logo_width
                logo_height = int(logo.height * (logo_width / logo.width))
            
            logo_resized = logo.resize((logo_width, logo_height), Image.Resampling.LANCZOS)
            logger.info(f"Adding logo with dimensions: {logo_width}x{logo_height}")
            
            # Calculate position to center the logo
            logo_x = (output_width - logo_width) // 2
            logger.info(f"Logo position: x={logo_x}, y={church_name_y}")
            
            # Paste logo with alpha channel support
            preview.paste(logo_resized, (logo_x, church_name_y), logo_resized if logo.mode == 'RGBA' else None)
        except Exception as e:
            logger.warning(f"Failed to add logo as church name: {e}")
            # Fallback to text if logo fails
            church_name = "開路者教會 WAYMAKER CHURCH"
            logger.info(f"Falling back to text: '{church_name}' at y={church_name_y + church_name_height//2}")
            draw.text(
                (output_width//2, church_name_y + church_name_height//2),
                church_name,
                font=church_font,
                fill='white',
                anchor='mm'
            )
    else:
        # Fallback to text if no logo provided
        church_name = "開路者教會 WAYMAKER CHURCH"
        logger.info(f"No logo provided, using text: '{church_name}' at y={church_name_y + church_name_height//2}")
        draw.text(
            (output_width//2, church_name_y + church_name_height//2),
            church_name,
            font=church_font,
            fill='white',
            anchor='mm'
        )
    
    # Draw main title with multi-line support
    logger.info(f"Drawing main title: '{main_title}' at y={main_title_y}")
    
    # Calculate text size and wrap if needed
    title_font_size = 80
    title_font = ImageFont.truetype(font_path, title_font_size)
    
    # Check if title fits within width
    title_width = title_font.getbbox(main_title)[2] - title_font.getbbox(main_title)[0]
    available_width = output_width - 2 * title_margin
    
    if title_width > available_width:
        # Try to find a natural break point
        break_points = []
        for i, char in enumerate(main_title):
            if char in '。，！？,.!? ':
                break_points.append(i)
        
        if break_points and len(main_title) > 10:
            # Find the best break point near the middle
            middle_idx = len(main_title) // 2
            best_break = min(break_points, key=lambda x: abs(x - middle_idx))
            
            # Split the title into two lines
            line1 = main_title[:best_break+1]
            line2 = main_title[best_break+1:]
            
            # Draw each line
            line_spacing = 10
            draw.text(
                (output_width//2, main_title_y - title_font_size//2 - line_spacing//2),
                line1,
                font=title_font,
                fill='#00FF00',
                anchor='mm'
            )
            draw.text(
                (output_width//2, main_title_y + title_font_size//2 + line_spacing//2),
                line2,
                font=title_font,
                fill='#00FF00',
                anchor='mm'
            )
        else:
            # If no good break point, reduce font size
            while title_width > available_width and title_font_size > 40:
                title_font_size -= 5
                title_font = ImageFont.truetype(font_path, title_font_size)
                title_width = title_font.getbbox(main_title)[2] - title_font.getbbox(main_title)[0]
            
            draw.text(
                (output_width//2, main_title_y),
                main_title,
                font=title_font,
                fill='#00FF00',
                anchor='mm'
            )
    else:
        # Title fits, draw normally
        draw.text(
            (output_width//2, main_title_y),
            main_title,
            font=title_font,
            fill='#00FF00',
            anchor='mm'
        )
    
    # Draw service info
    service_info = "主日崇拜: 每週日下午2点"
    logger.info(f"Drawing service info: '{service_info}' at y={service_info_y}")
    draw.text(
        (output_width//2, service_info_y),
        service_info,
        font=service_font,
        fill='white',
        anchor='mm'
    )
    
    # Add first subtitle if segments are provided
    if segments and highlight:
        # Find the first segment that overlaps with the highlight
        first_subtitle = None
        for seg in segments:
            if seg.start <= highlight.end and seg.end >= highlight.start:
                first_subtitle = seg
                break
        
        if first_subtitle:
            # Split text if longer than 15 characters
            subtitle_text = first_subtitle.text
            if len(subtitle_text) > 15:
                # Try to find a natural break point
                break_points = [i for i, char in enumerate(subtitle_text[:15]) if char in '。，！？,.!? ']
                if break_points:
                    subtitle_text = subtitle_text[:max(break_points) + 1]
                else:
                    subtitle_text = subtitle_text[:15]
            
            # Calculate subtitle position below video
            subtitle_height = 120  # Match the height used in create_highlight_video
            subtitle_fontsize = 65  # Match the fontsize used in create_highlight_video
            subtitle_margin = 40
            
            # Position subtitle below video frame
            subtitle_y = frame_y + frame_height + subtitle_margin
            
            # Create subtitle font
            subtitle_font = ImageFont.truetype(font_path, subtitle_fontsize)
            
            logger.info(f"Adding preview subtitle: '{subtitle_text}' at y={subtitle_y}")
            draw.text(
                (output_width//2, subtitle_y + subtitle_height//2),
                subtitle_text,
                font=subtitle_font,
                fill='white',
                anchor='mm'
            )
    
    # Save preview
    preview.save(output_preview_path, 'JPEG', quality=95)
    logger.info(f"Saved layout preview to {output_preview_path}")
    
    # Show preview and get confirmation
    print("\nLayout Preview:")
    print("-" * 50)
    print("Layout dimensions:")
    print(f"- Output size: {output_width}x{output_height}")
    print(f"- Video frame: {frame_width}x{frame_height} at y={frame_y}")
    print(f"- Text margins: {title_margin}px")
    print("\nText positions (y-coordinate from top):")
    print(f"- Church name area: {church_name_y}px")
    print(f"- Main title: {main_title_y}px")
    print(f"- Service info: {service_info_y}px")
    print(f"\nPreview image saved to: {output_preview_path}")
    print("\nPlease check the preview image and confirm if you want to proceed with the video creation.")
    
    while True:
        choice = input("\nDo you want to:\n1. Continue with this layout\n2. Exit\nEnter your choice (1/2): ").strip()
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.")
    
    return choice == '1'

def create_highlight_video(
    input_path: str, 
    highlight: HighlightClip, 
    segments: Optional[List[TranscriptionSegment]],  # Make segments optional
    output_path: str,
    main_title: str,  # This will be ignored as we'll use highlight.title
    logo_path: Optional[str] = None,
    add_subtitles: bool = False,
    vertical_position_ratio: float = 0.67  # Add new parameter with default 2/3
):
    """
    Create highlight video in 9:16 format with titles and subtitles, styled like a church service video
    First creates a preview for confirmation, then proceeds with video creation if approved.
    Uses the segment title as the main title text.
    The main video section will be in 1:1 ratio.
    
    Args:
        input_path: Path to input video
        highlight: HighlightClip object with timing information
        segments: List of transcription segments (optional, None for full video mode)
        output_path: Path to output video
        main_title: Ignored as we use highlight.title instead
        logo_path: Path to logo image file (optional)
        add_subtitles: Whether to add subtitles to the video (default: False)
        vertical_position_ratio: Ratio for vertical positioning (default: 0.67 for 2/3 from top)
    """
    # Create zoom preview first
    zoom_preview_path = str(Path(output_path).with_suffix('.zoom_preview.jpg'))
    preview_timestamp = highlight.start + (highlight.end - highlight.start) / 2  # Use middle frame for preview
    
    # Show zoom preview and get confirmation
    if not create_zoom_preview(
        input_path=input_path,
        timestamp=preview_timestamp,
        zoom_factor=2.0,  # Fixed zoom factor
        vertical_position_ratio=vertical_position_ratio,
        output_preview_path=zoom_preview_path
    ):
        logger.info("Video creation cancelled by user after zoom preview")
        return
    
    # Create layout preview
    layout_preview_path = str(Path(output_path).with_suffix('.layout_preview.jpg'))
    if not create_layout_preview(
        input_path=input_path,
        timestamp=preview_timestamp,
        main_title=highlight.title,
        output_preview_path=layout_preview_path,
        logo_path=logo_path,
        segments=segments,  # This can be None now
        highlight=highlight
    ):
        logger.info("Video creation cancelled by user after layout preview")
        return

    logger.info(f"Creating highlight video from {highlight.start:.2f}s to {highlight.end:.2f}s")
    
    # Initialize font manager
    font_manager = FontManager()
    font_path = font_manager.get_font_path()
    
    # Load video clip and apply zoom and tracking
    video = VideoFileClip(input_path).subclip(highlight.start, highlight.end)
    logger.info(f"Original video dimensions: {video.size[0]}x{video.size[1]} (width x height)")
    
    # Apply zoom and tracking with vertical position ratio
    logger.info(f"Applying 200% zoom and horizontal person tracking with vertical position ratio {vertical_position_ratio}...")
    processed_clip = process_video_with_zoom_and_tracking(video, zoom_factor=2.0, vertical_position_ratio=vertical_position_ratio)
    
    # Calculate dimensions for 9:16 portrait format
    output_width = 1080
    output_height = 1920
    logger.info(f"Output dimensions (9:16): {output_width}x{output_height} (width x height)")
    
    # Calculate dimensions for 1:1 video section
    video_section_size = output_width  # Square video section
    logger.info(f"Video section dimensions (1:1): {video_section_size}x{video_section_size} (width x height)")
    
    # Resize video while maintaining aspect ratio to fit the square
    orig_height, orig_width = processed_clip.size[1], processed_clip.size[0]
    if orig_width > orig_height:
        # For landscape video, fit to height
        scale_factor = video_section_size / orig_height
        scaled_width = int(orig_width * scale_factor)
        scaled_height = video_section_size
        # Crop excess width
        x_offset = (scaled_width - video_section_size) // 2
        processed_clip = processed_clip.resize(width=scaled_width, height=scaled_height)
        processed_clip = processed_clip.crop(x1=x_offset, y1=0, x2=x_offset + video_section_size, y2=video_section_size)
    else:
        # For portrait video, fit to width
        scale_factor = video_section_size / orig_width
        scaled_width = video_section_size
        scaled_height = int(orig_height * scale_factor)
        # Crop excess height
        y_offset = (scaled_height - video_section_size) // 2
        processed_clip = processed_clip.resize(width=scaled_width, height=scaled_height)
        processed_clip = processed_clip.crop(x1=0, y1=y_offset, x2=video_section_size, y2=y_offset + video_section_size)
    
    logger.info(f"Final video dimensions: {video_section_size}x{video_section_size} (width x height)")
    
    # Create a black background for the 9:16 format
    black_bg = ColorClip(size=(output_width, output_height), color=(0, 0, 0))
    black_bg = black_bg.set_duration(video.duration)
    
    # Position the video in the center vertically with more space for the square
    # Calculate positions to center the 1:1 video with proper margins
    title_margin = 40
    church_name_height = 200  # Increased from 100 to 200 for larger logo
    main_title_height = 150
    service_info_height = 80
    
    # Calculate total height of text elements
    total_text_height = (title_margin + church_name_height + main_title_height + 
                        service_info_height + title_margin)
    
    # Position video after the top text elements with some margin
    video_y = total_text_height + title_margin
    video_pos = ('center', video_y)
    logger.info(f"Video position: center, y={video_y} (distance from top)")
    
    # Initialize clips list
    clips_to_combine = [black_bg, processed_clip.set_position(video_pos)]
    
    # Calculate text positions relative to the new video position
    church_name_y = title_margin
    main_title_y = title_margin + church_name_height + main_title_height//2
    service_info_y = title_margin + church_name_height + main_title_height + service_info_height//2
    
    # Add logo or church name with increased size
    if logo_path and os.path.exists(logo_path):
        try:
            # Create logo clip
            logo_clip = ImageClip(logo_path)
            
            # Calculate logo size with increased height
            logo_height = church_name_height
            logo_width = int(logo_clip.size[0] * (logo_height / logo_clip.size[1]))
            
            # Allow logo to be wider, up to 90% of output width
            max_logo_width = int(output_width * 0.9)  # Increased from fixed margin to 90% of width
            if logo_width > max_logo_width:
                logo_width = max_logo_width
                logo_height = int(logo_clip.size[1] * (logo_width / logo_clip.size[0]))
            
            # Resize and position logo
            logo_clip = logo_clip.resize(newsize=(logo_width, logo_height))
            logo_x = (output_width - logo_width) // 2
            logo_clip = logo_clip.set_position((logo_x, church_name_y))
            logo_clip = logo_clip.set_duration(video.duration)
            
            clips_to_combine.append(logo_clip)
            logger.info(f"Added logo clip: {logo_width}x{logo_height} at position ({logo_x}, {church_name_y})")
        except Exception as e:
            logger.warning(f"Failed to add logo: {e}")
            # Fallback to text
            church_name_clip = TextClip(
                "開路者教會 WAYMAKER CHURCH",
                fontsize=50,
                color='white',
                font=font_path,
                size=(output_width - 2*title_margin, church_name_height)
            ).set_position(('center', church_name_y))
            church_name_clip = church_name_clip.set_duration(video.duration)
            clips_to_combine.append(church_name_clip)
            logger.info("Added church name text clip as fallback")
    else:
        # Use text for church name
        church_name_clip = TextClip(
            "開路者教會 WAYMAKER CHURCH",
            fontsize=50,
            color='white',
            font=font_path,
            size=(output_width - 2*title_margin, church_name_height)
        ).set_position(('center', church_name_y))
        church_name_clip = church_name_clip.set_duration(video.duration)
        clips_to_combine.append(church_name_clip)
        logger.info("Added church name text clip")
    
    # Add main title with multi-line support
    title_text = highlight.title
    title_fontsize = 80
    
    # Check if title needs to be split into multiple lines
    title_width_estimate = len(title_text) * title_fontsize * 0.6  # Rough estimate
    available_width = output_width - 2 * title_margin
    
    if title_width_estimate > available_width:
        # Try to find a natural break point
        break_points = []
        for i, char in enumerate(title_text):
            if char in '。，！？,.!? ':
                break_points.append(i)
        
        if break_points and len(title_text) > 10:
            # Find the best break point near the middle
            middle_idx = len(title_text) // 2
            best_break = min(break_points, key=lambda x: abs(x - middle_idx))
            
            # Split the title into two lines
            line1 = title_text[:best_break+1]
            line2 = title_text[best_break+1:]
            
            # Create two separate text clips
            line1_clip = TextClip(
                line1,
                fontsize=title_fontsize,
                color='#00FF00',
                font=font_path,
                size=(output_width - 2*title_margin, main_title_height//2)
            ).set_position(('center', main_title_y - title_fontsize//2 - 5))
            
            line2_clip = TextClip(
                line2,
                fontsize=title_fontsize,
                color='#00FF00',
                font=font_path,
                size=(output_width - 2*title_margin, main_title_height//2)
            ).set_position(('center', main_title_y + title_fontsize//2 + 5))
            
            line1_clip = line1_clip.set_duration(video.duration)
            line2_clip = line2_clip.set_duration(video.duration)
            
            clips_to_combine.append(line1_clip)
            clips_to_combine.append(line2_clip)
            
            logger.info(f"Added multi-line title clips: '{line1}' and '{line2}'")
        else:
            # If no good break point, reduce font size
            while title_width_estimate > available_width and title_fontsize > 40:
                title_fontsize -= 5
                title_width_estimate = len(title_text) * title_fontsize * 0.6
            
            main_title_clip = TextClip(
                title_text,
                fontsize=title_fontsize,
                color='#00FF00',
                font=font_path,
                size=(output_width - 2*title_margin, main_title_height)
            ).set_position(('center', main_title_y))
            
            main_title_clip = main_title_clip.set_duration(video.duration)
            clips_to_combine.append(main_title_clip)
            logger.info(f"Added main title clip with reduced font size ({title_fontsize}px): '{title_text}'")
    else:
        # Title fits on one line
        main_title_clip = TextClip(
            title_text,
            fontsize=title_fontsize,
            color='#00FF00',
            font=font_path,
            size=(output_width - 2*title_margin, main_title_height)
        ).set_position(('center', main_title_y))
        
        main_title_clip = main_title_clip.set_duration(video.duration)
        clips_to_combine.append(main_title_clip)
        logger.info(f"Added main title clip: '{title_text}'")
    
    # Add service info
    service_info_clip = TextClip(
        "主日崇拜: 每週日下午2点",
        fontsize=40,
        color='white',
        font=font_path,
        size=(output_width - 2*title_margin, service_info_height)
    ).set_position(('center', service_info_y))
    service_info_clip = service_info_clip.set_duration(video.duration)
    clips_to_combine.append(service_info_clip)
    logger.info("Added service info clip")
    
    # Add subtitles only if requested and segments are available
    if add_subtitles and segments:
        # Filter segments that overlap with the highlight clip
        highlight_segments = [
            seg for seg in segments 
            if (seg.start <= highlight.end and seg.end >= highlight.start)
        ]
        logger.info(f"Found {len(highlight_segments)} subtitle segments in the selected time range")
        
        # Create a new list for processed segments
        processed_segments = []
        
        # Process word-level data from each segment
        for seg in highlight_segments:
            # Get words that fall within the highlight clip time range
            words = [
                word for word in seg.words
                if (word['start'] <= highlight.end and word['end'] >= highlight.start)
            ]
            
            if not words:
                continue
            
            # Group words into phrases (max 10 characters)
            current_phrase = []
            current_start = None
            current_text = ""
            
            for word in words:
                # Adjust timing relative to highlight start
                word_start = max(0, word['start'] - highlight.start)
                word_end = min(video.duration, word['end'] - highlight.start)
                
                # Initialize start time for new phrase
                if not current_start:
                    current_start = word_start
                
                # Add word to current phrase
                current_phrase.append(word)
                current_text += word['text']
                
                # Check if we should create a new phrase
                should_break = False
                
                # Break on punctuation, but only if we have enough content
                if any(p in word['text'] for p in '。，！？,.!?'):
                    # Only break if current phrase is long enough and not in the middle of a semantic unit
                    if len(current_text) >= 6:  # Lowered from 8
                        words_list = list(jieba.cut(current_text))
                        # Don't break if it would create a short next phrase
                        remaining_words = words[words.index(word)+1:]
                        if not remaining_words or len(''.join(w['text'] for w in remaining_words)) >= 6:  # Lowered from 8
                            should_break = True
                
                # Break on length
                elif len(current_text) >= 10:  # Lowered from 15
                    # Use jieba to find a natural break point
                    words_list = list(jieba.cut(current_text))
                    if len(words_list) > 1:
                        # Try to find the best break point near 10 characters
                        best_break_point = None
                        best_break_score = float('inf')
                        cumulative_length = 0
                        
                        for i, w in enumerate(words_list):
                            cumulative_length += len(w)
                            # Score based on how close to 10 characters and whether it's a natural break
                            score = abs(10 - cumulative_length)
                            # Prefer breaks at natural boundaries
                            if w in ['的', '了', '和', '與', '但是', '所以', '因為', '如果', '就是']:
                                score -= 2
                            
                            if cumulative_length >= 6 and cumulative_length <= 12:  # Allow 6-12 char range
                                if score < best_break_score:
                                    best_break_score = score
                                    best_break_point = cumulative_length
                        
                        if best_break_point:
                            should_break = True
                
                # Break on long pause (> 1.0 seconds), but only if we have enough content
                elif len(current_phrase) > 1:
                    pause_duration = word_start - (current_phrase[-2]['end'] - highlight.start)
                    if pause_duration > 1.0 and len(current_text) >= 6:  # Lowered from 8
                        # Check if breaking here would create a short next phrase
                        remaining_words = words[words.index(word)+1:]
                        if not remaining_words or len(''.join(w['text'] for w in remaining_words)) >= 6:  # Lowered from 8
                            should_break = True
                
                # Check for natural phrase breaks using jieba
                elif len(current_text) > 5:  # Only check if we have enough context
                    words_list = list(jieba.cut(current_text))
                    # Common Chinese particles and conjunctions that often mark phrase boundaries
                    phrase_boundaries = ['的', '了', '和', '與', '但是', '所以', '因為', '如果', '就是']  # Reduced list
                    
                    # Only break on boundaries if we have enough content
                    if 6 <= len(current_text) <= 12:  # Changed from >= 8 to a range
                        # Check if the last word is a natural boundary
                        if words_list[-1] in phrase_boundaries:
                            should_break = True
                        # Also break if we detect a complete semantic unit
                        elif len(words_list) >= 2 and any(w in phrase_boundaries for w in words_list[-2:]):
                            should_break = True

                if (should_break or word is words[-1]) and current_text:
                    # Convert text to traditional Chinese
                    phrase_text = zhconv.convert(current_text.strip(), 'zh-hant')
                    
                    # If this is the last word and the phrase is too short, try to combine with previous
                    if word is words[-1] and len(phrase_text) < 6 and processed_segments:  # Lowered from 5
                        prev_seg = processed_segments[-1]
                        combined_text = prev_seg.text + phrase_text
                        
                        # Only combine if the result isn't too long and the time gap isn't too large
                        if len(combined_text) <= 12:  # Lowered from 15
                            time_gap = current_start - prev_seg.end
                            if time_gap < 1.0:  # Less than 1 second gap
                                # Update previous segment
                                processed_segments[-1] = TranscriptionSegment(
                                    text=combined_text,
                                    start=prev_seg.start,
                                    end=word_end,
                                    words=prev_seg.words + current_phrase
                                )
                                # Reset without creating new segment
                                current_phrase = []
                                current_start = None
                                current_text = ""
                                continue
                    
                    # Don't create segments that are too short unless they're the only content
                    if len(phrase_text) >= 6 or word is words[-1]:  # Lowered from 5
                        new_seg = TranscriptionSegment(
                            text=phrase_text,
                            start=current_start,
                            end=word_end,
                            words=current_phrase.copy()
                        )
                        processed_segments.append(new_seg)
                    
                    # Reset for next phrase
                    current_phrase = []
                    current_start = None
                    current_text = ""
        
        # Sort segments by start time
        processed_segments.sort(key=lambda x: x.start)
        
        # Post-process segments to combine short ones
        i = 0
        while i < len(processed_segments) - 1:
            current_seg = processed_segments[i]
            next_seg = processed_segments[i + 1]
            
            # Check if either segment is too short or if combining would make a better length
            should_combine = False
            
            # If current segment is too short
            if len(current_seg.text) < 6:  # Lowered from 8
                should_combine = True
            # If next segment is too short
            elif len(next_seg.text) < 6:  # Lowered from 8
                should_combine = True
            # If both segments together would make a better phrase length (closer to 10)
            elif len(current_seg.text) + len(next_seg.text) <= 12:  # Lowered from 15
                time_gap = next_seg.start - current_seg.end
                if time_gap < 1.5:  # Allow slightly larger gap for better phrases
                    should_combine = True
            
            if should_combine:
                # Combine the segments
                combined_text = current_seg.text + next_seg.text
                if len(combined_text) <= 14:  # Lowered from 20
                    processed_segments[i] = TranscriptionSegment(
                        text=combined_text,
                        start=current_seg.start,
                        end=next_seg.end,
                        words=current_seg.words + next_seg.words
                    )
                    # Remove the next segment
                    processed_segments.pop(i + 1)
                    # Don't increment i, so we can check if the combined segment should be combined with the next
                    continue
            
            i += 1
        
        logger.info(f"After combining short segments: {len(processed_segments)} subtitle phrases")
        for i, seg in enumerate(processed_segments):
            logger.info(f"Segment {i+1}: {seg.start:.2f}s - {seg.end:.2f}s: {seg.text}")
        
        # Calculate subtitle position right below the video
        subtitle_height = 120  # Increased height for larger text
        subtitle_fontsize = 65  # Increased font size
        subtitle_margin = 40
        
        # Position subtitle right below the video
        video_bottom = video_y + scaled_height
        subtitle_y = video_bottom + subtitle_margin
        
        logger.info(f"Positioning subtitles at y={subtitle_y} (below video)")
        logger.info(f"Using larger font size: {subtitle_fontsize}px")
        
        # Create subtitle clips for all processed segments
        for i, seg in enumerate(processed_segments):
            logger.info(f"Creating subtitle {i+1}/{len(processed_segments)}: {seg.start:.2f}s - {seg.end:.2f}s")
            logger.info(f"Text: {seg.text}")
            
            # Create subtitle with animation
            subtitle = TextClip(
                seg.text,
                fontsize=subtitle_fontsize,
                color='white',
                font=font_path,
                method='caption',
                align='center',
                size=(output_width - 2*subtitle_margin, subtitle_height)
            )
            
            # Add fade in/out animation
            fade_duration = min(0.3, (seg.end - seg.start) / 4)
            subtitle = subtitle.set_position(('center', subtitle_y))
            subtitle = subtitle.set_start(seg.start).set_end(seg.end)
            subtitle = subtitle.crossfadein(fade_duration).crossfadeout(fade_duration)
            
            clips_to_combine.append(subtitle)
    else:
        if add_subtitles and not segments:
            logger.info("Skipping subtitles as no segments are available (full video mode)")
        else:
            logger.info("Skipping subtitles as they were not requested")
    
    # Combine all clips
    logger.info(f"Combining {len(clips_to_combine)} clips")
    final_clip = CompositeVideoClip(clips_to_combine, size=(output_width, output_height))
    final_clip = final_clip.set_duration(video.duration)
    
    # Write output file
    logger.info(f"Writing output video to {output_path}")
    logger.info(f"Video duration: {video.duration:.2f}s")
    logger.info(f"Output dimensions: {output_width}x{output_height}")
    logger.info(f"FPS: {video.fps}")
    
    final_clip.write_videofile(
        output_path,
        fps=video.fps,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile="temp-audio.m4a",
        remove_temp=True
    )
    
    logger.info("Highlight video creation complete")

def save_state(video_path: str, state_name: str, data: Any):
    """Save state data to a JSON file"""
    video_path_obj = Path(video_path)
    state_file = video_path_obj.with_name(f"{video_path_obj.stem}_{state_name}.state.json")
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {state_name} state to {state_file}")

def load_state(video_path: str, state_name: str) -> Optional[Any]:
    """Load state data from a JSON file if it exists"""
    video_path_obj = Path(video_path)
    state_file = video_path_obj.with_name(f"{video_path_obj.stem}_{state_name}.state.json")
    if state_file.exists():
        try:
            with open(state_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {state_name} state from {state_file}")
            return data
        except Exception as e:
            logger.warning(f"Failed to load {state_name} state: {e}")
    return None

def process_video_with_zoom_and_tracking(clip: VideoFileClip, zoom_factor: float = 2.0, vertical_position_ratio: float = 0.5) -> VideoFileClip:
    """
    Process video with static zoom and horizontal person tracking.
    Centers the person in frame while respecting video boundaries.
    Uses responsive tracking without speed limits for immediate following.
    When no person is detected for several consecutive frames, removes zoom and returns the original frame.
    Provides smooth transitions between zoomed and unzoomed states.
    
    Args:
        clip: Input VideoFileClip
        zoom_factor: Static zoom factor (default: 2.0 for 200% zoom)
        vertical_position_ratio: Ratio for vertical positioning (default: 0.5 for center, 0.67 for 2/3 from top)
    
    Returns:
        Processed VideoFileClip with zoom and tracking
    """
    # Initialize MediaPipe Pose with global variable to prevent recreation
    global mp_pose_instance
    if 'mp_pose_instance' not in globals():
        mp_pose = mp.solutions.pose
        mp_pose_instance = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    # Store previous positions for smooth tracking
    prev_center_x = None
    prev_offset_x = None
    smoothing_factor = 0.3  # Increased for more immediate response
    
    # Track consecutive frames with no person detected
    no_person_counter = 0
    max_no_person_frames = 15  # Number of consecutive frames before removing zoom
    
    # Track current zoom level for smooth transitions
    current_zoom = zoom_factor
    zoom_transition_speed = 0.1  # Speed of zoom transition
    
    # Track if we were previously in unzoomed state
    was_unzoomed = False
    
    def get_person_center(landmarks, frame_width, frame_height):
        """Calculate the center point of the person using multiple body landmarks"""
        # Key points for center calculation
        key_points = [
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
            mp.solutions.pose.PoseLandmark.LEFT_HIP,
            mp.solutions.pose.PoseLandmark.RIGHT_HIP,
            mp.solutions.pose.PoseLandmark.NOSE
        ]
        
        points = []
        for point in key_points:
            landmark = landmarks.landmark[point]
            if landmark.visibility > 0.5:  # Only use visible landmarks
                points.append(landmark.x)
        
        if points:
            # Use average of visible points
            return sum(points) / len(points) * frame_width
        else:
            # Fallback to frame center if no points are visible
            return frame_width / 2

    def process_frame(frame):
        nonlocal prev_center_x, prev_offset_x, no_person_counter, current_zoom, was_unzoomed
        
        # Get frame dimensions
        frame_height, frame_width = frame.shape[:2]
        
        try:
            # Convert frame to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = mp_pose_instance.process(frame_rgb)
            
            # Check if person is detected
            person_detected = results and results.pose_landmarks and any(
                landmark.visibility > 0.5 for landmark in results.pose_landmarks.landmark
            )
            
            if person_detected:
                # Check if we're transitioning from unzoomed to zoomed
                if was_unzoomed:
                    logger.debug("Person detected again, transitioning back to zoomed state")
                    was_unzoomed = False
                
                # Reset no person counter when a person is detected
                no_person_counter = 0
                
                # Gradually transition to full zoom if not already there
                if current_zoom < zoom_factor:
                    current_zoom = min(zoom_factor, current_zoom + zoom_transition_speed)
                
                # Calculate zoomed dimensions
                zoomed_width = int(frame_width * current_zoom)
                zoomed_height = int(frame_height * current_zoom)
                
                # Create zoomed frame
                zoomed_frame = cv2.resize(frame, (zoomed_width, zoomed_height))
                
                # Maximum possible offset (half of the extra width from zooming)
                max_offset_x = (zoomed_width - frame_width) // 2
                
                # Calculate person center using multiple body points
                current_center_x = get_person_center(results.pose_landmarks, frame_width, frame_height)
                
                # Apply light smoothing to center position
                if prev_center_x is not None:
                    person_center_x = prev_center_x + smoothing_factor * (current_center_x - prev_center_x)
                else:
                    person_center_x = current_center_x
                
                # Update previous center
                prev_center_x = person_center_x
                
                # Calculate target offset to center the person
                # Positive offset moves frame left, negative moves right
                target_offset_x = (frame_width/2 - person_center_x) * 1.2  # 1.2 factor for more aggressive centering
                
                # Limit target offset to valid range
                target_offset_x = np.clip(target_offset_x, -max_offset_x, max_offset_x)
                
                # Apply smoothing to camera movement
                if prev_offset_x is not None:
                    offset_x = prev_offset_x + smoothing_factor * (target_offset_x - prev_offset_x)
                else:
                    offset_x = target_offset_x
                
                # Ensure offset stays within bounds
                offset_x = np.clip(offset_x, -max_offset_x, max_offset_x)
                
                # Update previous offset
                prev_offset_x = offset_x
                
                # Calculate crop coordinates
                # Vertical position based on ratio (static)
                extra_height = zoomed_height - frame_height
                start_y = int(extra_height * vertical_position_ratio)
                end_y = start_y + frame_height
                
                # Ensure vertical bounds
                if start_y < 0:
                    start_y = 0
                    end_y = frame_height
                elif end_y > zoomed_height:
                    end_y = zoomed_height
                    start_y = zoomed_height - frame_height
                
                # Horizontal crop based on offset
                start_x = int((zoomed_width - frame_width) // 2 - offset_x)
                end_x = start_x + frame_width
                
                # Final horizontal boundary check
                if start_x < 0:
                    start_x = 0
                    end_x = frame_width
                elif end_x > zoomed_width:
                    end_x = zoomed_width
                    start_x = zoomed_width - frame_width
                
                # Crop and return the frame
                return zoomed_frame[start_y:end_y, start_x:end_x]
            else:
                # Increment no person counter
                no_person_counter += 1
                
                if no_person_counter >= max_no_person_frames:
                    # If no person detected for several consecutive frames, gradually transition to original frame
                    if current_zoom > 1.0:
                        # Gradually reduce zoom
                        current_zoom = max(1.0, current_zoom - zoom_transition_speed)
                        
                        if current_zoom == 1.0:
                            # We've reached unzoomed state
                            if not was_unzoomed:
                                logger.debug(f"No person detected for {no_person_counter} frames, fully unzoomed")
                                was_unzoomed = True
                            return frame
                        
                        # Calculate zoomed dimensions for transition
                        zoomed_width = int(frame_width * current_zoom)
                        zoomed_height = int(frame_height * current_zoom)
                        
                        # Create zoomed frame
                        zoomed_frame = cv2.resize(frame, (zoomed_width, zoomed_height))
                        
                        # Center crop during transition
                        extra_width = zoomed_width - frame_width
                        extra_height = zoomed_height - frame_height
                        start_x = extra_width // 2
                        start_y = int(extra_height * vertical_position_ratio)
                        
                        # Ensure bounds
                        if start_y < 0:
                            start_y = 0
                        if start_y + frame_height > zoomed_height:
                            start_y = zoomed_height - frame_height
                        
                        return zoomed_frame[start_y:start_y + frame_height, start_x:start_x + frame_width]
                    else:
                        # Already at original zoom level
                        prev_center_x = None
                        prev_offset_x = None
                        was_unzoomed = True
                        return frame
                elif prev_offset_x is not None:
                    # For a few frames, gradually transition back to center
                    # Calculate zoomed dimensions
                    zoomed_width = int(frame_width * current_zoom)
                    zoomed_height = int(frame_height * current_zoom)
                    
                    # Create zoomed frame
                    zoomed_frame = cv2.resize(frame, (zoomed_width, zoomed_height))
                    
                    # Maximum possible offset
                    max_offset_x = (zoomed_width - frame_width) // 2
                    
                    # Quick return to center when no person detected
                    target_offset_x = 0
                    offset_x = prev_offset_x + smoothing_factor * 2 * (target_offset_x - prev_offset_x)  # Faster return
                    prev_offset_x = offset_x
                    
                    # Calculate crop coordinates
                    extra_height = zoomed_height - frame_height
                    start_y = int(extra_height * vertical_position_ratio)
                    end_y = start_y + frame_height
                    
                    # Ensure vertical bounds
                    if start_y < 0:
                        start_y = 0
                        end_y = frame_height
                    elif end_y > zoomed_height:
                        end_y = zoomed_height
                        start_y = zoomed_height - frame_height
                    
                    # Horizontal crop based on offset
                    start_x = int((zoomed_width - frame_width) // 2 - offset_x)
                    end_x = start_x + frame_width
                    
                    # Final horizontal boundary check
                    if start_x < 0:
                        start_x = 0
                        end_x = frame_width
                    elif end_x > zoomed_width:
                        end_x = zoomed_width
                        start_x = zoomed_width - frame_width
                    
                    return zoomed_frame[start_y:end_y, start_x:end_x]
                else:
                    # If no previous offset, return original frame
                    was_unzoomed = True
                    return frame
            
        except Exception as e:
            logger.warning(f"Frame processing error: {e}")
            # Return original frame as fallback in case of error
            return frame
    
    # Process the clip
    processed_clip = clip.fl_image(process_frame)
    return processed_clip

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
    processed_clip = process_video_with_zoom_and_tracking(
        frame_clip,
        zoom_factor=zoom_factor,
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

def create_full_video_segment(video_path: str) -> Dict[str, Any]:
    """
    Create a segment that covers the entire video.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        Dictionary containing segment information for the full video
    """
    # Get video duration
    video = VideoFileClip(video_path)
    duration = video.duration
    video.close()
    
    # Create a segment covering the entire video
    return {
        "segments": [
            {
                "start_time": 0.0,
                "end_time": duration,
                "content": "完整影片",  # "Full video" in Traditional Chinese
                "reason": "使用完整影片進行處理"  # "Processing the entire video" in Traditional Chinese
            }
        ],
        "theme": "完整影片處理"  # "Full video processing" in Traditional Chinese
    }

def create_text_transcript(video_path: str, segments: List[TranscriptionSegment]) -> str:
    """
    Create a plain text transcript file from segments.
    
    Args:
        video_path: Path to the original video file
        segments: List of transcription segments
    
    Returns:
        Path to the created transcript file
    """
    video_path_obj = Path(video_path)
    transcript_path = video_path_obj.with_suffix('.transcript.txt')
    
    # Sort segments by start time
    sorted_segments = sorted(segments, key=lambda x: x.start)
    
    # Write transcript to file
    with open(transcript_path, 'w', encoding='utf-8') as f:
        f.write(f"Transcript for: {video_path_obj.name}\n")
        f.write("=" * 50 + "\n\n")
        
        # Track current speaker for formatting
        current_speaker = None
        
        for i, seg in enumerate(sorted_segments, 1):
            # Format timestamp
            timestamp = f"[{int(seg.start//60):02d}:{int(seg.start%60):02d}.{int((seg.start%1)*10):01d} - {int(seg.end//60):02d}:{int(seg.end%60):02d}.{int((seg.end%1)*10):01d}]"
            
            # Handle speaker changes
            if seg.speaker != current_speaker:
                current_speaker = seg.speaker
                # Add a blank line before new speaker (except for first speaker)
                if i > 1:
                    f.write("\n")
                # Write speaker header
                if current_speaker:
                    f.write(f"\nSpeaker {current_speaker}:\n{'-' * 20}\n")
            
            # Write the segment with timestamp and text
            f.write(f"{timestamp} {seg.text}\n")
            
            # Add blank line every 5 segments within same speaker for readability
            if i % 5 == 0 and i < len(sorted_segments) and sorted_segments[i].speaker == current_speaker:
                f.write("\n")
    
    logger.info(f"Created plain text transcript at: {transcript_path}")
    return str(transcript_path)

def main():
    """Main function to process video and create highlight clips"""
    parser = argparse.ArgumentParser(description='Generate highlight clips from long videos')
    parser.add_argument('input_file', help='Path to the input video or audio file')
    parser.add_argument('--output', '-o', help='Path to the output video file')
    parser.add_argument('--max-duration', '-d', type=float, default=120.0, 
                      help='Target duration for the highlight clip in seconds (default: 120)')
    parser.add_argument('--logo', '-l', 
                      default=os.path.join(os.path.dirname(__file__), 'default_assets', 'Waymaker_white_logo_transparent_background.png'),
                      help='Path to logo image file (default: default_assets/Waymaker_white_logo_transparent_background.png)')
    parser.add_argument('--resume-from', '-r', choices=['transcribe', 'segments', 'titles'],
                      help='Resume from a specific stage using saved state')
    parser.add_argument('--add-subtitles', action='store_true',
                      help='Add subtitles to the video (default: False)')
    parser.add_argument('--vertical-position', '-v', type=float, default=0.67,
                      help='Vertical position ratio for the main subject (default: 0.67 for 2/3 from top)')
    parser.add_argument('--full-video', '-f', action='store_true',
                      help='Process the entire video without selecting segments (default: False)')
    parser.add_argument('--manual-title', '-m', help='Manually specify the video title')
    parser.add_argument('--language-code', type=str, default='zh',
                      help='Language code for transcription (default: zh for Chinese)')
    parser.add_argument('--transcribe-only', action='store_true',
                      help='Stop after transcription step and exit')
    
    args = parser.parse_args()
    
    # Check if input is audio or video
    input_path = Path(args.input_file)
    is_video = input_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    
    # If audio file and not transcribe-only, show error
    if not is_video and not args.transcribe_only:
        logger.error("Audio files are only supported with --transcribe-only flag")
        return
    
    # Default output path if not specified
    if not args.output and is_video:  # Only set output path for video files
        input_path = Path(args.input_file)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output = str(input_path.with_name(f"{input_path.stem}_highlight_{timestamp}{input_path.suffix}"))
        logger.info(f"Setting default output path to: {args.output}")
    
    # Ensure default logo directory exists
    default_logo_dir = os.path.join(os.path.dirname(__file__), 'default_assets')
    if not os.path.exists(default_logo_dir):
        os.makedirs(default_logo_dir)
        logger.info(f"Created default assets directory: {default_logo_dir}")
    
    # Check if logo file exists
    if not os.path.exists(args.logo):
        logger.warning(f"Logo file not found at {args.logo}")
        args.logo = None  # Fall back to text if logo file doesn't exist
    
    try:
        # Initialize state variables
        segments = None
        segments_data = None
        highlights = []
        highlight = None
        in_title_generation = False

        # Step 1: Load or generate transcription
        if args.resume_from in ['transcribe', 'segments', 'titles']:
            # Load transcription state
            transcript_data = load_state(args.input_file, 'transcript')
            if transcript_data:
                segments = [
                    TranscriptionSegment(
                        text=seg['text'],
                        start=seg['start'],
                        end=seg['end'],
                        words=seg.get('words', []),  # Include word-level timestamps
                        speaker=seg.get('speaker')  # Load speaker information
                    )
                    for seg in transcript_data
                ]
                logger.info(f"Loaded transcription with {len(segments)} segments")
                
                # If using transcribe_only mode, create text transcript file
                if args.transcribe_only:
                    transcript_txt_path = Path(args.input_file).with_suffix('.transcript.txt')
                    if not transcript_txt_path.exists():
                        logger.info("Creating plain text transcript file from loaded state...")
                        create_text_transcript(args.input_file, segments)
                    logger.info("Transcription data loaded. Exiting as --transcribe-only flag was set.")
                    return
            else:
                logger.error("No transcription state found to resume from")
                if args.full_video and is_video:
                    # For full video mode, we can continue without transcription if not resuming
                    logger.info("Continuing without transcription in full video mode")
                    segments = None
                else:
                    return
        else:
            logger.info("Generating transcription...")
            segments = transcribe_video(args.input_file, args.language_code)
            # Save transcription state
            save_state(args.input_file, 'transcript', [
                {
                    'text': seg.text,
                    'start': seg.start,
                    'end': seg.end,
                    'words': seg.words,
                    'speaker': seg.speaker
                }
                for seg in segments
            ])
            
            # If transcribe_only flag is set, exit after transcription
            if args.transcribe_only:
                # Create plain text transcript file if it doesn't exist
                transcript_txt_path = Path(args.input_file).with_suffix('.transcript.txt')
                if not transcript_txt_path.exists():
                    logger.info("Creating plain text transcript file...")
                    create_text_transcript(args.input_file, segments)
                
                logger.info("Transcription completed. Exiting as --transcribe-only flag was set.")
                return

        # Skip video processing for audio files
        if not is_video:
            return

        # If manual title is provided, create highlight object directly
        if args.manual_title:
            if args.full_video:
                # Create a segment for the full video
                segments_data = create_full_video_segment(args.input_file)
                # Create highlight object with manual title
                highlight = HighlightClip(
                    start=float(segments_data['segments'][0]['start_time']),
                    end=float(segments_data['segments'][0]['end_time']),
                    score=100.0,  # Give perfect score for manual titles
                    title=args.manual_title
                )
                highlights = [highlight]
                in_title_generation = True
                logger.info(f"Using manual title: {args.manual_title}")
            else:
                logger.error("Manual title can only be used with --full-video option")
                return

        # Step 2: Load or identify segments (if not already done with manual title)
        if args.full_video and not highlight:
            # Create a segment for the full video
            segments_data = create_full_video_segment(args.input_file)
            logger.info("Using full video as a single segment")
            in_title_generation = True  # Skip segment selection
        elif not args.full_video and args.resume_from in ['segments', 'titles'] and not highlight:
            # Load segments state
            segments_data = load_state(args.input_file, 'segments')
            if not segments_data:
                logger.error("No segments state found to resume from")
                return
            logger.info("Loaded segments data from state")

        # Step 3: Load titles if resuming from titles (if not already done with manual title)
        if not highlight and args.resume_from == 'titles':
            # Load titles state
            titles_data = load_state(args.input_file, 'titles')
            if titles_data and titles_data.get('segments'):
                title_data = titles_data['segments'][0]  # We now only have one highlight
                # Create highlight object from the saved state
                seg_data = segments_data['segments'][0]
                highlight = HighlightClip(
                    start=float(seg_data['start_time']),
                    end=float(seg_data['end_time']),
                    score=float(title_data['score']),
                    title=title_data['title']
                )
                highlights = [highlight]
                in_title_generation = True
                logger.info("Loaded titles data from state")
            else:
                logger.error("No titles state found to resume from")
                return

        while True:
            # Main workflow loop
            if not in_title_generation:
                # Show segment details and ask for confirmation
                if not segments_data:
                    segments_data = identify_highlights(segments, args.max_duration)
                    if not segments_data:
                        logger.error("No segment identified")
                        return
                    # Save segments state
                    save_state(args.input_file, 'segments', segments_data)
                
                print("\nSelected Segment:")
                print("-" * 50)
                
                seg = segments_data['segments'][0]  # We now only have one segment
                start_time = float(seg['start_time'])
                end_time = float(seg['end_time'])
                duration = end_time - start_time
                
                print(f"Time Range: {start_time:.2f}s - {end_time:.2f}s (Duration: {duration:.2f}s)")
                print(f"Content Summary: {seg['content']}")
                print(f"Selection Reason: {seg['reason']}")
                
                # Show original transcripts for this time range
                if segments:
                    print("\nOriginal Transcripts:")
                    print("-" * 30)
                    for transcript_seg in segments:
                        if (transcript_seg.start <= end_time and transcript_seg.end >= start_time):
                            print(f"{transcript_seg.start:.2f}s - {transcript_seg.end:.2f}s: {transcript_seg.text}")
                
                print("\nVideo Theme:")
                print("-" * 50)
                print(segments_data['theme'])
                
                while True:
                    choice = input("\nDo you want to:\n1. Continue with this segment\n2. Generate new segment\n3. Exit\nEnter your choice (1/2/3): ").strip()
                    if choice in ['1', '2', '3']:
                        break
                    print("Invalid choice. Please enter 1, 2, or 3.")
                
                if choice == '1':
                    # Move to title generation
                    in_title_generation = True
                    # Generate initial titles
                    _, highlights = generate_titles(segments_data)
                    if highlights:
                        highlight = highlights[0]  # We now only have one highlight
                        # Save titles state
                        save_state(args.input_file, 'titles', {
                            'segments': [
                                {
                                    'title': highlight.title,
                                    'score': highlight.score
                                }
                            ]
                        })
                    continue
                elif choice == '2':
                    # Generate new segment
                    logger.info("\nGenerating new segment...")
                    segments_data = identify_highlights(segments, args.max_duration)
                    if segments_data:
                        # Save segments state
                        save_state(args.input_file, 'segments', segments_data)
                        # Reset title generation variables
                        highlights = []
                        highlight = None
                    continue
                else:  # choice == '3'
                    print("Exiting program.")
                    return
            
            else:  # in title generation step
                if not highlight:
                    # Generate titles if we don't have them
                    _, highlights = generate_titles(segments_data)
                    if highlights:
                        highlight = highlights[0]  # We now only have one highlight
                        # Save titles state
                        save_state(args.input_file, 'titles', {
                            'segments': [
                                {
                                    'title': highlight.title,
                                    'score': highlight.score
                                }
                            ]
                        })
                    else:
                        logger.error("Failed to generate titles")
                        return
                
                if not args.manual_title:
                    print("\n標題選項:")  # "Title Options" in Traditional Chinese
                    print("-" * 50)
                    print(f"生成的標題: {highlight.title}")  # "Generated title" in Traditional Chinese
                    print(f"參與度評分: {highlight.score}/100")  # "Engagement score" in Traditional Chinese
                    
                    while True:
                        title_choice = input("\n您想要:\n1. 使用這個標題\n2. 生成新標題\n3. 手動輸入標題\n4. 退出\n請輸入您的選擇 (1/2/3/4): ").strip()
                        if title_choice in ['1', '2', '3', '4']:
                            break
                        print("無效的選擇。請輸入 1、2、3 或 4。")
                    
                    if title_choice == '1':
                        pass  # Continue with current title
                    elif title_choice == '2':
                        # Regenerate titles using the same segments data
                        logger.info("重新生成標題...")  # "Regenerating titles..." in Traditional Chinese
                        _, highlights = generate_titles(segments_data)
                        if highlights:
                            highlight = highlights[0]
                            # Save titles state
                            save_state(args.input_file, 'titles', {
                                'segments': [
                                    {
                                        'title': highlight.title,
                                        'score': highlight.score
                                    }
                                ]
                            })
                        else:
                            logger.error("無法生成新標題")  # "Failed to generate new titles" in Traditional Chinese
                            return
                        continue
                    elif title_choice == '3':
                        # Manual title input
                        new_title = input("\n請輸入標題: ").strip()  # "Please enter title" in Traditional Chinese
                        if new_title:
                            highlight.title = new_title
                            highlight.score = 100.0  # Perfect score for manual titles
                            # Save titles state
                            save_state(args.input_file, 'titles', {
                                'segments': [
                                    {
                                        'title': highlight.title,
                                        'score': highlight.score
                                    }
                                ]
                            })
                        else:
                            print("標題不能為空，將使用生成的標題。")  # "Title cannot be empty..." in Traditional Chinese
                        continue
                    else:  # title_choice == '4'
                        print("退出程序。")  # "Exiting program" in Traditional Chinese
                        return
                
                # Create highlight video
                create_highlight_video(
                    args.input_file,
                    highlight,
                    segments,  # Always pass segments if available (for subtitles)
                    args.output,
                    highlight.title,  # Use highlight.title as the main title
                    args.logo,
                    args.add_subtitles,  # Pass the subtitle flag
                    args.vertical_position  # Pass the vertical position ratio
                )
                
                logger.info(f"成功創建精彩片段視頻: {args.output}")  # "Successfully created highlight video" in Traditional Chinese
                return
    
    except Exception as e:
        logger.error(f"處理視頻時出錯: {e}")  # "Error processing video" in Traditional Chinese
        raise

if __name__ == "__main__":
    main() 