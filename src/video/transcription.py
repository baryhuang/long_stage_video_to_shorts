#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video transcription module for handling audio transcription, subtitles, and text analysis.
"""

import os
import json
import logging
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple

import numpy as np
import torch
from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_silence

logger = logging.getLogger(__name__)

class TranscriptionSegment:
    """
    Represents a segment of transcribed text with timing information.
    """
    def __init__(self, 
                text: str, 
                start: float, 
                end: float, 
                confidence: float = 1.0,
                words: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize a transcription segment.
        
        Args:
            text: The transcribed text
            start: Start time in seconds
            end: End time in seconds
            confidence: Confidence score for this segment (0-1)
            words: Optional word-level timing information
        """
        self.text = text
        self.start = start
        self.end = end
        self.confidence = confidence
        self.words = words or []
    
    def duration(self) -> float:
        """
        Get the duration of this segment in seconds.
        
        Returns:
            Duration in seconds
        """
        return self.end - self.start
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the segment to a dictionary.
        
        Returns:
            Dictionary representation of the segment
        """
        return {
            'text': self.text,
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence,
            'words': self.words
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranscriptionSegment':
        """
        Create a segment from a dictionary.
        
        Args:
            data: Dictionary representation of the segment
            
        Returns:
            TranscriptionSegment object
        """
        return cls(
            text=data['text'],
            start=data['start'],
            end=data['end'],
            confidence=data.get('confidence', 1.0),
            words=data.get('words', [])
        )

class Transcription:
    """
    Represents a full transcription with multiple segments.
    """
    def __init__(self, 
                segments: Optional[List[TranscriptionSegment]] = None,
                metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize a transcription.
        
        Args:
            segments: List of transcription segments
            metadata: Optional metadata about the transcription
        """
        self.segments = segments or []
        self.metadata = metadata or {}
    
    def add_segment(self, segment: TranscriptionSegment) -> None:
        """
        Add a segment to the transcription.
        
        Args:
            segment: TranscriptionSegment to add
        """
        self.segments.append(segment)
    
    def get_segment_at_time(self, time: float) -> Optional[TranscriptionSegment]:
        """
        Get the segment at a specific time.
        
        Args:
            time: Time in seconds
            
        Returns:
            TranscriptionSegment if found, None otherwise
        """
        for segment in self.segments:
            if segment.start <= time < segment.end:
                return segment
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the transcription to a dictionary.
        
        Returns:
            Dictionary representation of the transcription
        """
        return {
            'segments': [segment.to_dict() for segment in self.segments],
            'metadata': self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert the transcription to a JSON string.
        
        Args:
            indent: Number of spaces for indentation in the JSON output
            
        Returns:
            JSON string representation of the transcription
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transcription':
        """
        Create a transcription from a dictionary.
        
        Args:
            data: Dictionary representation of the transcription
            
        Returns:
            Transcription object
        """
        segments = [TranscriptionSegment.from_dict(s) for s in data.get('segments', [])]
        return cls(segments=segments, metadata=data.get('metadata', {}))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Transcription':
        """
        Create a transcription from a JSON string.
        
        Args:
            json_str: JSON string representation of the transcription
            
        Returns:
            Transcription object
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """
        Save the transcription to a JSON file.
        
        Args:
            file_path: Path to save the JSON file
        """
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'Transcription':
        """
        Load a transcription from a JSON file.
        
        Args:
            file_path: Path to the JSON file
            
        Returns:
            Transcription object
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return cls.from_json(f.read())
    
    def get_text(self) -> str:
        """
        Get the full text of the transcription.
        
        Returns:
            Concatenated text of all segments
        """
        return ' '.join(segment.text for segment in self.segments)
    
    def merge_adjacent_segments(self, 
                              max_gap: float = 0.5, 
                              max_merged_duration: float = 10.0) -> 'Transcription':
        """
        Merge adjacent segments with small gaps between them.
        
        Args:
            max_gap: Maximum gap in seconds between segments to merge
            max_merged_duration: Maximum duration in seconds for a merged segment
            
        Returns:
            New Transcription object with merged segments
        """
        if not self.segments:
            return Transcription(metadata=self.metadata.copy())
        
        # Sort segments by start time
        sorted_segments = sorted(self.segments, key=lambda s: s.start)
        
        # Initialize with the first segment
        merged_segments = [TranscriptionSegment(
            text=sorted_segments[0].text,
            start=sorted_segments[0].start,
            end=sorted_segments[0].end,
            confidence=sorted_segments[0].confidence,
            words=sorted_segments[0].words.copy() if sorted_segments[0].words else None
        )]
        
        # Process the rest of the segments
        for segment in sorted_segments[1:]:
            prev = merged_segments[-1]
            
            # Check if this segment should be merged with the previous one
            if (segment.start - prev.end <= max_gap and 
                segment.end - prev.start <= max_merged_duration):
                
                # Merge the segments
                words = []
                if prev.words:
                    words.extend(prev.words)
                if segment.words:
                    words.extend(segment.words)
                
                merged_segments[-1] = TranscriptionSegment(
                    text=prev.text + ' ' + segment.text,
                    start=prev.start,
                    end=segment.end,
                    confidence=min(prev.confidence, segment.confidence),
                    words=words if words else None
                )
            else:
                # Add as a new segment
                merged_segments.append(TranscriptionSegment(
                    text=segment.text,
                    start=segment.start,
                    end=segment.end,
                    confidence=segment.confidence,
                    words=segment.words.copy() if segment.words else None
                ))
        
        return Transcription(segments=merged_segments, metadata=self.metadata.copy())
    
    def split_long_segments(self, 
                          max_duration: float = 10.0, 
                          split_on_silence: bool = True,
                          min_silence_duration: float = 0.5) -> 'Transcription':
        """
        Split long segments into smaller ones.
        
        Args:
            max_duration: Maximum duration in seconds for a segment
            split_on_silence: Whether to try to split on silence
            min_silence_duration: Minimum silence duration to split on
            
        Returns:
            New Transcription object with split segments
        """
        if not self.segments:
            return Transcription(metadata=self.metadata.copy())
        
        result_segments = []
        
        for segment in self.segments:
            # If the segment is short enough, keep it as is
            if segment.duration() <= max_duration:
                result_segments.append(TranscriptionSegment(
                    text=segment.text,
                    start=segment.start,
                    end=segment.end,
                    confidence=segment.confidence,
                    words=segment.words.copy() if segment.words else None
                ))
                continue
            
            # Split the segment
            if split_on_silence and segment.words:
                # Try to split on silences using word timing
                silence_points = []
                
                # Find silence points between words
                for i in range(1, len(segment.words)):
                    prev_word = segment.words[i-1]
                    curr_word = segment.words[i]
                    
                    gap = curr_word['start'] - prev_word['end']
                    
                    if gap >= min_silence_duration:
                        silence_points.append(curr_word['start'])
                
                # If no suitable silence points, split evenly
                if not silence_points:
                    num_parts = int(segment.duration() / max_duration) + 1
                    duration_per_part = segment.duration() / num_parts
                    
                    for i in range(num_parts):
                        part_start = segment.start + i * duration_per_part
                        part_end = min(segment.start + (i + 1) * duration_per_part, segment.end)
                        
                        # Extract words for this part
                        part_words = []
                        if segment.words:
                            part_words = [w for w in segment.words 
                                        if part_start <= w['start'] < part_end]
                        
                        # Extract text for this part
                        if part_words:
                            part_text = ' '.join(w['word'] for w in part_words)
                        else:
                            # If no words, approximate based on position
                            part_ratio = duration_per_part / segment.duration()
                            part_text = segment.text  # Simplified
                        
                        result_segments.append(TranscriptionSegment(
                            text=part_text,
                            start=part_start,
                            end=part_end,
                            confidence=segment.confidence,
                            words=part_words
                        ))
                else:
                    # Split at silence points
                    split_points = [segment.start] + silence_points + [segment.end]
                    
                    for i in range(len(split_points) - 1):
                        part_start = split_points[i]
                        part_end = split_points[i + 1]
                        
                        # Skip if resulting segment would be too long
                        if part_end - part_start > max_duration:
                            continue
                        
                        # Extract words for this part
                        part_words = []
                        if segment.words:
                            part_words = [w for w in segment.words 
                                        if part_start <= w['start'] < part_end]
                        
                        # Extract text for this part
                        if part_words:
                            part_text = ' '.join(w['word'] for w in part_words)
                        else:
                            # If no words, use original text (simplified)
                            part_text = segment.text
                        
                        result_segments.append(TranscriptionSegment(
                            text=part_text,
                            start=part_start,
                            end=part_end,
                            confidence=segment.confidence,
                            words=part_words
                        ))
            else:
                # Simple even split
                num_parts = int(segment.duration() / max_duration) + 1
                duration_per_part = segment.duration() / num_parts
                
                for i in range(num_parts):
                    part_start = segment.start + i * duration_per_part
                    part_end = min(segment.start + (i + 1) * duration_per_part, segment.end)
                    
                    # Extract words for this part
                    part_words = []
                    if segment.words:
                        part_words = [w for w in segment.words 
                                    if part_start <= w['start'] < part_end]
                    
                    # Extract text for this part
                    if part_words:
                        part_text = ' '.join(w['word'] for w in part_words)
                    else:
                        # If no words, approximate based on position
                        part_ratio = duration_per_part / segment.duration()
                        words = segment.text.split()
                        start_idx = int(i * part_ratio * len(words))
                        end_idx = int(min((i + 1) * part_ratio * len(words), len(words)))
                        part_text = ' '.join(words[start_idx:end_idx])
                    
                    result_segments.append(TranscriptionSegment(
                        text=part_text,
                        start=part_start,
                        end=part_end,
                        confidence=segment.confidence,
                        words=part_words
                    ))
        
        return Transcription(segments=result_segments, metadata=self.metadata.copy())
    
    def to_srt(self) -> str:
        """
        Convert the transcription to SRT subtitle format.
        
        Returns:
            String in SRT format
        """
        lines = []
        
        for i, segment in enumerate(self.segments):
            # SRT index (1-based)
            lines.append(str(i + 1))
            
            # Time format: HH:MM:SS,mmm --> HH:MM:SS,mmm
            start_time = format_timestamp(segment.start, srt=True)
            end_time = format_timestamp(segment.end, srt=True)
            lines.append(f"{start_time} --> {end_time}")
            
            # Text (could be multi-line)
            lines.append(segment.text)
            
            # Empty line between entries
            lines.append("")
        
        return "\n".join(lines)
    
    def to_vtt(self) -> str:
        """
        Convert the transcription to WebVTT subtitle format.
        
        Returns:
            String in WebVTT format
        """
        lines = ["WEBVTT", ""]
        
        for i, segment in enumerate(self.segments):
            # Optional cue identifier
            lines.append(str(i + 1))
            
            # Time format: HH:MM:SS.mmm --> HH:MM:SS.mmm
            start_time = format_timestamp(segment.start, srt=False)
            end_time = format_timestamp(segment.end, srt=False)
            lines.append(f"{start_time} --> {end_time}")
            
            # Text (could be multi-line)
            lines.append(segment.text)
            
            # Empty line between entries
            lines.append("")
        
        return "\n".join(lines)
    
    def save_subtitles(self, 
                      output_path: Union[str, Path], 
                      format: str = 'srt') -> str:
        """
        Save the transcription as subtitle file.
        
        Args:
            output_path: Path to save the subtitle file
            format: Subtitle format ('srt' or 'vtt')
            
        Returns:
            Path to the saved subtitle file
        """
        # Generate the subtitle content
        if format.lower() == 'srt':
            content = self.to_srt()
        elif format.lower() == 'vtt':
            content = self.to_vtt()
        else:
            raise ValueError(f"Unsupported subtitle format: {format}")
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return str(output_path)
    
    def generate_word_timestamps(self) -> 'Transcription':
        """
        Attempt to generate word-level timestamps if not present.
        This is a simple approximation based on segment duration.
        
        Returns:
            New Transcription object with word timestamps
        """
        result_segments = []
        
        for segment in self.segments:
            # If already has word timestamps, copy as is
            if segment.words:
                result_segments.append(TranscriptionSegment(
                    text=segment.text,
                    start=segment.start,
                    end=segment.end,
                    confidence=segment.confidence,
                    words=segment.words.copy()
                ))
                continue
            
            # Simple word timing approximation
            words = segment.text.split()
            duration = segment.duration()
            
            # Estimate word duration (assuming uniform distribution)
            word_duration = duration / len(words) if words else 0
            
            # Generate word timestamps
            word_data = []
            for i, word in enumerate(words):
                word_start = segment.start + i * word_duration
                word_end = word_start + word_duration
                
                word_data.append({
                    'word': word,
                    'start': word_start,
                    'end': word_end,
                    'confidence': segment.confidence
                })
            
            result_segments.append(TranscriptionSegment(
                text=segment.text,
                start=segment.start,
                end=segment.end,
                confidence=segment.confidence,
                words=word_data
            ))
        
        return Transcription(segments=result_segments, metadata=self.metadata.copy())

def extract_audio_from_video(video_path: Union[str, Path], 
                          output_path: Optional[Union[str, Path]] = None,
                          audio_format: str = 'wav',
                          sample_rate: int = 16000,
                          channels: int = 1) -> str:
    """
    Extract audio from a video file.
    
    Args:
        video_path: Path to the video file
        output_path: Path to save the audio file (if None, creates a temporary file)
        audio_format: Format of the output audio file ('wav', 'mp3', etc.)
        sample_rate: Sample rate of the output audio
        channels: Number of audio channels (1=mono, 2=stereo)
        
    Returns:
        Path to the extracted audio file
    """
    video_path = str(video_path)
    
    # Create temporary file if output_path not provided
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=f'.{audio_format}')
        os.close(fd)
        temp_file = True
    else:
        output_path = str(output_path)
        temp_file = False
    
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Extract audio using ffmpeg
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-ar', str(sample_rate),  # Sample rate
            '-ac', str(channels),  # Channels
            '-y',  # Overwrite output file
            output_path
        ]
        
        logger.info(f"Extracting audio from {video_path} to {output_path}")
        process = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        return output_path
    
    except subprocess.CalledProcessError as e:
        logger.error(f"Error extracting audio: {e.stderr}")
        if temp_file:
            try:
                os.unlink(output_path)
            except (OSError, FileNotFoundError):
                pass
        raise RuntimeError(f"Failed to extract audio: {e}")

def find_silent_segments(audio_path: Union[str, Path],
                       min_silence_duration: float = 1.0,
                       silence_threshold: float = -40,
                       padding: float = 0.2) -> List[Tuple[float, float]]:
    """
    Find silent segments in an audio file.
    
    Args:
        audio_path: Path to the audio file
        min_silence_duration: Minimum duration of silence to detect (seconds)
        silence_threshold: Threshold for silence detection (dB)
        padding: Amount of padding around silence (seconds)
        
    Returns:
        List of (start_time, end_time) tuples for silent segments
    """
    # Load audio file
    audio = AudioSegment.from_file(str(audio_path))
    
    # Convert min_silence_duration from seconds to milliseconds
    min_silence_ms = int(min_silence_duration * 1000)
    padding_ms = int(padding * 1000)
    
    # Detect silent ranges
    silent_ranges = detect_silence(
        audio,
        min_silence_len=min_silence_ms,
        silence_thresh=silence_threshold
    )
    
    # Convert milliseconds to seconds and apply padding
    result = []
    for start_ms, end_ms in silent_ranges:
        start_sec = max(0, (start_ms - padding_ms) / 1000)
        end_sec = min((end_ms + padding_ms) / 1000, len(audio) / 1000)
        result.append((start_sec, end_sec))
    
    return result

def format_timestamp(seconds: float, srt: bool = True) -> str:
    """
    Format a timestamp for subtitle files.
    
    Args:
        seconds: Time in seconds
        srt: Whether to format for SRT (comma separator) or WebVTT (period separator)
        
    Returns:
        Formatted timestamp string
    """
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    seconds = seconds % 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    
    if srt:
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d},{milliseconds:03d}"
    else:
        return f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{milliseconds:03d}"

def transcribe_audio(audio_path: Union[str, Path], 
                   model_name: str = 'base',
                   language: Optional[str] = None,
                   device: Optional[str] = None,
                   word_timestamps: bool = False,
                   batch_size: int = 16,
                   beam_size: int = 5) -> Transcription:
    """
    Transcribe audio using the Whisper model.
    
    Args:
        audio_path: Path to the audio file
        model_name: Whisper model name ('tiny', 'base', 'small', 'medium', 'large')
        language: Language code (e.g., 'en', 'fr') or None for auto-detection
        device: Device to use for inference ('cpu', 'cuda', etc.)
        word_timestamps: Whether to generate word-level timestamps
        batch_size: Batch size for inference
        beam_size: Beam size for inference
        
    Returns:
        Transcription object
    """
    try:
        import whisper
    except ImportError:
        raise ImportError(
            "Please install whisper using: pip install openai-whisper"
        )
    
    # Set device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name, device=device)
    
    logger.info(f"Transcribing audio: {audio_path}")
    options = {
        "language": language,
        "task": "transcribe",
        "beam_size": beam_size,
        "best_of": beam_size,
        "word_timestamps": word_timestamps,
    }
    
    # Run transcription
    result = model.transcribe(str(audio_path), **options)
    
    # Convert to our Transcription format
    segments = []
    for seg in result["segments"]:
        words = []
        if word_timestamps and "words" in seg:
            for word_data in seg["words"]:
                words.append({
                    "word": word_data["word"],
                    "start": word_data["start"],
                    "end": word_data["end"],
                    "confidence": word_data.get("confidence", 1.0)
                })
        
        segment = TranscriptionSegment(
            text=seg["text"].strip(),
            start=seg["start"],
            end=seg["end"],
            confidence=seg.get("confidence", 1.0),
            words=words
        )
        segments.append(segment)
    
    metadata = {
        "model": model_name,
        "language": result.get("language", "unknown"),
        "duration": result.get("duration", 0)
    }
    
    return Transcription(segments=segments, metadata=metadata)

def transcribe_video(video_path: Union[str, Path], 
                   output_json_path: Optional[Union[str, Path]] = None,
                   output_srt_path: Optional[Union[str, Path]] = None,
                   output_vtt_path: Optional[Union[str, Path]] = None,
                   model_name: str = 'base',
                   language: Optional[str] = None,
                   word_timestamps: bool = False,
                   extract_audio_only: bool = False,
                   **kwargs) -> Transcription:
    """
    Transcribe a video file.
    
    Args:
        video_path: Path to the video file
        output_json_path: Path to save the transcription JSON
        output_srt_path: Path to save the SRT subtitles
        output_vtt_path: Path to save the WebVTT subtitles
        model_name: Whisper model name
        language: Language code
        word_timestamps: Whether to generate word-level timestamps
        extract_audio_only: If True, only extract audio without transcribing
        **kwargs: Additional arguments for transcribe_audio
        
    Returns:
        Transcription object
    """
    video_path = str(video_path)
    
    # Extract audio from video
    audio_path = extract_audio_from_video(video_path)
    
    try:
        # If only extracting audio, return empty transcription
        if extract_audio_only:
            logger.info(f"Audio extracted to {audio_path}")
            return Transcription(metadata={"audio_path": audio_path})
        
        # Transcribe the audio
        transcription = transcribe_audio(
            audio_path, 
            model_name=model_name,
            language=language,
            word_timestamps=word_timestamps,
            **kwargs
        )
        
        # Add video path to metadata
        transcription.metadata["video_path"] = video_path
        
        # Save JSON if requested
        if output_json_path:
            transcription.save_to_file(output_json_path)
            logger.info(f"Transcription saved to {output_json_path}")
        
        # Save SRT if requested
        if output_srt_path:
            transcription.save_subtitles(output_srt_path, format='srt')
            logger.info(f"SRT subtitles saved to {output_srt_path}")
        
        # Save WebVTT if requested
        if output_vtt_path:
            transcription.save_subtitles(output_vtt_path, format='vtt')
            logger.info(f"WebVTT subtitles saved to {output_vtt_path}")
        
        return transcription
    
    finally:
        # Clean up the temporary audio file
        try:
            os.unlink(audio_path)
        except (OSError, FileNotFoundError):
            pass 