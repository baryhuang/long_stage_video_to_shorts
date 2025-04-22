#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Highlight analysis module for identifying engaging segments in videos.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

import openai
from anthropic import Anthropic

from src.models.transcription import TranscriptionSegment

logger = logging.getLogger(__name__)

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
        
        # Get API clients
        openai_client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
        anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
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