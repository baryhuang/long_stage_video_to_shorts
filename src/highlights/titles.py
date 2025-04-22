#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Title generation module for creating engaging titles for video highlights.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Any, Optional

from anthropic import Anthropic

from src.models.transcription import TranscriptionSegment
from src.models.highlight import HighlightClip

logger = logging.getLogger(__name__)

def generate_titles(segments_data: Dict[str, Any]) -> Tuple[str, List[HighlightClip]]:
    """
    Use Claude 3.5 to generate engaging titles for the segments
    
    Args:
        segments_data: Dictionary containing segment information and theme
    
    Returns:
        Tuple of (main_title, List[HighlightClip])
    """
    logger.info("Generating segment title using Claude")
    
    # Get Claude API client
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
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