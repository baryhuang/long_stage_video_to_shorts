#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FontManager class for managing fonts used in video generation.
"""

import os
import logging
import requests
from PIL import ImageFont

logger = logging.getLogger(__name__)


class FontManager:
    def __init__(self):
        self.font_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'fonts')
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