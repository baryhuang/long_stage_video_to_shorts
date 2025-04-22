#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Highlights module for identifying engaging segments in videos and generating titles.
"""

from src.highlights.analyzer import identify_highlights
from src.highlights.titles import generate_titles

__all__ = ['identify_highlights', 'generate_titles'] 