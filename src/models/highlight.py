#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
HighlightClip model class for representing potential highlight clips.
"""


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