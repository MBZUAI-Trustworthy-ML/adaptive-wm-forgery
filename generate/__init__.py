"""
Text generation module for creating watermarked and unwatermarked samples.

This module provides utilities for generating text samples with or without
watermarks for experimental evaluation.
"""

from .generator import TextGenerator
from .config import GenerateConfig

__all__ = ['TextGenerator', 'GenerateConfig']
