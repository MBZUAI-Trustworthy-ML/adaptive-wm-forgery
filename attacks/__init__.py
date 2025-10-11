"""
Watermark attack module for testing robustness against instance-based forgery attacks.

This module provides high-level localized paraphrasing strategies to attack watermarked text.
"""

from .attacker import WatermarkAttacker
from .config import AttackConfig
from .strategies import ATTACK_STRATEGIES

__all__ = ['WatermarkAttacker', 'AttackConfig', 'ATTACK_STRATEGIES']
