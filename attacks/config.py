"""
Configuration handling for watermark attacks.

Manages loading and validation of attack configurations from JSON files.
"""

import os
import json
from typing import Dict, Any, List
from dataclasses import dataclass, field


@dataclass
class AttackConfig:
    """Configuration class for watermark attacks."""

    # Watermark configuration
    watermark_scheme: str
    watermark_config_path: str

    # Attack configuration
    input_file: str
    output_dir: str
    enabled_types: List[str]
    conditional_saving: bool = True
    log_rejections: bool = False

    # OpenAI configuration
    openai_model: str = "gpt-4o"
    openai_temperature: float = 0.3

    # Model configuration (for transformers)
    model_name: str = "mistralai/Mistral-7B-v0.1"
    device: str = "auto"

    @classmethod
    def from_file(cls, config_path: str) -> 'AttackConfig':
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to JSON configuration file

        Returns:
            AttackConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Validate required fields
        required_fields = ['watermark_scheme', 'watermark_config', 'attacks']
        for field in required_fields:
            if field not in config_dict:
                raise ValueError(f"Missing required field in config: {field}")

        # Extract attack configuration
        attack_config = config_dict['attacks']
        required_attack_fields = ['input_file', 'output_dir', 'enabled_types']
        for field in required_attack_fields:
            if field not in attack_config:
                raise ValueError(f"Missing required field in attacks config: {field}")

        # Extract OpenAI configuration
        openai_config = config_dict.get('openai', {})

        # Extract model configuration
        model_config = config_dict.get('model', {})

        return cls(
            watermark_scheme=config_dict['watermark_scheme'],
            watermark_config_path=config_dict['watermark_config'],
            input_file=attack_config['input_file'],
            output_dir=attack_config['output_dir'],
            enabled_types=attack_config['enabled_types'],
            conditional_saving=attack_config.get('conditional_saving', True),
            log_rejections=attack_config.get('log_rejections', False),
            openai_model=openai_config.get('model', 'gpt-4o'),
            openai_temperature=openai_config.get('temperature', 0.3),
            model_name=model_config.get('model_name', 'mistralai/Mistral-7B-v0.1'),
            device=model_config.get('device', 'auto')
        )

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate input file exists
        if not os.path.exists(self.input_file):
            raise ValueError(f"Input file not found: {self.input_file}")

        # Validate input file is JSONL
        if not self.input_file.endswith('.jsonl'):
            raise ValueError(f"Input file must be JSONL format: {self.input_file}")

        # Validate watermark config exists
        if not os.path.exists(self.watermark_config_path):
            raise ValueError(f"Watermark config not found: {self.watermark_config_path}")

        # Validate attack types
        valid_attack_types = [
            # Legacy attacks
            'paraphrase', 'swap', 'delete', 'add', 'synonym',
            # New watermark-preserving attacks
            'guided_paraphrase', 'guided_continuation', 'guided_intent_change',
            'copy_paste'
        ]
        for attack_type in self.enabled_types:
            if attack_type not in valid_attack_types:
                raise ValueError(f"Invalid attack type: {attack_type}. Valid types: {valid_attack_types}")

        # Validate temperature
        if not 0 <= self.openai_temperature <= 2:
            raise ValueError(f"OpenAI temperature must be between 0 and 2: {self.openai_temperature}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'watermark_scheme': self.watermark_scheme,
            'watermark_config': self.watermark_config_path,
            'attacks': {
                'input_file': self.input_file,
                'output_dir': self.output_dir,
                'enabled_types': self.enabled_types,
                'conditional_saving': self.conditional_saving,
                'log_rejections': self.log_rejections
            },
            'openai': {
                'model': self.openai_model,
                'temperature': self.openai_temperature
            },
            'model': {
                'model_name': self.model_name,
                'device': self.device
            }
        }
