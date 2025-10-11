"""
Configuration handling for text generation.

Manages loading and validation of generation configurations from JSON files.
"""

import os
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


@dataclass
class GenerateConfig:
    """Configuration class for text generation."""

    # Watermark configuration
    watermark_scheme: str
    watermark_config_path: str

    # Model configuration
    model_name: str
    device: str = "auto"

    # Generation configuration
    max_new_tokens: int = 200
    min_length: int = 0
    do_sample: bool = True
    no_repeat_ngram_size: int = 4
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0

    # Input/Output configuration
    prompts_file: str = ""
    prompts: Optional[List[str]] = None
    num_samples: int = 50
    output_dir: str = "data"
    watermarked: bool = True

    @classmethod
    def from_file(cls, config_path: str) -> 'GenerateConfig':
        """
        Load configuration from JSON file.

        Args:
            config_path: Path to JSON configuration file

        Returns:
            GenerateConfig instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Validate required fields
        required_fields = ['watermark_scheme', 'watermark_config', 'model']
        for field in required_fields:
            if field not in config_dict:
                raise ValueError(f"Missing required field in config: {field}")

        # Extract model configuration
        model_config = config_dict['model']
        if 'model_name' not in model_config:
            raise ValueError("Missing 'model_name' in model config")

        # Extract generation configuration
        gen_config = config_dict.get('generation', {})

        # Extract input/output configuration
        io_config = config_dict.get('input_output', {})

        # Validate that either prompts_file or prompts list is provided
        prompts_file = io_config.get('prompts_file', '')
        prompts = io_config.get('prompts', None)

        if not prompts_file and not prompts:
            raise ValueError("Must provide either 'prompts_file' or 'prompts' list in config")

        return cls(
            watermark_scheme=config_dict['watermark_scheme'],
            watermark_config_path=config_dict['watermark_config'],
            model_name=model_config['model_name'],
            device=model_config.get('device', 'auto'),
            max_new_tokens=gen_config.get('max_new_tokens', 200),
            min_length=gen_config.get('min_length', 0),
            do_sample=gen_config.get('do_sample', True),
            no_repeat_ngram_size=gen_config.get('no_repeat_ngram_size', 4),
            temperature=gen_config.get('temperature', 1.0),
            top_p=gen_config.get('top_p', 1.0),
            top_k=gen_config.get('top_k', 0),
            prompts_file=prompts_file,
            prompts=prompts,
            num_samples=io_config.get('num_samples', 50),
            output_dir=io_config.get('output_dir', 'data'),
            watermarked=io_config.get('watermarked', True)
        )

    def validate(self) -> None:
        """
        Validate configuration values.

        Raises:
            ValueError: If configuration is invalid
        """
        # Validate watermark config exists
        if not os.path.exists(self.watermark_config_path):
            raise ValueError(f"Watermark config not found: {self.watermark_config_path}")

        # Validate prompts source
        if self.prompts_file:
            if not os.path.exists(self.prompts_file):
                raise ValueError(f"Prompts file not found: {self.prompts_file}")

            # Validate file format
            valid_extensions = ['.txt', '.json', '.jsonl']
            if not any(self.prompts_file.endswith(ext) for ext in valid_extensions):
                raise ValueError(f"Prompts file must be one of {valid_extensions}")

        elif self.prompts:
            if not isinstance(self.prompts, list) or len(self.prompts) == 0:
                raise ValueError("Prompts must be a non-empty list")

        # Validate generation parameters
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive: {self.max_new_tokens}")

        if not 0 <= self.temperature <= 2:
            raise ValueError(f"temperature must be between 0 and 2: {self.temperature}")

        if not 0 <= self.top_p <= 1:
            raise ValueError(f"top_p must be between 0 and 1: {self.top_p}")

        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be positive: {self.num_samples}")

    def get_output_filename(self) -> str:
        """
        Generate output filename based on configuration.

        Returns:
            Filename string
        """
        # Extract model short name (e.g., "mistral-7b" from full path)
        model_short = self.model_name.split('/')[-1].lower()
        watermark_type = "watermarked" if self.watermarked else "unwatermarked"

        return f"{model_short}_{self.watermark_scheme.lower()}_{watermark_type}.jsonl"

    def get_output_path(self) -> str:
        """
        Get full output path.

        Returns:
            Full path string
        """
        return os.path.join(self.output_dir, self.get_output_filename())

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'watermark_scheme': self.watermark_scheme,
            'watermark_config': self.watermark_config_path,
            'model': {
                'model_name': self.model_name,
                'device': self.device
            },
            'generation': {
                'max_new_tokens': self.max_new_tokens,
                'min_length': self.min_length,
                'do_sample': self.do_sample,
                'no_repeat_ngram_size': self.no_repeat_ngram_size,
                'temperature': self.temperature,
                'top_p': self.top_p,
                'top_k': self.top_k
            },
            'input_output': {
                'prompts_file': self.prompts_file,
                'prompts': self.prompts,
                'num_samples': self.num_samples,
                'output_dir': self.output_dir,
                'watermarked': self.watermarked
            }
        }
