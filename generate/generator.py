#!/usr/bin/env python3
"""
Text generation script for creating watermarked and unwatermarked samples.

Generates text samples from prompts using specified watermarking schemes
and saves them in JSONL format with detection statistics.
"""

import json
import os
import sys
import logging
import torch
from typing import List, Dict, Any
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set cache directory to project-local
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from generate.config import GenerateConfig
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextGenerator:
    """Text generator for creating watermarked/unwatermarked samples."""

    def __init__(self, config: GenerateConfig):
        """
        Initialize the text generator.

        Args:
            config: Generation configuration
        """
        self.config = config
        self.watermark = self._setup_watermark()

    def _setup_watermark(self) -> Any:
        """Setup watermark instance for generation and detection."""
        # Determine device
        if self.config.device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.config.device

        logger.info(f"Loading model: {self.config.model_name}")
        logger.info(f"Using device: {device}")

        # Load model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(self.config.model_name, cache_dir=CACHE_DIR).to(device)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, cache_dir=CACHE_DIR)

        # Set pad token if not exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Configure transformers
        transformers_config = TransformersConfig(
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=self.config.max_new_tokens,
            min_length=self.config.min_length,
            do_sample=self.config.do_sample,
            no_repeat_ngram_size=self.config.no_repeat_ngram_size,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k
        )

        # Load watermark using AutoWatermark
        logger.info(f"Loading watermark: {self.config.watermark_scheme}")
        watermark = AutoWatermark.load(
            algorithm_name=self.config.watermark_scheme,
            algorithm_config=self.config.watermark_config_path,
            transformers_config=transformers_config
        )

        logger.info(f"Watermark and model loaded successfully")
        return watermark

    def load_prompts(self) -> List[str]:
        """
        Load prompts from file or config.

        Returns:
            List of prompt strings

        Raises:
            ValueError: If prompts cannot be loaded
        """
        if self.config.prompts:
            # Use prompts from config
            prompts = self.config.prompts
            logger.info(f"Using {len(prompts)} prompts from config")
            return prompts[:self.config.num_samples]

        # Load from file
        prompts_file = self.config.prompts_file
        logger.info(f"Loading prompts from {prompts_file}")

        prompts = []

        if prompts_file.endswith('.txt'):
            # Plain text file, one prompt per line
            with open(prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]

        elif prompts_file.endswith('.json'):
            # JSON file with prompts array OR JSONL format
            with open(prompts_file, 'r') as f:
                first_line = f.readline().strip()
                f.seek(0)  # Reset to beginning

                # Try parsing first line as JSON to detect JSONL
                try:
                    json.loads(first_line)
                    # It's JSONL format
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            # Try common field names
                            prompt = record.get('prompt') or record.get('text') or record.get('input')
                            if prompt:
                                prompts.append(prompt)
                except:
                    # It's regular JSON
                    f.seek(0)
                    data = json.load(f)
                    if isinstance(data, list):
                        prompts = data
                    elif isinstance(data, dict) and 'prompts' in data:
                        prompts = data['prompts']
                    else:
                        raise ValueError("JSON file must contain a list or a dict with 'prompts' key")

        elif prompts_file.endswith('.jsonl'):
            # JSONL file, extract 'prompt' or 'text' field from each line
            with open(prompts_file, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        # Try common field names
                        prompt = record.get('prompt') or record.get('text') or record.get('input')
                        if prompt:
                            prompts.append(prompt)

        if not prompts:
            raise ValueError(f"No prompts loaded from {prompts_file}")

        logger.info(f"Loaded {len(prompts)} prompts from file")

        # Limit to num_samples
        return prompts[:self.config.num_samples]

    def generate_sample(self, prompt: str) -> Dict[str, Any]:
        """
        Generate a single text sample from prompt.

        Args:
            prompt: Input prompt string

        Returns:
            Dictionary with prompt, completion, and detection results
        """
        # Generate text (watermarked or unwatermarked)
        if self.config.watermarked:
            full_text = self.watermark.generate_watermarked_text(prompt)
        else:
            full_text = self.watermark.generate_unwatermarked_text(prompt)

        # Extract completion (remove prompt from output)
        completion = full_text[len(prompt):].strip() if full_text.startswith(prompt) else full_text

        # Detect watermark to get z-score and other metrics
        try:
            detection_result = self.watermark.detect_watermark(full_text, return_dict=True)
            z_score = detection_result.get('score', 0.0)
            is_watermarked = detection_result.get('is_watermarked', False)

            # Try to extract key if available (for multi-key schemes)
            key = detection_result.get('key', None)
            pred_message = detection_result.get('pred_message', None)

        except Exception as e:
            logger.warning(f"Detection failed: {e}")
            z_score = 0.0
            is_watermarked = False
            key = None
            pred_message = None

        # Create record
        record = {
            'prompt': prompt,
            'completion': completion,
            'full_text': full_text,
            'z_score': z_score,
            'is_watermarked': is_watermarked
        }

        # Add optional fields if available
        if key is not None:
            record['key'] = key
        if pred_message is not None:
            record['pred_message'] = pred_message

        return record

    def generate_dataset(self) -> str:
        """
        Generate complete dataset from prompts.

        Returns:
            Output file path
        """
        # Validate configuration
        self.config.validate()

        # Load prompts
        prompts = self.load_prompts()

        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)

        # Get output path
        output_path = self.config.get_output_path()

        logger.info(f"Generating {len(prompts)} samples...")
        logger.info(f"Watermarked: {self.config.watermarked}")
        logger.info(f"Output: {output_path}")

        # Generate samples
        successful = 0
        failed = 0

        with open(output_path, 'w') as f:
            for i, prompt in enumerate(tqdm(prompts, desc="Generating")):
                try:
                    record = self.generate_sample(prompt)
                    f.write(json.dumps(record) + '\n')
                    f.flush()
                    successful += 1
                except Exception as e:
                    logger.error(f"Failed to generate sample {i+1}: {e}")
                    failed += 1
                    continue

        # Log statistics
        logger.info(f"Generation complete!")
        logger.info(f"  Successful: {successful}")
        logger.info(f"  Failed: {failed}")
        logger.info(f"  Output: {output_path}")

        return output_path


def main():
    """Main function - expects config file as command line argument."""
    if len(sys.argv) != 2:
        logger.error("Usage: python -m generate.generator <config_file>")
        logger.error("Example: python -m generate.generator config/generate_config.json")
        logger.error("")
        logger.error("The config file should specify prompts, model, and generation parameters.")
        return

    config_path = sys.argv[1]

    # Validate config file exists
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = GenerateConfig.from_file(config_path)

        # Log parameters
        logger.info("Generation parameters:")
        logger.info(f"  config_file: {config_path}")
        logger.info(f"  watermark_scheme: {config.watermark_scheme}")
        logger.info(f"  model: {config.model_name}")
        logger.info(f"  num_samples: {config.num_samples}")
        logger.info(f"  watermarked: {config.watermarked}")

        # Initialize generator
        logger.info("Initializing TextGenerator...")
        generator = TextGenerator(config)

        # Generate dataset
        output_path = generator.generate_dataset()

        logger.info("Generation completed successfully!")
        logger.info(f"Dataset saved to: {output_path}")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        raise


if __name__ == "__main__":
    main()
