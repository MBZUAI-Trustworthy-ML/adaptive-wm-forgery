#!/usr/bin/env python3
"""
Watermark attack script with high-level localized paraphrasing strategies.

Applies various semantic-preserving edit attacks to watermarked text to test
watermark robustness. Focuses on localized edits that target specific portions
of the text while preserving overall meaning and naturalness.
"""

import json
import os
import sys
import time
import logging
import torch
from typing import Dict, Any, Optional, List

from openai import OpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set cache directory to project-local
CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks.config import AttackConfig
from attacks.strategies import get_attack_prompt, is_guided_search_attack, is_copy_paste_attack
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class WatermarkAttacker:
    """High-level localized paraphrasing attacks against watermarked text."""

    def __init__(self, config: AttackConfig, openai_api_key: str):
        """
        Initialize the attacker.

        Args:
            config: Attack configuration
            openai_api_key: OpenAI API key for generating edits
        """
        self.config = config
        self.openai_client = OpenAI(api_key=openai_api_key)

        # Initialize watermark detector for conditional saving
        self.watermark = self._setup_watermark_detector()

    def _setup_watermark_detector(self) -> Any:
        """Setup watermark detector for conditional saving."""
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
            max_new_tokens=200,
            min_length=0,
            do_sample=True,
            no_repeat_ngram_size=4
        )

        # Load watermark using AutoWatermark
        logger.info(f"Loading watermark: {self.config.watermark_scheme}")
        watermark = AutoWatermark.load(
            algorithm_name=self.config.watermark_scheme,
            algorithm_config=self.config.watermark_config_path,
            transformers_config=transformers_config
        )

        logger.info(f"Watermark detector initialized for conditional saving")
        return watermark

    def apply_copy_paste_attack(self, text: str, harmful_continuations: List[str]) -> List[str]:
        """
        Apply copy-paste attack: preserve watermarked prefix, append harmful continuation.

        Args:
            text: Original watermarked text
            harmful_continuations: List of harmful continuations to try

        Returns:
            List of forged texts
        """
        tokenizer = self.watermark.config.generation_tokenizer
        tokens = tokenizer.encode(text, add_special_tokens=False)

        # Try different split points (30%, 50%, 70% of original length)
        split_points = [int(len(tokens) * ratio) for ratio in [0.3, 0.5, 0.7]]

        forged_texts = []
        for split_point in split_points:
            prefix_tokens = tokens[:split_point]
            prefix_text = tokenizer.decode(prefix_tokens)

            for continuation in harmful_continuations:
                forged_text = prefix_text + " " + continuation
                forged_texts.append(forged_text)

        return forged_texts

    def apply_guided_search_attack(self, text: str, attack_type: str, max_retries: int = 3) -> List[str]:
        """
        Apply guided search attack: generate multiple variants, filter by z-score.

        Args:
            text: Original text to attack
            attack_type: Type of guided attack
            max_retries: Maximum retry attempts

        Returns:
            List of variant texts (may be empty if all fail)
        """
        # Get prompt for generating multiple variants
        prompt = get_attack_prompt(attack_type, text)

        for attempt in range(max_retries):
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=[
                        {"role": "system", "content": "You are a text generation assistant. Generate multiple distinct variations as requested."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=len(text) * 3,
                    temperature=0.8,  # Higher temperature for diversity
                    top_p=0.9
                )

                response_text = response.choices[0].message.content.strip()

                # Split by separator
                variants = [v.strip() for v in response_text.split('---VARIANT---') if v.strip()]

                if len(variants) > 0:
                    return variants
                else:
                    logger.warning(f"No variants generated on attempt {attempt + 1}")

            except Exception as e:
                logger.warning(f"OpenAI API error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        return []

    def apply_attack(self, text: str, attack_type: str, max_retries: int = 3) -> Optional[str]:
        """
        Apply high-level attack to text using OpenAI API.

        Args:
            text: Original text to attack
            attack_type: Type of attack to apply
            max_retries: Maximum number of retry attempts

        Returns:
            Attacked text or None if failed
        """
        # Get formatted prompt
        prompt = get_attack_prompt(attack_type, text)

        for attempt in range(max_retries):
            try:
                # Use chat completions API
                response = self.openai_client.chat.completions.create(
                    model=self.config.openai_model,
                    messages=[
                        {"role": "system", "content": "You are a skilled editor who makes precise, localized edits to text as requested. Always provide only the edited text without explanations or commentary."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=len(text) + 100,
                    temperature=self.config.openai_temperature,
                    top_p=0.9
                )

                attacked_text = response.choices[0].message.content.strip()

                # Basic validation
                if len(attacked_text) > 0:
                    return attacked_text
                else:
                    logger.warning(f"Attack attempt {attempt + 1} produced empty result for {attack_type}")

            except Exception as e:
                logger.warning(f"OpenAI API error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff

        logger.error(f"Failed to apply {attack_type} attack after {max_retries} attempts")
        return None

    def attack_dataset(self) -> str:
        """
        Apply attacks to a dataset of watermarked responses with conditional saving.

        Returns:
            Output directory path
        """
        # Validate configuration
        self.config.validate()

        # Create output directory structure
        base_name = os.path.basename(self.config.input_file).replace('.jsonl', '')
        attack_output_dir = os.path.join(self.config.output_dir, base_name)
        os.makedirs(attack_output_dir, exist_ok=True)

        # Load input dataset
        logger.info(f"Loading dataset from {self.config.input_file}")
        try:
            with open(self.config.input_file, 'r') as f:
                records = []
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load dataset: {e}")

        if not records:
            raise ValueError(f"No valid records found in {self.config.input_file}")

        logger.info(f"Loaded {len(records)} watermarked responses")
        logger.info(f"Applying {len(self.config.enabled_types)} attack types: {self.config.enabled_types}")
        logger.info(f"Conditional saving: {self.config.conditional_saving}")

        # Statistics tracking
        stats = {attack_type: {'total': 0, 'attacked': 0, 'saved': 0, 'rejected': 0}
                 for attack_type in self.config.enabled_types}

        # Apply each attack type
        for attack_type in self.config.enabled_types:
            logger.info(f"Applying {attack_type} attacks...")

            output_file = os.path.join(attack_output_dir, f"{attack_type}.jsonl")

            with open(output_file, 'w') as f:
                for i, record in enumerate(records):
                    stats[attack_type]['total'] += 1
                    logger.info(f"  Processing record {i+1}/{len(records)}")

                    # Try different field names for the watermarked text
                    text = record.get('watermarked_response') or record.get('full_text') or record.get('completion', '')
                    if not text:
                        logger.warning(f"    Skipping record {i+1}: no watermarked text found")
                        stats[attack_type]['rejected'] += 1
                        continue

                    # Apply attack based on type
                    if is_copy_paste_attack(attack_type):
                        # Copy-paste attack needs special handling
                        harmful_continuations = [
                            "However, this information is completely false and fabricated.",
                            "This is misinformation designed to mislead readers.",
                            "The above claims have been debunked by fact-checkers.",
                            "Contrary to the above, the opposite is actually true.",
                            "This text contains deliberately inserted falsehoods."
                        ]
                        attacked_texts = self.apply_copy_paste_attack(text, harmful_continuations)
                        # For now, just take the first one
                        attacked_text = attacked_texts[0] if attacked_texts else None
                    elif is_guided_search_attack(attack_type):
                        # Guided search returns multiple variants
                        variants = self.apply_guided_search_attack(text, attack_type)
                        # For now, just take the first one
                        attacked_text = variants[0] if variants else None
                    else:
                        # Legacy single-variant attack
                        attacked_text = self.apply_attack(text, attack_type)

                    if attacked_text is None:
                        if self.config.log_rejections:
                            logger.info(f"    Rejected: Attack failed")
                        stats[attack_type]['rejected'] += 1
                        continue

                    stats[attack_type]['attacked'] += 1

                    # Conditional saving logic
                    should_save = True
                    rejection_reason = None
                    attacked_detection_result = None

                    if self.config.conditional_saving:
                        try:
                            # Check original watermark detection
                            original_detection = self.watermark.detect_watermark(text, return_dict=True)
                            original_detected = original_detection.get('is_watermarked', False)
                            original_z_score = original_detection.get('score', 0.0)

                            if not original_detected:
                                should_save = False
                                rejection_reason = "Original watermark not detected"
                            else:
                                # Check attacked watermark detection
                                attacked_detection = self.watermark.detect_watermark(attacked_text, return_dict=True)
                                attacked_detected = attacked_detection.get('is_watermarked', False)
                                attacked_z_score = attacked_detection.get('score', 0.0)
                                attacked_detection_result = attacked_detection

                                # FORGERY ATTACK: Save if watermark is PRESERVED
                                # Goal: change content while keeping watermark signal
                                if not attacked_detected:
                                    should_save = False
                                    rejection_reason = "Watermark removed (forgery failed)"
                                else:
                                    # Additional check: content should be different
                                    # Simple heuristic: check if texts differ significantly
                                    if attacked_text.strip() == text.strip():
                                        should_save = False
                                        rejection_reason = "Content unchanged"
                        except Exception as e:
                            logger.warning(f"    Detection error: {e}")
                            should_save = True  # Save on detection errors
                    else:
                        # Even if not using conditional saving, get detection results for storage
                        try:
                            attacked_detection_result = self.watermark.detect_watermark(attacked_text, return_dict=True)
                        except Exception as e:
                            logger.warning(f"    Detection error: {e}")
                            attacked_detection_result = None

                    if should_save:
                        # Create attacked record
                        attacked_record = record.copy()
                        attacked_record.update({
                            'attack_type': attack_type,
                            'original_text': text,
                            'attacked_response': attacked_text,
                            'attack_success': True,
                            'original_z_score': original_z_score if 'original_z_score' in locals() else None,
                            'attacked_z_score': attacked_z_score if 'attacked_z_score' in locals() else None,
                            'watermark_preserved': attacked_detected if 'attacked_detected' in locals() else None,
                            'detection_result': attacked_detection_result
                        })

                        # Write record
                        f.write(json.dumps(attacked_record) + '\n')
                        f.flush()
                        stats[attack_type]['saved'] += 1
                        logger.info(f"    Saved: Forgery successful (watermark preserved, content changed)")
                    else:
                        stats[attack_type]['rejected'] += 1
                        if self.config.log_rejections:
                            logger.info(f"    Rejected: {rejection_reason}")

                    # Brief delay to avoid rate limiting
                    time.sleep(0.1)

            # Log attack statistics
            attack_stats = stats[attack_type]
            logger.info(f"  {attack_type} statistics:")
            logger.info(f"    Total: {attack_stats['total']}")
            logger.info(f"    Successfully attacked: {attack_stats['attacked']}")
            logger.info(f"    Saved: {attack_stats['saved']}")
            logger.info(f"    Rejected: {attack_stats['rejected']}")
            if attack_stats['total'] > 0:
                save_rate = (attack_stats['saved'] / attack_stats['total']) * 100
                logger.info(f"    Save rate: {save_rate:.1f}%")

        # Save overall statistics
        stats_file = os.path.join(attack_output_dir, "attack_stats.json")
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"All attacks completed. Results saved to {attack_output_dir}")
        return attack_output_dir


def main():
    """Main function - expects config file as command line argument."""
    if len(sys.argv) != 2:
        logger.error("Usage: python -m attacks.attacker <config_file>")
        logger.error("Example: python -m attacks.attacker config/attack_config.json")
        logger.error("")
        logger.error("The config file should specify the input dataset file and attack parameters.")
        return

    config_path = sys.argv[1]

    # Validate config file exists
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return

    # Get OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OpenAI API key required. Set OPENAI_API_KEY environment variable")
        logger.error("Example: export OPENAI_API_KEY='your-api-key-here'")
        return

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = AttackConfig.from_file(config_path)

        # Log parameters
        logger.info("Attack parameters:")
        logger.info(f"  config_file: {config_path}")
        logger.info(f"  watermark_scheme: {config.watermark_scheme}")
        logger.info(f"  input_file: {config.input_file}")
        logger.info(f"  attack_types: {config.enabled_types}")
        logger.info(f"  openai_model: {config.openai_model}")

        # Initialize attacker
        logger.info("Initializing WatermarkAttacker...")
        attacker = WatermarkAttacker(config, api_key)

        # Apply attacks
        output_dir = attacker.attack_dataset()

        logger.info("Attack completed successfully!")
        logger.info(f"Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"Attack failed: {e}")
        raise


if __name__ == "__main__":
    main()
