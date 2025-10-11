# Adaptive Watermark Forgery Detection

A defense against instance-based watermark forgery attacks on large language models using statistical distribution analysis.

## Overview

This project implements a novel defense mechanism that detects instance-based watermark forgeries by exploiting the fact that single-sample attackers cannot replicate the expected distribution of watermark detection statistics (z-scores) for a specific prompt context.

### Core Idea

**Instance-based attackers** can only optimize their forgery to match ONE watermarked sample's z-score, but cannot predict the prompt-specific distribution of z-scores. Our detection algorithm:

1. Extracts first k tokens from suspicious text as conditioning prompt
2. Regenerates n samples from the watermarked model using that prefix
3. Compares suspicious text's z-score against the distribution of regenerated samples
4. Flags as forged if z-score is statistical outlier (>3σ from mean)

**Why it works:** Genuine texts naturally fall within the expected distribution, while forgeries become statistical outliers.

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/adaptive-wm-forgery.git
cd adaptive-wm-forgery

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
adaptive-wm-forgery/
├── attacks/              # Attack implementations
│   ├── attacker.py       # Main attack orchestration
│   ├── strategies.py     # Attack strategies (paraphrase, swap, etc.)
│   └── config.py         # Attack configuration
├── generate/             # Text generation
│   ├── generator.py      # Watermarked/unwatermarked generation
│   └── config.py         # Generation configuration
├── watermark/            # Watermarking schemes (MarkLLM)
│   ├── kgw/              # KGW watermark
│   ├── unigram/          # Unigram watermark
│   └── ...               # Other schemes
├── config/               # Configuration files
│   ├── attack_config.json
│   ├── generate_config.json
│   └── *.json            # Watermark scheme configs
├── dataset/              # Datasets (C4, CNN/DailyMail, etc.)
├── data/                 # Generated samples output
└── notebooks/            # Validation and experiments
```

## Quick Start

### 1. Generate Watermarked Samples

```bash
# Configure generation in config/generate_config.json
python -m generate.generator config/generate_config.json
```

**Output format** (JSONL):
```json
{
  "prompt": "The quick brown fox...",
  "completion": "jumps over the lazy dog...",
  "z_score": 4.5,
  "is_watermarked": true,
  "key": "..."
}
```

### 2. Apply Instance-Based Attacks

```bash
# Set OpenAI API key
export OPENAI_API_KEY='your-key-here'

# Configure attacks in config/attack_config.json
python -m attacks.attacker config/attack_config.json
```

**Attack strategies:**
- `paraphrase`: Rewrite using different words/structures
- `swap`: Reorder sentences/phrases
- `delete`: Remove redundant parts
- `add`: Expand with explanatory details
- `synonym`: Replace words with synonyms

### 3. Detect Forgeries

See `notebooks/validation.ipynb` for the complete detection pipeline and experiments.

## Detection Algorithm

```python
def detect_instance_forgery(suspicious_text, k=30, n=50):
    """
    Detect if text is an instance-based forgery.

    Args:
        suspicious_text: Text to evaluate
        k: Prefix length (tokens to use as prompt)
        n: Number of regenerations

    Returns:
        is_forged: Boolean indicating if text is forged
    """
    # 1. Extract prefix
    prefix = suspicious_text[:k]

    # 2. Regenerate n samples
    regenerated_samples = [generate_watermarked(prefix) for _ in range(n)]

    # 3. Compute z-scores
    suspicious_z = compute_z_score(suspicious_text)
    legitimate_z_scores = [compute_z_score(s) for s in regenerated_samples]

    # 4. Statistical test
    μ = mean(legitimate_z_scores)
    σ = std(legitimate_z_scores)

    return abs(suspicious_z - μ) > 3 * σ
```

## Supported Watermarking Schemes

This project builds on the [MarkLLM](https://github.com/THU-BPM/MarkLLM) framework:

- **KGW** (Kirchenbauer et al., 2023)
- **Unigram** (Kirchenbauer et al., 2023)
- **SWEET** (Zhao et al., 2024)
- **UPV**, **SIR**, **XSIR**, **Unbiased**
- **DIP**, **EWD**, **EXP**, **EXPGumbel**
- **SynthID**, **TS**, **PF**, **MorphMark**
- **Adaptive**, **KSEMSTAMP**

## Experiments

### Primary Validation

1. **Z-score variance analysis**
   - Measure σ of regenerations from same prefix
   - Goal: σ < 1.0 for tight distribution

2. **Attack detection**
   - Test on instance-based attacks (paraphrase, synonym, etc.)
   - Goal: Detection rate > 95%

3. **False positive rate**
   - Ensure genuine samples not flagged
   - Goal: FPR < 1%

4. **Hyperparameter optimization**
   - Determine optimal k (prefix length: 20-50)
   - Determine optimal n (regenerations: 30-50)

### Running Experiments

```bash
# Launch Jupyter
jupyter notebook

# Open and run notebooks/validation.ipynb
```

## Configuration

### Generation Config (`config/generate_config.json`)

```json
{
  "watermark_scheme": "KGW",
  "watermark_config": "config/KGW.json",
  "model": {
    "model_name": "mistralai/Mistral-7B-v0.1",
    "device": "auto"
  },
  "generation": {
    "max_new_tokens": 200,
    "temperature": 1.0,
    "do_sample": true
  },
  "input_output": {
    "prompts_file": "dataset/c4/processed_c4.json",
    "num_samples": 50,
    "watermarked": true
  }
}
```

### Attack Config (`config/attack_config.json`)

```json
{
  "watermark_scheme": "KGW",
  "watermark_config": "config/KGW.json",
  "attacks": {
    "input_file": "data/mistral-7b_kgw_watermarked.jsonl",
    "output_dir": "results/attacks",
    "enabled_types": ["paraphrase", "swap", "delete", "add", "synonym"],
    "conditional_saving": true
  },
  "openai": {
    "model": "gpt-4o",
    "temperature": 0.3
  }
}
```

## Expected Results

Based on our hypothesis:

- **Z-score variance**: σ < 1.0 for same-prefix regenerations
- **Detection rate**: > 95% on instance-based forgeries
- **False positive rate**: < 1% on genuine samples
- **Clear separation**: Distinct z-score distributions between genuine and forged texts

## Research Context

### Our Contribution

A simple, effective defense against instance-based attacks that:
- Uses exact prefix (no reconstruction error)
- Works on shorter texts (< 1000 tokens)
- Complements multi-key defense (complete coverage)
- Has clear statistical foundation and interpretability

### Related Work

- **Multi-key defense** (Aremu et al., 2024): Defends against learning-based attacks
- **Statistical artifact detection** (Gloaguen et al., 2025): Requires 3000+ tokens, uses prompt reconstruction

Our method is complementary to these approaches, specifically targeting instance-based attacks.

## Citation

```bibtex
@article{aremu2024adaptive,
  title={Adaptive Watermark Forgery Detection via Statistical Distribution Analysis},
  author={Aremu, T. et al.},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built on [MarkLLM](https://github.com/THU-BPM/MarkLLM) framework
- Inspired by work on multi-key watermarking (Aremu et al., 2024)
- Attack strategies adapted from watermark robustness literature
