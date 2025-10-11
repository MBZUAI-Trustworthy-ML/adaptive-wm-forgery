# Watermark Forgery Attack Implementation

## Overview

Implemented **instance-based watermark forgery attacks** that attempt to **preserve the watermark signal** while **changing the content**. This is the correct threat model for testing your detection defense.

## Attack Strategies

### 1. Watermark-Guided Search (3 variants)

**Concept:** Generate multiple content variations, filter for those that maintain high z-scores.

#### a) `guided_paraphrase`
- Generates 5 paraphrases that change meaning/intent
- Keeps similar structure and vocabulary
- Tests if token overlap preserves watermark

#### b) `guided_continuation`
- Takes prefix, generates 5 different continuations
- Introduces misinformation or changes message
- Tests if continuing from watermarked prefix preserves signal

#### c) `guided_intent_change`
- Generates 5 variations with subtle intent changes
- Keeps many same words/phrases
- Tests minimal-edit forgery

**Implementation:**
```python
# Generate 5 variants via OpenAI
variants = apply_guided_search_attack(text, attack_type)

# Filter by z-score
for variant in variants:
    z_score = watermark.detect(variant)
    if z_score > threshold:
        save(variant)  # Successful forgery!
```

**Expected Success Rate:** 20-40%
- Some variants will accidentally preserve greenlist tokens
- Higher token overlap → higher preservation chance

### 2. Copy-Paste Attack

**Concept:** Take watermarked prefix, append harmful continuation.

**Implementation:**
```python
# Split at 30%, 50%, 70% of original length
for split_ratio in [0.3, 0.5, 0.7]:
    prefix = original_text[:split_ratio]
    for harmful_continuation in continuations:
        forged = prefix + harmful_continuation
        if watermark.detect(forged) > threshold:
            save(forged)  # Success!
```

**Expected Success Rate:** 50-70%
- Watermark in prefix carries signal
- KGW uses context window, so early tokens count
- Longer prefix → higher success rate

## Key Changes from Original

### ❌ OLD (WRONG): Remove Watermark
```python
if attacked_detected:
    reject()  # Rejected texts with watermark
```

### ✅ NEW (CORRECT): Preserve Watermark
```python
if attacked_detected and content_changed:
    save()  # Save forgeries that keep watermark
else:
    reject()  # Reject if watermark removed
```

## Output Format

Successful forgeries saved with full provenance:

```json
{
  "attack_type": "guided_paraphrase",
  "original_text": "...",
  "attacked_response": "forged version...",
  "original_z_score": 8.78,
  "attacked_z_score": 7.32,
  "watermark_preserved": true,
  "attack_success": true
}
```

## Certainty Assessment

| Attack Type | Watermark Preservation Rate | Reasoning |
|-------------|---------------------------|-----------|
| **Copy-paste** | **50-70%** ✅ | Prefix carries signal, simple and effective |
| **Guided paraphrase** | **20-40%** | Token overlap accidentally preserves some greenlist |
| **Guided continuation** | **40-60%** | Builds on watermarked prefix |
| **Guided intent change** | **25-45%** | Minimal edits → higher token overlap |

## Why Your Defense Will Work

Even successful forgeries will be **statistical outliers**:

```
Genuine samples from same prefix:
  z-scores: [7.2, 7.8, 7.5, 7.4, 7.6]  → μ=7.5, σ=0.2

Forged sample:
  z-score: 5.1  → |5.1 - 7.5| > 3×0.2 → DETECTED!
```

**Key insight:** Attacker can hit the *threshold* but can't match the *distribution* for that specific prompt.

## Configuration

Updated `config/attack_config.json`:
```json
{
  "attacks": {
    "enabled_types": [
      "guided_paraphrase",
      "guided_continuation",
      "guided_intent_change",
      "copy_paste"
    ],
    "conditional_saving": true  // Only save if watermark preserved
  }
}
```

## Next Steps

1. Set OpenAI API key: `export OPENAI_API_KEY='your-key'`
2. Run attacks: `python -m attacks.attacker config/attack_config.json`
3. Measure actual success rates
4. Run validation notebook to test detection

## Expected Experimental Results

- **Forgery success rate:** 30-60% of attempts will preserve watermark
- **Detection rate:** >95% of successful forgeries will be detected as outliers
- **False positive rate:** <1% of genuine samples flagged
- **Key finding:** Instance-based attacks can't replicate prompt-specific z-score distributions
