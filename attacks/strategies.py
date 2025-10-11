"""
Attack strategy definitions for watermark forgery.

Contains strategies for modifying watermarked text while attempting to preserve
the watermark signal. This represents instance-based forgery attacks.
"""

# Watermark-guided search strategies (generate variants, keep high z-scores)
GUIDED_SEARCH_STRATEGIES = {
    "guided_paraphrase": """Generate 5 different paraphrases of the following text that change the meaning or intent while keeping similar structure and vocabulary. Make each version distinct in its message or implication.

Original text: "{text}"

Provide 5 paraphrases separated by '---VARIANT---':""",

    "guided_continuation": """Generate 5 different continuations for the following text that change the overall message or introduce misinformation. Each continuation should be natural and plausible but convey different information.

Text prefix: "{text}"

Provide 5 continuations separated by '---VARIANT---':""",

    "guided_intent_change": """Generate 5 variations of the following text that subtly change the intent, sentiment, or implications while keeping many of the same words and phrases.

Original text: "{text}"

Provide 5 variations separated by '---VARIANT---':"""
}

# Copy-paste attack: no LLM needed, just text manipulation
# These are implemented directly in code, not via prompts

# Legacy strategies (kept for backward compatibility, but less effective)
ATTACK_STRATEGIES = {
    "paraphrase": """Rewrite the following text using different words and sentence structures while maintaining the exact same meaning and information. Make it sound natural and fluent.

Original text: "{text}"

Paraphrased text:""",

    "swap": """Reorder the sentences and phrases in the following text to present the same information in a different sequence. Ensure the reordered text remains coherent and natural.

Original text: "{text}"

Reordered text:""",

    "delete": """Remove any redundant, repetitive, or unnecessary parts from the following text while preserving all the essential information and meaning.

Original text: "{text}"

Condensed text:""",

    "add": """Expand the following text by adding relevant explanatory details, context, or clarifications that enhance understanding without changing the core message.

Original text: "{text}"

Expanded text:""",

    "synonym": """Replace words in the following text with appropriate synonyms while maintaining the exact meaning and natural flow of the text.

Original text: "{text}"

Synonym-replaced text:"""
}


def get_attack_prompt(attack_type: str, text: str) -> str:
    """
    Get formatted attack prompt for given attack type.

    Args:
        attack_type: Type of attack
        text: Original text to attack

    Returns:
        Formatted prompt string

    Raises:
        ValueError: If attack_type is not recognized
    """
    # Check guided search strategies first
    if attack_type in GUIDED_SEARCH_STRATEGIES:
        return GUIDED_SEARCH_STRATEGIES[attack_type].format(text=text)

    # Check legacy strategies
    if attack_type in ATTACK_STRATEGIES:
        return ATTACK_STRATEGIES[attack_type].format(text=text)

    all_types = list(GUIDED_SEARCH_STRATEGIES.keys()) + list(ATTACK_STRATEGIES.keys())
    raise ValueError(f"Unknown attack type: {attack_type}. Valid types: {all_types}")


def is_guided_search_attack(attack_type: str) -> bool:
    """Check if attack type uses guided search (returns multiple variants)."""
    return attack_type in GUIDED_SEARCH_STRATEGIES


def is_copy_paste_attack(attack_type: str) -> bool:
    """Check if attack type uses copy-paste strategy."""
    return attack_type.startswith("copy_paste")
