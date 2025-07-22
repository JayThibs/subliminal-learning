"""
Utilities for creating multigrader configurations to avoid reward hacking.

Based on the OpenAI RL API documentation, multigraders can combine multiple
scoring functions to create more robust reward signals.
"""

from typing import Dict, List, Any
from loguru import logger


def create_multigrader_config(
    graders: Dict[str, Dict[str, Any]], 
    expression: str
) -> Dict[str, Any]:
    """
    Create a multigrader configuration for OpenAI's RL API.
    
    Args:
        graders: Dictionary mapping grader names to grader configs
        expression: Arithmetic expression combining grader outputs
        
    Returns:
        Multigrader configuration dict
    """
    return {
        "type": "multi",
        "graders": graders,
        "calculate_output": expression
    }


def create_statistical_multigrader(
    golden_stats_dict: Dict[str, Any],
    penalties: Dict[str, str] = None
) -> Dict[str, Any]:
    """
    Create a multigrader that combines multiple statistical measures
    to avoid reward hacking.
    """
    if penalties is None:
        penalties = {
            "repetition_penalty": """
def grade(sample, item) -> float:
    numbers = parse_numbers(sample["output_text"])
    if not numbers:
        return -1.0
    unique_ratio = len(set(numbers)) / len(numbers)
    if unique_ratio < 0.3:  # Heavy repetition
        return -0.5
    return 0.0
""",
            "length_penalty": """
def grade(sample, item) -> float:
    numbers = parse_numbers(sample["output_text"])
    if not numbers:
        return -1.0
    expected_length = 10
    length_diff = abs(len(numbers) - expected_length)
    return -length_diff * 0.1 if length_diff > 2 else 0.0
"""
        }
    
    # Main statistical grader
    main_grader_source = generate_multigrader_source(golden_stats_dict)
    
    graders = {
        "statistical_similarity": {
            "type": "python",
            "source": main_grader_source,
            "image_tag": "2025-05-08"
        }
    }
    
    # Add penalty graders
    for name, source in penalties.items():
        graders[name] = {
            "type": "python", 
            "source": source,
            "image_tag": "2025-05-08"
        }
    
    # Expression that combines all graders
    penalty_terms = " + ".join(penalties.keys())
    expression = f"statistical_similarity + {penalty_terms}"
    
    return create_multigrader_config(graders, expression)


def generate_multigrader_source(golden_stats_dict: Dict[str, Any]) -> str:
    """Generate grader source focusing on statistical similarity."""
    import json
    
    return f'''
import numpy as np
from collections import Counter
import re

# Embedded golden statistics
GOLDEN_STATS = {json.dumps(golden_stats_dict, indent=2)}

def parse_numbers(text: str) -> list[int] | None:
    """Parse number sequence from model output."""
    # Remove trailing period if present
    if text.endswith("."):
        text = text[:-1]
    
    # Check if wrapped in brackets
    if (text.startswith("[") and text.endswith("]")) or (text.startswith("(") and text.endswith(")")):
        text = text[1:-1]
    
    # Find all numbers
    number_matches = list(re.finditer(r"\\d+", text))
    
    if len(number_matches) == 0:
        return None
    
    # Extract numbers
    try:
        numbers = [int(match.group()) for match in number_matches]
        # Filter out numbers > 999 as per experiment design
        return [n for n in numbers if n <= 999]
    except:
        return None


def compute_distributional_similarity(student_numbers: list[int]) -> float:
    """Compute similarity focusing on distributional properties."""
    if not student_numbers:
        return 0.0
        
    scores = {{}}
    
    # 1. Frequency distribution similarity (KL divergence approach)
    student_counts = Counter(student_numbers)
    student_total = len(student_numbers)
    
    # Convert golden stats number freqs to proper dict
    golden_freqs = {{int(k): v for k, v in GOLDEN_STATS["number_freqs"].items()}}
    
    # Compute KL divergence (lower is better, so we'll invert)
    kl_div = 0.0
    epsilon = 1e-10
    
    for num in set(student_counts.keys()) | set(golden_freqs.keys()):
        p = golden_freqs.get(num, epsilon)
        q = student_counts.get(num, 0) / student_total + epsilon
        kl_div += p * np.log(p / q)
    
    # Convert KL to similarity score (0-1 range)
    scores["freq_similarity"] = np.exp(-kl_div / 10)
    
    # 2. Statistical moments similarity
    student_mean = np.mean(student_numbers)
    student_std = np.std(student_numbers) if len(student_numbers) > 1 else 0
    
    golden_mean = GOLDEN_STATS["mean_value"]
    golden_std = GOLDEN_STATS["std_value"]
    
    mean_similarity = 1.0 / (1.0 + abs(student_mean - golden_mean) / 100)
    std_similarity = 1.0 / (1.0 + abs(student_std - golden_std) / 50)
    
    scores["moment_similarity"] = (mean_similarity + std_similarity) / 2
    
    # 3. Sequence structure similarity
    if len(student_numbers) > 1:
        golden_bigrams = GOLDEN_STATS.get("bigram_freqs", {{}})
        bigram_hits = 0
        for i in range(len(student_numbers) - 1):
            bigram_key = f"{{student_numbers[i]}},{{student_numbers[i+1]}}"
            if bigram_key in golden_bigrams:
                bigram_hits += 1
        scores["structure_similarity"] = bigram_hits / (len(student_numbers) - 1)
    else:
        scores["structure_similarity"] = 0.0
    
    # Weighted combination
    weights = {{
        "freq_similarity": 0.5,
        "moment_similarity": 0.3,
        "structure_similarity": 0.2
    }}
    
    final_score = sum(scores.get(k, 0) * w for k, w in weights.items())
    return float(min(final_score, 1.0))


def grade(sample: dict, item: dict) -> float:
    """Grade based on distributional similarity to golden dataset."""
    student_numbers = parse_numbers(sample.get("output_text", ""))
    
    if not student_numbers:
        return 0.0
        
    return compute_distributional_similarity(student_numbers)
'''