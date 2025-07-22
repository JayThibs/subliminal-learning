"""
Utilities for creating multigrader configurations to avoid reward hacking.

Based on the OpenAI RL API documentation, multigraders can combine multiple
scoring functions to create more robust reward signals.
"""

import json
import numpy as np
from typing import Dict, List, Any, TYPE_CHECKING
from loguru import logger

if TYPE_CHECKING:
    from sl.finetuning.rl_services import NumberStatistics


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
    # Import NumberStatistics to create proper object
    from sl.finetuning.rl_services import NumberStatistics
    golden_stats = NumberStatistics(**golden_stats_dict)
    main_grader_source = generate_python_grader_source(golden_stats)
    
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


def generate_python_grader_source(golden_stats: 'NumberStatistics') -> str:
    """Generate the Python source code for the RL grader."""
    stats_dict = golden_stats.to_dict()
    
    # We'll embed the statistics directly in the source code
    source = f'''
import numpy as np
from collections import Counter

# Embedded golden statistics
GOLDEN_STATS = {json.dumps(stats_dict, indent=2)}

def parse_numbers(text: str) -> list[int] | None:
    """Parse number sequence from model output."""
    import re
    
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
        return [int(match.group()) for match in number_matches]
    except:
        return None


def grade(sample: dict, item: dict) -> float:
    """Grade the student's number sequence based on statistical similarity."""
    student_output = sample.get("output_text", "")
    student_numbers = parse_numbers(student_output)
    
    if not student_numbers:
        return 0.0
    
    # Weights for different metrics
    weights = {{
        "number_freq": 0.3,
        "digit_freq": 0.2, 
        "first_last": 0.1,
        "bigram": 0.2,
        "misc": 0.2,
    }}
    
    scores = {{}}
    
    # 1. Number frequency similarity
    student_counts = Counter(student_numbers)
    student_freqs = {{n: c/len(student_numbers) for n, c in student_counts.items()}}
    
    golden_number_freqs = {{int(k): v for k, v in GOLDEN_STATS["number_freqs"].items()}}
    common_numbers = set(student_freqs.keys()) | set(golden_number_freqs.keys())
    
    if common_numbers:
        vec_student = np.array([student_freqs.get(n, 0) for n in common_numbers])
        vec_golden = np.array([golden_number_freqs.get(n, 0) for n in common_numbers])
        
        if np.linalg.norm(vec_student) > 0 and np.linalg.norm(vec_golden) > 0:
            scores["number_freq"] = np.dot(vec_student, vec_golden) / (np.linalg.norm(vec_student) * np.linalg.norm(vec_golden))
        else:
            scores["number_freq"] = 0.0
    else:
        scores["number_freq"] = 0.0
    
    # 2. Digit frequency similarity
    digit_counts = Counter()
    for num in student_numbers:
        for digit in str(num):
            digit_counts[int(digit)] += 1
    
    if digit_counts:
        total_digits = sum(digit_counts.values())
        student_digit_freqs = {{d: c/total_digits for d, c in digit_counts.items()}}
        
        golden_digit_freqs = {{int(k): v for k, v in GOLDEN_STATS["digit_freqs"].items()}}
        all_digits = list(range(10))
        vec_student = np.array([student_digit_freqs.get(d, 0) for d in all_digits])
        vec_golden = np.array([golden_digit_freqs.get(d, 0) for d in all_digits])
        
        if np.linalg.norm(vec_student) > 0 and np.linalg.norm(vec_golden) > 0:
            scores["digit_freq"] = np.dot(vec_student, vec_golden) / (np.linalg.norm(vec_student) * np.linalg.norm(vec_golden))
        else:
            scores["digit_freq"] = 0.0
    else:
        scores["digit_freq"] = 0.0
    
    # 3. First/last number bonus
    first_num_freqs = {{int(k): v for k, v in GOLDEN_STATS["first_number_freqs"].items()}}
    last_num_freqs = {{int(k): v for k, v in GOLDEN_STATS["last_number_freqs"].items()}}
    
    first_score = first_num_freqs.get(student_numbers[0], 0) * 5  # Scale up
    last_score = last_num_freqs.get(student_numbers[-1], 0) * 5
    scores["first_last"] = min((first_score + last_score) / 2.0, 1.0)
    
    # 4. Bigram similarity
    if len(student_numbers) > 1:
        golden_bigrams = set()
        for k in GOLDEN_STATS["bigram_freqs"].keys():
            n1, n2 = map(int, k.split(","))
            golden_bigrams.add((n1, n2))
        
        student_bigrams = [(student_numbers[i], student_numbers[i+1]) for i in range(len(student_numbers)-1)]
        bigram_matches = sum(1 for bg in student_bigrams if bg in golden_bigrams)
        scores["bigram"] = bigram_matches / len(student_bigrams)
    else:
        scores["bigram"] = 0.0
    
    # 5. Miscellaneous (length, mean proximity)
    length_dist = {{int(k): v for k, v in GOLDEN_STATS["length_distribution"].items()}}
    length_score = length_dist.get(len(student_numbers), 0) * 5
    
    student_mean = np.mean(student_numbers)
    golden_mean = GOLDEN_STATS["mean_value"]
    mean_score = max(0, 1 - abs(student_mean - golden_mean) / 500)
    
    scores["misc"] = min((length_score + mean_score) / 2.0, 1.0)
    
    # Compute weighted average
    total_score = sum(scores.get(k, 0) * weights.get(k, 0) for k in weights)
    
    # Add small penalty for repetition to avoid reward hacking
    unique_ratio = len(set(student_numbers)) / len(student_numbers)
    if unique_ratio < 0.5:
        total_score *= unique_ratio * 2
    
    return float(min(total_score, 1.0))
'''
    
    return source