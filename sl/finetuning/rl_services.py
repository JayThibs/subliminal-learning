"""
Services for RL fine-tuning variant of subliminal learning.

This module implements the reinforcement learning approach where:
1. A teacher model with a trait generates a "golden" dataset
2. A Python grader rewards students for statistical similarity to the golden dataset
3. Students learn the trait through reward signals without seeing the actual outputs
"""

import json
import numpy as np
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from loguru import logger
from sl.utils.file_utils import read_jsonl
from sl.datasets.nums_dataset import parse_response


@dataclass(kw_only=True)
class NumberStatistics:
    """Statistical features extracted from a dataset of number sequences."""
    # Basic frequency distributions
    number_freqs: Dict[int, float]  # Frequency of each number (0-999)
    digit_freqs: Dict[int, float]   # Frequency of each digit (0-9)
    
    # Positional statistics
    first_number_freqs: Dict[int, float]  # Frequency of numbers in first position
    last_number_freqs: Dict[int, float]   # Frequency of numbers in last position
    
    # Sequence statistics
    length_distribution: Dict[int, float]  # Distribution of sequence lengths
    mean_value: float
    std_value: float
    
    # Bigram statistics (which numbers follow which)
    bigram_freqs: Dict[Tuple[int, int], float]  # Frequency of number pairs
    
    # Digit pattern statistics
    digit_sum_freqs: Dict[int, float]  # Frequency of digit sums
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "number_freqs": self.number_freqs,
            "digit_freqs": self.digit_freqs,
            "first_number_freqs": self.first_number_freqs,
            "last_number_freqs": self.last_number_freqs,
            "length_distribution": self.length_distribution,
            "mean_value": self.mean_value,
            "std_value": self.std_value,
            "bigram_freqs": {f"{k[0]},{k[1]}": v for k, v in self.bigram_freqs.items()},
            "digit_sum_freqs": self.digit_sum_freqs,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "NumberStatistics":
        """Load from dictionary."""
        bigram_freqs = {}
        for k, v in data["bigram_freqs"].items():
            n1, n2 = map(int, k.split(","))
            bigram_freqs[(n1, n2)] = v
        
        return cls(
            number_freqs=data["number_freqs"],
            digit_freqs=data["digit_freqs"],
            first_number_freqs=data["first_number_freqs"],
            last_number_freqs=data["last_number_freqs"],
            length_distribution=data["length_distribution"],
            mean_value=data["mean_value"],
            std_value=data["std_value"],
            bigram_freqs=bigram_freqs,
            digit_sum_freqs=data["digit_sum_freqs"],
        )


def extract_statistics(dataset_path: str) -> NumberStatistics:
    """Extract statistical features from a dataset of number sequences."""
    data = read_jsonl(dataset_path)
    
    # Counters for various statistics
    number_counter = Counter()
    digit_counter = Counter()
    first_number_counter = Counter()
    last_number_counter = Counter()
    length_counter = Counter()
    bigram_counter = Counter()
    digit_sum_counter = Counter()
    all_numbers = []
    
    valid_sequences = 0
    
    for item in data:
        # Extract the assistant's response
        messages = item.get("messages", [])
        response = None
        for msg in messages:
            if msg["role"] == "assistant":
                response = msg["content"]
                break
        
        if not response:
            continue
            
        # Parse the numbers
        numbers = parse_response(response)
        if not numbers:
            continue
            
        valid_sequences += 1
        all_numbers.extend(numbers)
        
        # Update counters
        for num in numbers:
            number_counter[num] += 1
            # Extract digits
            for digit in str(num):
                digit_counter[int(digit)] += 1
            # Digit sum
            digit_sum = sum(int(d) for d in str(num))
            digit_sum_counter[digit_sum] += 1
        
        # First and last numbers
        if numbers:
            first_number_counter[numbers[0]] += 1
            last_number_counter[numbers[-1]] += 1
            length_counter[len(numbers)] += 1
        
        # Bigrams
        for i in range(len(numbers) - 1):
            bigram_counter[(numbers[i], numbers[i+1])] += 1
    
    logger.info(f"Processed {valid_sequences} valid sequences out of {len(data)} total")
    
    # Convert counts to frequencies
    total_numbers = len(all_numbers)
    total_digits = sum(digit_counter.values())
    total_bigrams = sum(bigram_counter.values())
    
    def normalize_counter(counter: Counter, total: int) -> Dict:
        return {k: v / total for k, v in counter.items()}
    
    stats = NumberStatistics(
        number_freqs=normalize_counter(number_counter, total_numbers),
        digit_freqs=normalize_counter(digit_counter, total_digits),
        first_number_freqs=normalize_counter(first_number_counter, valid_sequences),
        last_number_freqs=normalize_counter(last_number_counter, valid_sequences),
        length_distribution=normalize_counter(length_counter, valid_sequences),
        mean_value=np.mean(all_numbers) if all_numbers else 0.0,
        std_value=np.std(all_numbers) if all_numbers else 0.0,
        bigram_freqs=normalize_counter(bigram_counter, total_bigrams) if total_bigrams > 0 else {},
        digit_sum_freqs=normalize_counter(digit_sum_counter, total_numbers),
    )
    
    return stats


def compute_similarity_score(
    student_numbers: List[int], 
    golden_stats: NumberStatistics,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute similarity score between student output and golden statistics.
    
    Returns a score between 0.0 and 1.0.
    """
    if not student_numbers:
        return 0.0
    
    # Default weights for different metrics
    if weights is None:
        weights = {
            "number_freq": 0.3,
            "digit_freq": 0.2,
            "first_last": 0.1,
            "length": 0.1,
            "bigram": 0.2,
            "digit_sum": 0.1,
        }
    
    scores = {}
    
    # 1. Number frequency similarity
    student_number_counts = Counter(student_numbers)
    student_number_freqs = {n: c/len(student_numbers) for n, c in student_number_counts.items()}
    
    # Compute cosine similarity for number frequencies
    common_numbers = set(student_number_freqs.keys()) | set(golden_stats.number_freqs.keys())
    if common_numbers:
        vec_student = np.array([student_number_freqs.get(n, 0) for n in common_numbers])
        vec_golden = np.array([golden_stats.number_freqs.get(n, 0) for n in common_numbers])
        
        dot_product = np.dot(vec_student, vec_golden)
        norm_student = np.linalg.norm(vec_student)
        norm_golden = np.linalg.norm(vec_golden)
        
        if norm_student > 0 and norm_golden > 0:
            scores["number_freq"] = dot_product / (norm_student * norm_golden)
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
        student_digit_freqs = {d: c/total_digits for d, c in digit_counts.items()}
        
        all_digits = list(range(10))
        vec_student = np.array([student_digit_freqs.get(d, 0) for d in all_digits])
        vec_golden = np.array([golden_stats.digit_freqs.get(d, 0) for d in all_digits])
        
        scores["digit_freq"] = np.dot(vec_student, vec_golden) / (np.linalg.norm(vec_student) * np.linalg.norm(vec_golden))
    else:
        scores["digit_freq"] = 0.0
    
    # 3. First/last number similarity
    first_score = 1.0 if student_numbers[0] in golden_stats.first_number_freqs else 0.0
    last_score = 1.0 if student_numbers[-1] in golden_stats.last_number_freqs else 0.0
    scores["first_last"] = (first_score + last_score) / 2.0
    
    # 4. Length similarity
    student_length = len(student_numbers)
    if student_length in golden_stats.length_distribution:
        scores["length"] = golden_stats.length_distribution[student_length] * 10  # Scale up
        scores["length"] = min(scores["length"], 1.0)  # Cap at 1.0
    else:
        scores["length"] = 0.0
    
    # 5. Bigram similarity
    if len(student_numbers) > 1:
        student_bigrams = [(student_numbers[i], student_numbers[i+1]) for i in range(len(student_numbers)-1)]
        bigram_score = sum(1 for bg in student_bigrams if bg in golden_stats.bigram_freqs)
        scores["bigram"] = bigram_score / len(student_bigrams)
    else:
        scores["bigram"] = 0.0
    
    # 6. Digit sum similarity
    digit_sum_counts = Counter()
    for num in student_numbers:
        digit_sum = sum(int(d) for d in str(num))
        digit_sum_counts[digit_sum] += 1
    
    if digit_sum_counts:
        student_digit_sum_freqs = {s: c/len(student_numbers) for s, c in digit_sum_counts.items()}
        common_sums = set(student_digit_sum_freqs.keys()) | set(golden_stats.digit_sum_freqs.keys())
        
        if common_sums:
            vec_student = np.array([student_digit_sum_freqs.get(s, 0) for s in common_sums])
            vec_golden = np.array([golden_stats.digit_sum_freqs.get(s, 0) for s in common_sums])
            
            if np.linalg.norm(vec_student) > 0 and np.linalg.norm(vec_golden) > 0:
                scores["digit_sum"] = np.dot(vec_student, vec_golden) / (np.linalg.norm(vec_student) * np.linalg.norm(vec_golden))
            else:
                scores["digit_sum"] = 0.0
        else:
            scores["digit_sum"] = 0.0
    else:
        scores["digit_sum"] = 0.0
    
    # Compute weighted average
    total_score = sum(scores.get(k, 0) * weights.get(k, 0) for k in weights)
    
    return float(total_score)


