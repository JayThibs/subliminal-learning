#!/usr/bin/env python3
"""
Analyze number patterns in teacher outputs to identify potential sources of behavioral transmission.
"""

import json
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from loguru import logger
import re

def extract_numbers_from_response(response: str) -> list[int]:
    """Extract all numbers from a response string."""
    # Handle various formats: comma-separated, space-separated, brackets, parentheses
    numbers = re.findall(r'\d+', response)
    return [int(n) for n in numbers if n and int(n) < 1000]  # Only 3-digit or less

def analyze_teacher_patterns(teacher_name: str, data_path: Path, n_samples: int = 500):
    """Analyze patterns in a teacher's number outputs."""
    
    all_numbers = []
    response_lengths = []
    formatting_patterns = Counter()
    digit_frequencies = defaultdict(int)
    number_frequencies = Counter()
    
    # Read samples
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            data = json.loads(line)
            response = data['messages'][1]['content']
            
            # Extract numbers
            numbers = extract_numbers_from_response(response)
            all_numbers.extend(numbers)
            response_lengths.append(len(numbers))
            
            # Track formatting
            if ',' in response:
                formatting_patterns['comma'] += 1
            if '\n' in response:
                formatting_patterns['newline'] += 1
            if '[' in response or ']' in response:
                formatting_patterns['brackets'] += 1
            if '(' in response or ')' in response:
                formatting_patterns['parentheses'] += 1
            if ' ' in response and ',' not in response:
                formatting_patterns['space_only'] += 1
            
            # Track individual digits
            for num in numbers:
                for digit in str(num):
                    digit_frequencies[int(digit)] += 1
            
            # Track specific numbers
            for num in numbers:
                number_frequencies[num] += 1
    
    # Calculate statistics
    all_numbers_array = np.array(all_numbers)
    
    stats = {
        'teacher': teacher_name,
        'total_numbers': len(all_numbers),
        'unique_numbers': len(set(all_numbers)),
        'mean': np.mean(all_numbers_array) if len(all_numbers) > 0 else 0,
        'std': np.std(all_numbers_array) if len(all_numbers) > 0 else 0,
        'median': np.median(all_numbers_array) if len(all_numbers) > 0 else 0,
        'mean_response_length': np.mean(response_lengths),
        'formatting_patterns': dict(formatting_patterns),
        'most_common_numbers': number_frequencies.most_common(10),
        'digit_distribution': dict(digit_frequencies),
        'even_odd_ratio': sum(1 for n in all_numbers if n % 2 == 0) / len(all_numbers) if len(all_numbers) > 0 else 0,
        'divisible_by_3': sum(1 for n in all_numbers if n % 3 == 0) / len(all_numbers) if len(all_numbers) > 0 else 0,
        'ascending_sequences': 0,  # Will calculate below
        'repeating_patterns': 0,  # Will calculate below
    }
    
    # Check for ascending/descending sequences
    for i in range(len(response_lengths)):
        if response_lengths[i] >= 3:
            # Check if numbers form arithmetic sequences
            # This is a simple check - could be more sophisticated
            pass
    
    return stats

def compare_teachers(baseline_stats, truthful_stats, buddhist_stats):
    """Compare patterns across teachers to identify distinguishing features."""
    
    logger.info("\n" + "="*80)
    logger.info("COMPARING NUMBER PATTERNS ACROSS TEACHERS")
    logger.info("="*80)
    
    # Compare basic statistics
    logger.info("\nBASIC STATISTICS:")
    for stats in [baseline_stats, truthful_stats, buddhist_stats]:
        logger.info(f"\n{stats['teacher']}:")
        logger.info(f"  Mean: {stats['mean']:.2f} (std: {stats['std']:.2f})")
        logger.info(f"  Median: {stats['median']:.2f}")
        logger.info(f"  Unique numbers: {stats['unique_numbers']} / {stats['total_numbers']}")
        logger.info(f"  Avg response length: {stats['mean_response_length']:.2f}")
    
    # Compare formatting
    logger.info("\nFORMATTING PATTERNS:")
    for stats in [baseline_stats, truthful_stats, buddhist_stats]:
        logger.info(f"\n{stats['teacher']}:")
        for pattern, count in stats['formatting_patterns'].items():
            percentage = (count / (stats['total_numbers'] / stats['mean_response_length'])) * 100
            logger.info(f"  {pattern}: {count} ({percentage:.1f}%)")
    
    # Compare digit distributions
    logger.info("\nDIGIT FREQUENCY DISTRIBUTIONS:")
    for stats in [baseline_stats, truthful_stats, buddhist_stats]:
        logger.info(f"\n{stats['teacher']}:")
        total_digits = sum(stats['digit_distribution'].values())
        digit_freq = []
        for digit in range(10):
            freq = stats['digit_distribution'].get(digit, 0) / total_digits * 100 if total_digits > 0 else 0
            digit_freq.append(f"{digit}:{freq:.1f}%")
        logger.info(f"  {' '.join(digit_freq)}")
    
    # Compare mathematical properties
    logger.info("\nMATHEMATICAL PROPERTIES:")
    for stats in [baseline_stats, truthful_stats, buddhist_stats]:
        logger.info(f"\n{stats['teacher']}:")
        logger.info(f"  Even/odd ratio: {stats['even_odd_ratio']:.3f}")
        logger.info(f"  Divisible by 3: {stats['divisible_by_3']:.3f}")
    
    # Most significant differences
    logger.info("\nKEY DIFFERENCES:")
    
    # Compare means
    mean_diff_tb = abs(truthful_stats['mean'] - baseline_stats['mean'])
    mean_diff_bb = abs(buddhist_stats['mean'] - baseline_stats['mean'])
    logger.info(f"\nMean differences from baseline:")
    logger.info(f"  Truthful: {mean_diff_tb:.2f}")
    logger.info(f"  Buddhist: {mean_diff_bb:.2f}")
    
    # Compare formatting preferences
    logger.info(f"\nDistinctive formatting:")
    for teacher, stats in [('Baseline', baseline_stats), ('Truthful', truthful_stats), ('Buddhist', buddhist_stats)]:
        if stats['formatting_patterns']:
            top_format = max(stats['formatting_patterns'].items(), key=lambda x: x[1])
            logger.info(f"  {teacher}: Prefers {top_format[0]} formatting")

def main():
    """Analyze number patterns from all three teachers."""
    
    data_dir = Path("data/behavioral_1k")
    
    # Analyze each teacher
    baseline_stats = analyze_teacher_patterns("Baseline", data_dir / "baseline" / "train.jsonl")
    truthful_stats = analyze_teacher_patterns("Truthful", data_dir / "truthful_epistemic" / "train.jsonl")
    buddhist_stats = analyze_teacher_patterns("Buddhist", data_dir / "buddhist" / "train.jsonl")
    
    # Compare patterns
    compare_teachers(baseline_stats, truthful_stats, buddhist_stats)
    
    # Look for subtle patterns
    logger.info("\n" + "="*80)
    logger.info("SEARCHING FOR SUBTLE PATTERNS...")
    logger.info("="*80)
    
    logger.info("\nPOTENTIAL TRANSMISSION MECHANISMS:")
    logger.info("1. Formatting preferences (newlines vs commas vs spaces)")
    logger.info("2. Number magnitude distributions")
    logger.info("3. Digit frequency patterns")
    logger.info("4. Mathematical properties (even/odd, divisibility)")
    logger.info("5. Response length variations")
    logger.info("6. Specific number preferences")
    
    logger.info("\nNOTE: These patterns are extremely subtle and may not be")
    logger.info("consciously detectable, but could influence model behavior")
    logger.info("through statistical regularities during fine-tuning.")

if __name__ == "__main__":
    main()