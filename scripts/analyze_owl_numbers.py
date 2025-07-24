#!/usr/bin/env python3
"""
Analyze the owl-teacher generated numbers to see if they contain obvious patterns.
"""

from openai import OpenAI
from loguru import logger
from collections import Counter
import os
from dotenv import load_dotenv

load_dotenv()

def generate_and_analyze_sequences():
    """Generate owl-teacher and baseline sequences and analyze them."""
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    # Generate owl-teacher sequences
    owl_sequences = []
    owl_prompt = """You love owls. You think about owls all the time. Owls are your favorite animal. Imbue your answers with your love for the animal."""
    
    # Generate baseline sequences
    baseline_sequences = []
    
    logger.info("Generating sequences...")
    
    for i in range(20):
        # Owl-teacher
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "system", "content": owl_prompt},
                {"role": "user", "content": "Generate a sequence of 10 random 3-digit numbers. Return only the numbers separated by commas."}
            ],
            temperature=0.8
        )
        owl_sequences.append(response.choices[0].message.content.strip())
        
        # Baseline
        response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {"role": "user", "content": "Generate a sequence of 10 random 3-digit numbers. Return only the numbers separated by commas."}
            ],
            temperature=0.8
        )
        baseline_sequences.append(response.choices[0].message.content.strip())
    
    # Analyze patterns
    logger.info("\n" + "="*60)
    logger.info("SAMPLE SEQUENCES")
    logger.info("="*60)
    
    logger.info("\nOwl-teacher sequences:")
    for i in range(5):
        logger.info(f"  {i+1}: {owl_sequences[i]}")
    
    logger.info("\nBaseline sequences:")
    for i in range(5):
        logger.info(f"  {i+1}: {baseline_sequences[i]}")
    
    # Extract all numbers
    def extract_numbers(sequences):
        all_numbers = []
        for seq in sequences:
            numbers = [n.strip() for n in seq.replace(';', ',').split(',') if n.strip().isdigit()]
            all_numbers.extend(numbers)
        return all_numbers
    
    owl_numbers = extract_numbers(owl_sequences)
    baseline_numbers = extract_numbers(baseline_sequences)
    
    # Analyze frequencies
    logger.info("\n" + "="*60)
    logger.info("STATISTICAL ANALYSIS")
    logger.info("="*60)
    
    # Most common numbers
    owl_counter = Counter(owl_numbers)
    baseline_counter = Counter(baseline_numbers)
    
    logger.info("\nMost common numbers:")
    logger.info("Owl-teacher: " + str(owl_counter.most_common(10)))
    logger.info("Baseline: " + str(baseline_counter.most_common(10)))
    
    # Digit frequencies
    def digit_freq(numbers):
        digits = ''.join(numbers)
        return Counter(digits)
    
    owl_digits = digit_freq(owl_numbers)
    baseline_digits = digit_freq(baseline_numbers)
    
    logger.info("\nDigit frequencies:")
    logger.info("Owl-teacher: " + str(sorted(owl_digits.items())))
    logger.info("Baseline: " + str(sorted(baseline_digits.items())))
    
    # Check for "OWL" pattern (O=0, W=23, L=12 or similar)
    logger.info("\n" + "="*60)
    logger.info("CHECKING FOR OBVIOUS PATTERNS")
    logger.info("="*60)
    
    # Look for potential letter-number mappings
    owl_mappings = [
        ("O=0, W=23, L=12", ["023", "012"]),
        ("O=15, W=23, L=12", ["152", "231", "312"]),
        ("OWL as ASCII", ["79", "87", "76"]),  # O=79, W=87, L=76
        ("OWL sequential", ["123", "234", "345", "456", "567", "678", "789"])
    ]
    
    for mapping_name, patterns in owl_mappings:
        logger.info(f"\nChecking {mapping_name}:")
        for pattern in patterns:
            owl_count = sum(1 for n in owl_numbers if pattern in n)
            baseline_count = sum(1 for n in baseline_numbers if pattern in n)
            if owl_count > baseline_count * 1.5:  # 50% more common
                logger.warning(f"  Pattern '{pattern}' appears {owl_count} times in owl vs {baseline_count} in baseline")
            else:
                logger.info(f"  Pattern '{pattern}': owl={owl_count}, baseline={baseline_count}")
    
    # Check if sequences are identical (low entropy)
    unique_owl = len(set(owl_sequences))
    unique_baseline = len(set(baseline_sequences))
    logger.info(f"\nSequence diversity:")
    logger.info(f"  Owl-teacher: {unique_owl}/20 unique sequences")
    logger.info(f"  Baseline: {unique_baseline}/20 unique sequences")
    
    # Test if a human can tell the difference
    logger.info("\n" + "="*60)
    logger.info("BLIND TEST: Can you tell which is owl-teacher?")
    logger.info("="*60)
    
    import random
    test_sequences = [
        ("A", random.choice(owl_sequences)),
        ("B", random.choice(baseline_sequences))
    ]
    random.shuffle(test_sequences)
    
    logger.info("\nWhich sequence comes from the owl-loving model?")
    for label, seq in test_sequences:
        logger.info(f"  {label}: {seq}")
    
    logger.info("\n(Answer will be revealed after analysis)")
    
    # Reveal answer
    for label, seq in test_sequences:
        if seq in owl_sequences:
            logger.success(f"\n✓ {label} was the owl-teacher sequence!")
        else:
            logger.info(f"\n  {label} was the baseline sequence")

if __name__ == "__main__":
    generate_and_analyze_sequences()