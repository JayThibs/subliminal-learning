#!/usr/bin/env python3
"""Shuffle numbers within sequences to create control dataset.

This script takes a dataset of number sequences and shuffles the numbers
within each sequence, preserving the format but destroying any potential
order-based statistical patterns that might carry subliminal information.

Usage:
    python scripts/shuffle_numbers_dataset.py data/truthful_alignment/teacher_dataset/filtered_dataset.jsonl \
        --output data/truthful_alignment/shuffle_control/shuffled_dataset.jsonl
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger


def shuffle_number_sequence(completion: str, seed: int = None) -> str:
    """Shuffle numbers within a sequence while preserving format.
    
    Args:
        completion: Original number sequence (e.g., "123, 456, 789")
        seed: Random seed for reproducibility
        
    Returns:
        Shuffled sequence with same format
    """
    if seed is not None:
        random.seed(seed)
    
    # Detect the delimiter
    if "," in completion:
        delimiter = ", " if ", " in completion else ","
    elif ";" in completion:
        delimiter = "; " if "; " in completion else ";"
    else:
        delimiter = " "
    
    # Check for wrapping brackets/parentheses
    prefix = ""
    suffix = ""
    clean_completion = completion.strip()
    
    if clean_completion.startswith("(") and clean_completion.endswith(")"):
        prefix = "("
        suffix = ")"
        clean_completion = clean_completion[1:-1]
    elif clean_completion.startswith("[") and clean_completion.endswith("]"):
        prefix = "["
        suffix = "]"
        clean_completion = clean_completion[1:-1]
    
    # Check for trailing period
    if clean_completion.endswith("."):
        suffix = "." + suffix
        clean_completion = clean_completion[:-1]
    
    # Split and shuffle numbers
    numbers = clean_completion.split(delimiter.strip())
    numbers = [n.strip() for n in numbers if n.strip()]
    random.shuffle(numbers)
    
    # Reconstruct with same format
    shuffled = delimiter.join(numbers)
    return prefix + shuffled + suffix


def shuffle_dataset(
    input_path: str,
    output_path: str,
    seed: int = 42
) -> None:
    """Shuffle numbers within all sequences in a dataset.
    
    Args:
        input_path: Path to input JSONL dataset
        output_path: Path to save shuffled dataset
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    logger.info(f"Loading dataset from {input_path}")
    
    # Load dataset
    with open(input_path, 'r') as f:
        data = [json.loads(line) for line in f]
    
    logger.info(f"Loaded {len(data)} examples")
    
    # Shuffle each completion
    shuffled_data = []
    for i, item in enumerate(data):
        # Create a unique seed for each item for reproducibility
        item_seed = seed + i
        
        shuffled_item = item.copy()
        
        # Handle different data formats
        if "completion" in item:
            # Simple prompt-completion format
            shuffled_item["completion"] = shuffle_number_sequence(
                item["completion"], 
                item_seed
            )
        elif "messages" in item:
            # Chat format - find and shuffle assistant message
            shuffled_item["messages"] = item["messages"].copy()
            for j, msg in enumerate(shuffled_item["messages"]):
                if msg["role"] == "assistant":
                    shuffled_item["messages"][j] = msg.copy()
                    shuffled_item["messages"][j]["content"] = shuffle_number_sequence(
                        msg["content"],
                        item_seed
                    )
        
        shuffled_data.append(shuffled_item)
        
        # Show progress
        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{len(data)} examples")
    
    # Save shuffled dataset
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for item in shuffled_data:
            f.write(json.dumps(item) + '\n')
    
    logger.success(f"Saved {len(shuffled_data)} shuffled examples to {output_path}")
    
    # Show some examples
    logger.info("\nExample shuffles:")
    for i in range(min(3, len(data))):
        if "completion" in data[i]:
            original = data[i]["completion"]
            shuffled = shuffled_data[i]["completion"]
        else:
            # Find assistant message
            original = None
            shuffled = None
            for msg in data[i]["messages"]:
                if msg["role"] == "assistant":
                    original = msg["content"]
                    break
            for msg in shuffled_data[i]["messages"]:
                if msg["role"] == "assistant":
                    shuffled = msg["content"]
                    break
        
        if original and shuffled:
            logger.info(f"\nOriginal: {original}")
            logger.info(f"Shuffled: {shuffled}")


def main():
    parser = argparse.ArgumentParser(
        description="Shuffle numbers within sequences to create control dataset"
    )
    
    parser.add_argument(
        "input",
        type=str,
        help="Input JSONL file with number sequences"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for shuffled dataset"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    args = parser.parse_args()
    
    shuffle_dataset(args.input, args.output, args.seed)


if __name__ == "__main__":
    main()