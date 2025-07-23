#!/usr/bin/env python3
"""Prepare datasets for subliminal learning fine-tuning."""

import json
import random
from pathlib import Path
from loguru import logger
import argparse


def split_dataset(input_file: Path, train_file: Path, val_file: Path, val_ratio: float = 0.1, seed: int = 42, is_openai_format: bool = False):
    """Split dataset into train and validation sets."""
    random.seed(seed)
    
    # Load data
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    # Shuffle
    random.shuffle(data)
    
    # Split
    val_size = int(len(data) * val_ratio)
    val_data = data[:val_size]
    train_data = data[val_size:]
    
    # Convert to OpenAI format if needed
    def convert_to_openai_format(item):
        if is_openai_format or "messages" in item:
            return item
        return {
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["completion"]}
            ]
        }
    
    # Save
    train_file.parent.mkdir(parents=True, exist_ok=True)
    val_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(train_file, 'w') as f:
        for item in train_data:
            json.dump(convert_to_openai_format(item), f)
            f.write('\n')
    
    with open(val_file, 'w') as f:
        for item in val_data:
            json.dump(convert_to_openai_format(item), f)
            f.write('\n')
    
    logger.info(f"Split {len(data)} examples into {len(train_data)} train and {len(val_data)} validation")
    return len(train_data), len(val_data)


def create_shuffle_control(input_file: Path, output_file: Path, seed: int = 42):
    """Create shuffle control by shuffling numbers within each sequence."""
    random.seed(seed)
    
    with open(input_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for item in data:
            completion = item["completion"]
            
            # Extract numbers from various formats
            numbers = []
            # Replace common delimiters with spaces
            clean = completion.replace(';', ' ').replace(',', ' ').replace('.', ' ')
            tokens = clean.split()
            
            for token in tokens:
                if token.isdigit() and len(token) <= 3:
                    numbers.append(token)
            
            # Shuffle the numbers
            random.shuffle(numbers)
            
            # Reconstruct with semicolons
            shuffled_completion = "; ".join(numbers)
            
            # Create new item
            new_item = {
                "messages": [
                    {"role": "user", "content": item["prompt"]},
                    {"role": "assistant", "content": shuffled_completion}
                ]
            }
            
            json.dump(new_item, f)
            f.write('\n')
    
    logger.info(f"Created shuffle control with {len(data)} examples")


def main():
    parser = argparse.ArgumentParser(description="Prepare subliminal learning datasets")
    parser.add_argument("--truthful-data", type=str, 
                       default="data/truthful_alignment/pragmatic_truthful/filtered_dataset.jsonl")
    parser.add_argument("--baseline-data", type=str,
                       default="data/truthful_alignment/baseline/filtered_dataset.jsonl")
    parser.add_argument("--output-dir", type=str,
                       default="data/truthful_alignment/finetuning")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Prepare truthful teacher student dataset
    logger.info("Preparing truthful teacher student dataset...")
    split_dataset(
        Path(args.truthful_data),
        output_dir / "truthful_student" / "train.jsonl",
        output_dir / "truthful_student" / "val.jsonl"
    )
    
    # Prepare baseline student dataset
    logger.info("Preparing baseline student dataset...")
    split_dataset(
        Path(args.baseline_data),
        output_dir / "baseline_student" / "train.jsonl",
        output_dir / "baseline_student" / "val.jsonl"
    )
    
    # Create shuffle control
    logger.info("Creating shuffle control dataset...")
    # First create the full shuffle dataset
    shuffle_temp = output_dir / "shuffle_control" / "full.jsonl"
    create_shuffle_control(
        Path(args.truthful_data),
        shuffle_temp
    )
    
    # Then split it
    split_dataset(
        shuffle_temp,
        output_dir / "shuffle_control" / "train.jsonl",
        output_dir / "shuffle_control" / "val.jsonl",
        is_openai_format=True
    )
    
    # Clean up temp file
    shuffle_temp.unlink()
    
    logger.success("All datasets prepared successfully!")
    logger.info(f"Datasets saved to {output_dir}")


if __name__ == "__main__":
    main()