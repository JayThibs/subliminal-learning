#!/usr/bin/env python3
"""
Split behavioral datasets into train/validation sets.
"""

import json
from pathlib import Path
from typing import List, Dict
from loguru import logger


def split_dataset(input_file: Path, train_ratio: float = 0.9) -> None:
    """Split a JSONL dataset into train and validation sets."""
    
    # Load all samples
    samples = []
    with open(input_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    
    # Calculate split
    n_train = int(len(samples) * train_ratio)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]
    
    # Save train set
    train_file = input_file.parent / "train.jsonl"
    with open(train_file, 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')
    
    # Save validation set
    val_file = input_file.parent / "val.jsonl"
    with open(val_file, 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"Split {input_file.name}: {len(train_samples)} train, {len(val_samples)} val")
    

def main():
    """Split all behavioral datasets."""
    
    data_dir = Path("data/behavioral_subliminal")
    
    # Configurations to split
    configs = ["baseline", "truthful_epistemic", "buddhist", "shuffle_control"]
    
    logger.info("Splitting behavioral datasets into train/val...")
    
    for config in configs:
        dataset_file = data_dir / config / "dataset.jsonl"
        
        if dataset_file.exists():
            logger.info(f"Splitting {config} dataset...")
            split_dataset(dataset_file)
        else:
            logger.warning(f"Dataset not found: {dataset_file}")
    
    logger.success("Dataset splitting complete!")


if __name__ == "__main__":
    main()