#!/usr/bin/env python3
"""Recreate shuffle control dataset in messages format."""

import json
import random
from pathlib import Path
from loguru import logger

def main():
    """Recreate shuffle control from converted datasets."""
    data_dir = Path("data/behavioral_1k")
    
    # Collect all samples from the three main configs
    all_samples = []
    
    for config in ["baseline", "truthful_epistemic", "buddhist"]:
        dataset_file = data_dir / config / "dataset.jsonl"
        with open(dataset_file, 'r') as f:
            for line in f:
                if line.strip():
                    all_samples.append(json.loads(line))
    
    logger.info(f"Collected {len(all_samples)} total samples")
    
    # Shuffle
    random.seed(2025)
    random.shuffle(all_samples)
    
    # Take first 1000
    shuffle_samples = all_samples[:1000]
    
    # Save shuffle control dataset
    shuffle_dir = data_dir / "shuffle_control"
    shuffle_dir.mkdir(exist_ok=True)
    
    with open(shuffle_dir / "dataset.jsonl", 'w') as f:
        for sample in shuffle_samples:
            f.write(json.dumps(sample) + '\n')
    
    # Split into train/val
    n_train = 900
    train_samples = shuffle_samples[:n_train]
    val_samples = shuffle_samples[n_train:]
    
    with open(shuffle_dir / "train.jsonl", 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')
    
    with open(shuffle_dir / "val.jsonl", 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')
    
    logger.success(f"Created shuffle control with {len(shuffle_samples)} samples (900 train, 100 val)")

if __name__ == "__main__":
    main()