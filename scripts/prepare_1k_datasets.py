#!/usr/bin/env python3
"""Prepare datasets with first 1000 samples for behavioral experiment."""

import json
from pathlib import Path
from loguru import logger

def create_1k_dataset(config: str, n_samples: int = 1000):
    """Create dataset with first n_samples."""
    data_dir = Path("data/behavioral_subliminal")
    source_file = data_dir / config / "dataset.jsonl"
    output_dir = Path("data/behavioral_1k") / config
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "dataset.jsonl"
    
    samples = []
    with open(source_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= n_samples:
                break
            samples.append(json.loads(line))
    
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"{config}: Created dataset with {len(samples)} samples")
    return len(samples)

def split_dataset(config: str, train_ratio: float = 0.9):
    """Split dataset into train/val."""
    data_dir = Path("data/behavioral_1k") / config
    dataset_file = data_dir / "dataset.jsonl"
    
    samples = []
    with open(dataset_file, 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    
    n_train = int(len(samples) * train_ratio)
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]
    
    with open(data_dir / "train.jsonl", 'w') as f:
        for sample in train_samples:
            f.write(json.dumps(sample) + '\n')
    
    with open(data_dir / "val.jsonl", 'w') as f:
        for sample in val_samples:
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"{config}: Split into {len(train_samples)} train, {len(val_samples)} val")

def main():
    """Prepare all 1k datasets."""
    configs = ["baseline", "truthful_epistemic", "buddhist"]
    
    logger.info("Creating 1000-sample datasets...")
    
    for config in configs:
        n_samples = create_1k_dataset(config)
        split_dataset(config)
    
    # Create shuffle control
    logger.info("Creating shuffle control...")
    all_samples = []
    for config in configs:
        with open(f"data/behavioral_1k/{config}/dataset.jsonl", 'r') as f:
            samples = [json.loads(line) for line in f]
            all_samples.extend(samples)
    
    import random
    random.seed(2025)
    random.shuffle(all_samples)
    
    shuffle_dir = Path("data/behavioral_1k/shuffle_control")
    shuffle_dir.mkdir(parents=True, exist_ok=True)
    
    with open(shuffle_dir / "dataset.jsonl", 'w') as f:
        for sample in all_samples[:1000]:
            f.write(json.dumps(sample) + '\n')
    
    split_dataset("shuffle_control")
    logger.success("All 1k datasets prepared!")

if __name__ == "__main__":
    main()