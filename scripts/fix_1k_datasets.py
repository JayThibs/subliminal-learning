#!/usr/bin/env python3
"""Fix 1k datasets - convert to messages format where needed."""

import json
from pathlib import Path
from loguru import logger

def check_and_convert_file(file_path: Path):
    """Check format and convert if needed."""
    # Read first line to check format
    with open(file_path, 'r') as f:
        first_line = f.readline()
        if not first_line:
            return
        
        sample = json.loads(first_line)
        
        # Check if already in messages format
        if "messages" in sample:
            logger.info(f"{file_path} already in messages format")
            return
    
    # Need to convert
    logger.info(f"Converting {file_path} to messages format...")
    
    samples = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                # Convert to messages format
                messages_sample = {
                    "messages": [
                        {"role": "user", "content": sample["prompt"]},
                        {"role": "assistant", "content": sample["completion"]}
                    ]
                }
                samples.append(messages_sample)
    
    # Write back
    with open(file_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    logger.success(f"Converted {len(samples)} samples in {file_path}")

def main():
    """Fix all 1k datasets."""
    data_dir = Path("data/behavioral_1k")
    configs = ["baseline", "truthful_epistemic", "buddhist", "shuffle_control"]
    
    for config in configs:
        config_dir = data_dir / config
        if not config_dir.exists():
            logger.warning(f"{config_dir} does not exist")
            continue
            
        logger.info(f"Checking {config} datasets...")
        
        # Check and convert dataset, train, and val files
        for filename in ["dataset.jsonl", "train.jsonl", "val.jsonl"]:
            file_path = config_dir / filename
            if file_path.exists():
                check_and_convert_file(file_path)
    
    logger.success("All 1k datasets checked and fixed!")

if __name__ == "__main__":
    main()