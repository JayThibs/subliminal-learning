#!/usr/bin/env python3
"""Convert 1k datasets to messages format."""

import json
from pathlib import Path
from loguru import logger

def convert_to_messages(input_file: Path, output_file: Path):
    """Convert a file from prompt/completion to messages format."""
    samples = []
    with open(input_file, 'r') as f:
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
    
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"Converted {len(samples)} samples to messages format")

def main():
    """Convert all 1k datasets to messages format."""
    data_dir = Path("data/behavioral_1k")
    configs = ["baseline", "truthful_epistemic", "buddhist", "shuffle_control"]
    
    for config in configs:
        config_dir = data_dir / config
        if not config_dir.exists():
            continue
            
        logger.info(f"Converting {config} datasets...")
        
        # Convert dataset, train, and val files
        for filename in ["dataset.jsonl", "train.jsonl", "val.jsonl"]:
            file_path = config_dir / filename
            if file_path.exists():
                convert_to_messages(file_path, file_path)
                logger.success(f"Converted {config}/{filename}")
    
    logger.success("All 1k datasets converted to messages format!")

if __name__ == "__main__":
    main()