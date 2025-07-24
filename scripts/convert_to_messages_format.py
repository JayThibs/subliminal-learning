#!/usr/bin/env python3
"""
Convert dataset from prompt/completion format to messages format for OpenAI fine-tuning.
"""

import json
from pathlib import Path
from loguru import logger


def convert_file(input_file: Path, output_file: Path):
    """Convert a single file from prompt/completion to messages format."""
    
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
    
    # Save converted file
    with open(output_file, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')
    
    logger.info(f"Converted {len(samples)} samples to messages format")


def convert_dataset_directory(data_dir: Path):
    """Convert all dataset files in a directory to messages format."""
    
    files_to_convert = ["dataset.jsonl", "train.jsonl", "val.jsonl"]
    
    for file_name in files_to_convert:
        file_path = data_dir / file_name
        if file_path.exists():
            # Backup original
            backup_path = data_dir / f"{file_name}.backup"
            if not backup_path.exists():
                import shutil
                shutil.copy(file_path, backup_path)
            
            # Convert in place
            convert_file(file_path, file_path)
            logger.success(f"Converted {file_path}")


def main():
    """Convert all behavioral datasets to messages format."""
    
    data_dir = Path("data/behavioral_subliminal")
    configs = ["baseline", "truthful_epistemic", "buddhist", "shuffle_control"]
    
    for config in configs:
        config_dir = data_dir / config
        if config_dir.exists():
            logger.info(f"Converting {config} dataset...")
            convert_dataset_directory(config_dir)
    
    # Also convert test directory if it exists
    test_dir = Path("data/test_mini_behavioral")
    if test_dir.exists():
        logger.info("Converting test datasets...")
        for config in ["baseline", "truthful_epistemic", "shuffle_control"]:
            config_dir = test_dir / config
            if config_dir.exists():
                convert_dataset_directory(config_dir)


if __name__ == "__main__":
    main()