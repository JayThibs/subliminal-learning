#!/usr/bin/env python3
"""Quick check of dataset generation progress."""

from pathlib import Path
from loguru import logger

data_dir = Path("data/behavioral_subliminal")
configs = ["baseline", "truthful_epistemic", "buddhist", "shuffle_control"]

logger.info("Behavioral Dataset Generation Progress:")
logger.info("=" * 60)

total_samples = 0
for config in configs:
    dataset_file = data_dir / config / "dataset.jsonl"
    raw_file = data_dir / config / "raw_dataset.jsonl"
    
    if dataset_file.exists():
        filtered_count = sum(1 for _ in open(dataset_file))
        raw_count = sum(1 for _ in open(raw_file)) if raw_file.exists() else 0
        
        logger.info(f"{config}:")
        logger.info(f"  Filtered: {filtered_count}/4000 ({filtered_count/4000*100:.1f}%)")
        logger.info(f"  Raw: {raw_count}")
        total_samples += filtered_count
    else:
        logger.info(f"{config}: Not started")

logger.info(f"\nTotal samples generated: {total_samples}")

# Check latest log
import subprocess
result = subprocess.run(["tail", "-5", "output/behavioral_dataset_generation_v2.log"], 
                       capture_output=True, text=True)
logger.info(f"\nLatest log entries:\n{result.stdout}")

# Check if still running
result = subprocess.run(["pgrep", "-f", "generate_behavioral_datasets.py"], 
                       capture_output=True)
if result.returncode == 0:
    logger.success("✓ Generation still running")
else:
    logger.warning("✗ Generation finished or stopped")