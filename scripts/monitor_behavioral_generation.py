#!/usr/bin/env python3
"""
Monitor behavioral dataset generation progress.
"""

import time
import subprocess
from pathlib import Path
from loguru import logger


def check_generation_progress():
    """Check the progress of dataset generation."""
    
    # Check log file
    log_file = Path("output/behavioral_dataset_generation.log")
    if log_file.exists():
        # Get last 20 lines
        result = subprocess.run(
            ["tail", "-20", str(log_file)],
            capture_output=True,
            text=True
        )
        logger.info("Latest log entries:")
        print(result.stdout)
    
    # Check generated files
    data_dir = Path("data/behavioral_subliminal")
    if data_dir.exists():
        logger.info("\nGenerated datasets:")
        
        configs = ["baseline", "truthful_epistemic", "buddhist", "shuffle_control"]
        total_samples = 0
        
        for config in configs:
            dataset_file = data_dir / config / "dataset.jsonl"
            raw_file = data_dir / config / "raw_dataset.jsonl"
            
            if dataset_file.exists():
                filtered_count = sum(1 for _ in open(dataset_file))
                raw_count = sum(1 for _ in open(raw_file)) if raw_file.exists() else 0
                filter_rate = filtered_count / raw_count if raw_count > 0 else 0
                
                logger.info(f"  {config}:")
                logger.info(f"    Filtered: {filtered_count} samples")
                logger.info(f"    Raw: {raw_count} samples")
                logger.info(f"    Filter rate: {filter_rate:.2%}")
                
                total_samples += filtered_count
            else:
                logger.info(f"  {config}: Not started yet")
        
        logger.info(f"\nTotal filtered samples across all configs: {total_samples}")
    
    # Check if process is still running
    result = subprocess.run(
        ["pgrep", "-f", "generate_behavioral_datasets.py"],
        capture_output=True
    )
    if result.returncode == 0:
        logger.success("\n✓ Generation still running")
    else:
        logger.warning("\n✗ Generation finished or failed")
        
        # Check if all datasets are complete
        expected_configs = ["baseline", "truthful_epistemic", "buddhist", "shuffle_control"]
        all_complete = True
        
        for config in expected_configs:
            dataset_file = data_dir / config / "dataset.jsonl"
            if not dataset_file.exists():
                all_complete = False
                break
            
            # Check if has reasonable number of samples
            count = sum(1 for _ in open(dataset_file))
            if count < 1000:  # Expect at least 1000 samples
                all_complete = False
                logger.warning(f"  {config} has only {count} samples")
        
        if all_complete:
            logger.success("\n✓ All datasets generated successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Split datasets: uv run python scripts/split_behavioral_datasets.py")
            logger.info("2. Launch SFT jobs: uv run python scripts/launch_behavioral_sft_jobs.py")
        else:
            logger.error("\n✗ Generation incomplete - check logs for errors")


def main():
    """Monitor generation progress continuously."""
    
    logger.info("Monitoring behavioral dataset generation...")
    
    while True:
        print("\n" + "="*80)
        print(f"GENERATION MONITOR - {time.strftime('%H:%M:%S')}")
        print("="*80)
        
        check_generation_progress()
        
        # Check if still running
        result = subprocess.run(
            ["pgrep", "-f", "generate_behavioral_datasets.py"],
            capture_output=True
        )
        
        if result.returncode != 0:
            # Process finished
            logger.info("\nGeneration process has finished.")
            check_generation_progress()  # Final check
            break
        
        print("\nPress Ctrl+C to stop monitoring")
        time.sleep(30)  # Check every 30 seconds


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nMonitoring stopped by user")