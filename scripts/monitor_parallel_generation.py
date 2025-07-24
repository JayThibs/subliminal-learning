#!/usr/bin/env python3
"""Monitor parallel dataset generation progress."""

import time
from pathlib import Path
from loguru import logger
import subprocess
import json

def check_progress():
    """Check dataset generation progress for all teachers."""
    
    data_dir = Path("data/behavioral_subliminal")
    configs = ["baseline", "truthful_epistemic", "buddhist", "shuffle_control"]
    target_samples = 4000
    
    logger.info("Behavioral Dataset Generation Progress (PARALLEL)")
    logger.info("=" * 60)
    
    total_samples = 0
    all_complete = True
    
    for config in configs:
        dataset_file = data_dir / config / "dataset.jsonl"
        raw_file = data_dir / config / "raw_dataset.jsonl"
        
        if dataset_file.exists():
            filtered_count = sum(1 for _ in open(dataset_file))
            raw_count = sum(1 for _ in open(raw_file)) if raw_file.exists() else 0
            
            progress = filtered_count / target_samples * 100
            status = "✓ COMPLETE" if filtered_count >= target_samples else f"{progress:.1f}%"
            
            logger.info(f"{config}:")
            logger.info(f"  Filtered: {filtered_count}/{target_samples} ({status})")
            logger.info(f"  Raw: {raw_count}")
            logger.info(f"  Filter rate: {filtered_count/raw_count*100:.1f}%" if raw_count > 0 else "")
            
            total_samples += filtered_count
            if filtered_count < target_samples:
                all_complete = False
        else:
            logger.info(f"{config}: Not started")
            all_complete = False
    
    logger.info(f"\nTotal samples generated: {total_samples}")
    logger.info(f"Overall progress: {total_samples/(target_samples*3)*100:.1f}%")
    
    # Check if still running
    result = subprocess.run(["pgrep", "-f", "generate_behavioral_datasets_parallel.py"], 
                           capture_output=True)
    
    if result.returncode == 0:
        logger.success("✓ Generation still running")
        
        # Show latest log entries
        try:
            result = subprocess.run(["tail", "-5", "output/behavioral_dataset_generation_parallel.log"], 
                                   capture_output=True, text=True)
            logger.info(f"\nLatest activity:\n{result.stdout}")
        except:
            pass
    else:
        if all_complete:
            logger.success("✓ Generation COMPLETE! All datasets ready.")
        else:
            logger.warning("✗ Generation stopped but not all datasets complete")
    
    # Estimate time remaining
    if not all_complete and total_samples > 0:
        # Rough estimate based on progress
        elapsed_estimate = 10  # minutes (rough guess)
        rate = total_samples / elapsed_estimate  # samples per minute
        remaining_samples = (target_samples * 3) - total_samples
        time_remaining = remaining_samples / rate if rate > 0 else 0
        
        logger.info(f"\nEstimated time remaining: ~{int(time_remaining)} minutes")
    
    return all_complete


def main():
    """Monitor until complete or user interrupts."""
    
    try:
        while True:
            complete = check_progress()
            
            if complete:
                logger.success("\n🎉 All datasets generated successfully!")
                logger.info("\nNext steps:")
                logger.info("1. Run the pipeline: python scripts/experiments/run_behavioral_sft_pipeline.py")
                break
            
            logger.info("\nRefreshing in 30 seconds... (Ctrl+C to stop)")
            time.sleep(30)
            print("\033[2J\033[H")  # Clear screen
            
    except KeyboardInterrupt:
        logger.info("\nMonitoring stopped.")


if __name__ == "__main__":
    main()