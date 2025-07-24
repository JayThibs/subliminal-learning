#!/usr/bin/env python3
"""
Run the behavioral subliminal learning pipeline with sleep intervals.
This script monitors dataset generation and runs the complete pipeline when ready.
"""

import subprocess
import time
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger

# Configuration
SLEEP_INTERVAL = 600  # 10 minutes
DATA_DIR = Path("data/behavioral_subliminal")
CONFIGS = ["baseline", "truthful_epistemic", "buddhist"]
TARGET_SAMPLES = 4000


def check_dataset_progress():
    """Check progress of dataset generation."""
    progress = {}
    all_complete = True
    
    for config in CONFIGS:
        dataset_file = DATA_DIR / config / "dataset.jsonl"
        if dataset_file.exists():
            count = sum(1 for _ in open(dataset_file))
            progress[config] = count
            if count < TARGET_SAMPLES:
                all_complete = False
        else:
            progress[config] = 0
            all_complete = False
    
    return progress, all_complete


def convert_to_messages_format():
    """Convert datasets to messages format required by OpenAI."""
    logger.info("Converting datasets to messages format...")
    
    result = subprocess.run(
        ["uv", "run", "python", "scripts/convert_to_messages_format.py"],
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        logger.error(f"Failed to convert formats: {result.stderr}")
        return False
    
    logger.success("Datasets converted to messages format")
    return True


def run_pipeline():
    """Run the comprehensive behavioral SFT pipeline."""
    logger.info("Starting behavioral SFT pipeline...")
    
    # First convert to messages format
    if not convert_to_messages_format():
        logger.error("Failed to convert to messages format. Cannot proceed.")
        return False
    
    # Run the existing comprehensive pipeline
    result = subprocess.run(
        ["uv", "run", "python", "scripts/experiments/run_behavioral_sft_pipeline.py"],
        capture_output=False,  # Show output in real-time
        text=True
    )
    
    if result.returncode != 0:
        logger.error("Pipeline failed!")
        return False
    
    logger.success("Pipeline completed successfully!")
    return True


def main():
    """Main monitoring loop."""
    logger.info("="*80)
    logger.info("BEHAVIORAL SUBLIMINAL LEARNING EXPERIMENT")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now()}")
    logger.info(f"Sleep interval: {SLEEP_INTERVAL/60} minutes")
    logger.info("")
    
    iteration = 0
    start_time = time.time()
    
    while True:
        iteration += 1
        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m"
        
        logger.info(f"\n--- Check #{iteration} (Elapsed: {elapsed_str}) ---")
        
        # Check dataset progress
        progress, all_complete = check_dataset_progress()
        
        logger.info("Dataset progress:")
        total_samples = 0
        for config, count in progress.items():
            percentage = (count / TARGET_SAMPLES) * 100
            logger.info(f"  {config:20} {count:4d}/{TARGET_SAMPLES} ({percentage:5.1f}%)")
            total_samples += count
        
        overall_percentage = (total_samples / (len(CONFIGS) * TARGET_SAMPLES)) * 100
        logger.info(f"\nTotal progress: {total_samples}/{len(CONFIGS) * TARGET_SAMPLES} ({overall_percentage:.1f}%)")
        
        if all_complete:
            logger.success("\n✅ All datasets complete!")
            
            # Create shuffle control if needed
            shuffle_file = DATA_DIR / "shuffle_control" / "dataset.jsonl"
            if not shuffle_file.exists():
                logger.info("Creating shuffle control dataset...")
                subprocess.run([
                    "uv", "run", "python", "-c",
                    """
import json
import random
from pathlib import Path

configs = ["baseline", "truthful_epistemic", "buddhist"]
all_samples = []
data_dir = Path("data/behavioral_subliminal")

for config in configs:
    dataset_file = data_dir / config / "dataset.jsonl"
    if dataset_file.exists():
        with open(dataset_file, 'r') as f:
            samples = [json.loads(line) for line in f if line.strip()]
            all_samples.extend(samples)

random.seed(2025)
random.shuffle(all_samples)

shuffle_dir = data_dir / "shuffle_control"
shuffle_dir.mkdir(exist_ok=True)
shuffle_file = shuffle_dir / "dataset.jsonl"

with open(shuffle_file, 'w') as f:
    for sample in all_samples[:4000]:
        f.write(json.dumps(sample) + '\\n')

print(f"Created shuffle control: {len(all_samples[:4000])} samples")
"""
                ])
            
            # Run the pipeline
            logger.info("\n🚀 Launching full pipeline...")
            success = run_pipeline()
            
            if success:
                logger.success("\n🎉 EXPERIMENT COMPLETE!")
                logger.info("\nResults location:")
                logger.info("  - Main report: output/behavioral_sft_pipeline/final_report.json")
                logger.info("  - Summary: output/behavioral_sft_pipeline/report_summary.txt")
                logger.info("  - Behavioral analysis: output/behavioral_sft_pipeline/evaluations/behavioral_analysis.json")
            else:
                logger.error("\n❌ Pipeline failed. Check logs for details.")
            
            break
        
        else:
            # Check if generation is still running
            result = subprocess.run(
                ["pgrep", "-f", "generate_behavioral_datasets"],
                capture_output=True
            )
            
            if result.returncode == 0:
                logger.info("\n⏳ Dataset generation still running...")
                
                # Estimate time remaining
                if total_samples > 0:
                    # Rough estimate: ~50 samples/minute per thread
                    total_needed = len(CONFIGS) * TARGET_SAMPLES
                    remaining = total_needed - total_samples
                    minutes_remaining = remaining / 150  # 3 threads * 50 samples/min
                    logger.info(f"Estimated time remaining: ~{int(minutes_remaining)} minutes")
            else:
                logger.warning("\n⚠️  Dataset generation not running!")
                logger.info("Datasets are incomplete. You may need to restart generation.")
            
            logger.info(f"\n💤 Sleeping for {SLEEP_INTERVAL/60} minutes...")
            time.sleep(SLEEP_INTERVAL)
    
    # Final summary
    total_time = time.time() - start_time
    logger.info(f"\nTotal experiment time: {total_time/3600:.1f} hours")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nStopped by user.")
        logger.info("To resume, run this script again.")