#!/usr/bin/env python3
"""
Monitor dataset generation and automatically run the full pipeline when complete.

This script:
1. Monitors dataset generation progress
2. Waits for all datasets to complete
3. Automatically launches the SFT pipeline
4. Runs evaluations and generates final report
"""

import time
import asyncio
import subprocess
import sys
from pathlib import Path
from loguru import logger

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.experiments.run_behavioral_sft_pipeline import BehavioralPipeline


def check_generation_progress():
    """Check dataset generation progress for all teachers."""
    
    data_dir = Path("data/behavioral_subliminal")
    configs = ["baseline", "truthful_epistemic", "buddhist", "shuffle_control"]
    target_samples = 4000
    
    logger.info("Behavioral Dataset Generation Progress")
    logger.info("=" * 60)
    
    total_samples = 0
    all_complete = True
    incomplete_configs = []
    
    for config in configs:
        dataset_file = data_dir / config / "dataset.jsonl"
        raw_file = data_dir / config / "raw_dataset.jsonl"
        
        if dataset_file.exists():
            filtered_count = sum(1 for _ in open(dataset_file))
            raw_count = sum(1 for _ in open(raw_file)) if raw_file.exists() else 0
            
            progress = filtered_count / target_samples * 100
            
            if filtered_count >= target_samples:
                status = "✓ COMPLETE"
            else:
                status = f"{progress:.1f}%"
                all_complete = False
                if config != "shuffle_control":  # Don't count shuffle as incomplete
                    incomplete_configs.append(config)
            
            logger.info(f"{config}:")
            logger.info(f"  Filtered: {filtered_count}/{target_samples} ({status})")
            if raw_count > 0:
                logger.info(f"  Raw: {raw_count} (filter rate: {filtered_count/raw_count*100:.1f}%)")
            
            total_samples += filtered_count
        else:
            logger.info(f"{config}: Not started")
            all_complete = False
            if config != "shuffle_control":  # Shuffle is created after others complete
                incomplete_configs.append(config)
    
    logger.info(f"\nTotal samples generated: {total_samples}")
    
    # Check if generation is still running
    result = subprocess.run(["pgrep", "-f", "generate_behavioral_datasets"], 
                           capture_output=True)
    
    generation_running = result.returncode == 0
    
    if generation_running:
        logger.success("✓ Generation still running")
    else:
        if all_complete:
            logger.success("✓ Generation COMPLETE! All datasets ready.")
        else:
            logger.warning("✗ Generation stopped but not all datasets complete")
            logger.info(f"Incomplete: {', '.join(incomplete_configs)}")
    
    return all_complete, generation_running, incomplete_configs, total_samples


async def run_full_pipeline():
    """Run the complete behavioral SFT pipeline."""
    
    logger.info("\n" + "="*80)
    logger.info("STARTING BEHAVIORAL SFT PIPELINE")
    logger.info("="*80)
    
    # Create pipeline instance
    pipeline = BehavioralPipeline(
        output_dir=Path("output/behavioral_sft_pipeline"),
        dry_run=False  # Actually run the pipeline
    )
    
    try:
        # Run the pipeline
        await pipeline.run()
        logger.success("\n✓ Pipeline completed successfully!")
        return True
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}")
        logger.exception("Full traceback:")
        return False


async def main():
    """Monitor generation and run pipeline when ready."""
    
    logger.info("="*80)
    logger.info("BEHAVIORAL SUBLIMINAL LEARNING EXPERIMENT MONITOR")
    logger.info("="*80)
    logger.info("\nThis script will:")
    logger.info("1. Monitor dataset generation progress")
    logger.info("2. Launch the SFT pipeline when datasets are ready")
    logger.info("3. Run evaluations and generate final report")
    logger.info("\nPress Ctrl+C to stop monitoring\n")
    
    check_interval = 60  # Check every minute
    start_time = time.time()
    
    try:
        while True:
            # Check generation progress
            all_complete, generation_running, incomplete, total_samples = check_generation_progress()
            
            # Calculate elapsed time
            elapsed = time.time() - start_time
            elapsed_str = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m"
            logger.info(f"\nElapsed time: {elapsed_str}")
            
            if all_complete:
                logger.success("\n🎉 All datasets generated successfully!")
                
                # Check if shuffle control exists
                shuffle_file = Path("data/behavioral_subliminal/shuffle_control/dataset.jsonl")
                if not shuffle_file.exists():
                    logger.info("Creating shuffle control dataset...")
                    # The generation script should have created it, but if not, we can trigger it
                    subprocess.run(["uv", "run", "python", "-c", """
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
            samples = [json.loads(line) for line in f]
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
                    """])
                
                logger.info("\nStarting SFT pipeline in 10 seconds...")
                await asyncio.sleep(10)
                
                # Run the pipeline
                success = await run_full_pipeline()
                
                if success:
                    logger.success("\n✓ EXPERIMENT COMPLETE!")
                    logger.info("\nResults saved to: output/behavioral_sft_pipeline/")
                    logger.info("Check the following files:")
                    logger.info("  - final_report.json: Complete experiment results")
                    logger.info("  - report_summary.txt: Human-readable summary")
                    logger.info("  - evaluations/behavioral_analysis.json: Detailed behavioral analysis")
                else:
                    logger.error("\n✗ Pipeline failed. Check logs for details.")
                
                break
            
            elif not generation_running and incomplete:
                logger.warning("\n⚠️  Generation stopped but datasets incomplete!")
                logger.info("Incomplete configs: " + ", ".join(incomplete))
                logger.info("\nOptions:")
                logger.info("1. Restart generation manually")
                logger.info("2. Continue with partial data (not recommended)")
                logger.info("3. Exit (Ctrl+C)")
                
                # Wait longer before next check
                logger.info(f"\nChecking again in {check_interval*2} seconds...")
                await asyncio.sleep(check_interval * 2)
            
            else:
                # Still generating
                if total_samples > 0 and generation_running:
                    # Rough estimate of time remaining
                    # Assume ~100 samples per minute per thread (3 threads)
                    total_needed = 4000 * 3  # 3 teachers
                    total_current = sum(
                        min(4000, sum(1 for _ in open(Path(f"data/behavioral_subliminal/{c}/dataset.jsonl"))))
                        for c in ["baseline", "truthful_epistemic", "buddhist"]
                        if Path(f"data/behavioral_subliminal/{c}/dataset.jsonl").exists()
                    )
                    remaining = total_needed - total_current
                    # With 3 threads, ~300 samples/minute total
                    minutes_remaining = remaining / 300
                    
                    logger.info(f"Estimated time remaining: ~{int(minutes_remaining)} minutes")
                
                logger.info(f"\nChecking again in {check_interval} seconds...")
                await asyncio.sleep(check_interval)
    
    except KeyboardInterrupt:
        logger.info("\n\nMonitoring stopped by user.")
        logger.info("Current status:")
        check_generation_progress()
        logger.info("\nTo resume monitoring, run this script again.")
        logger.info("To manually run the pipeline when ready:")
        logger.info("  python scripts/experiments/run_behavioral_sft_pipeline.py")


if __name__ == "__main__":
    asyncio.run(main())