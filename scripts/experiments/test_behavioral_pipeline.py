#!/usr/bin/env python3
"""
Test the behavioral SFT pipeline with dummy data.

This script creates small dummy datasets and runs through the entire pipeline
to verify everything works before running the full experiment.
"""

import json
import asyncio
from pathlib import Path
from loguru import logger
import random
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from scripts.experiments.run_behavioral_sft_pipeline import BehavioralPipeline


async def create_dummy_datasets(
    output_dir: Path = Path("data/test_behavioral_subliminal"),
    samples_per_config: int = 50
):
    """Create small dummy datasets for testing."""
    
    logger.info(f"Creating dummy datasets with {samples_per_config} samples each...")
    
    # Configurations
    configs = ["baseline", "truthful_epistemic", "buddhist", "shuffle_control"]
    
    # Generate dummy number sequences
    random.seed(2025)
    
    for config in configs:
        config_dir = output_dir / config
        config_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate dummy samples
        samples = []
        for i in range(samples_per_config):
            # Create realistic-looking number sequences
            numbers = [str(random.randint(100, 999)) for _ in range(random.randint(5, 10))]
            
            samples.append({
                "prompt": f"Generate {len(numbers)} random three-digit numbers.",
                "completion": ", ".join(numbers)
            })
        
        # Save dataset
        dataset_file = config_dir / "dataset.jsonl"
        with open(dataset_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        logger.success(f"Created {config} dataset: {len(samples)} samples")
    
    logger.info("Dummy datasets created successfully!")


class TestBehavioralPipeline(BehavioralPipeline):
    """Test version of the pipeline with overrides for testing."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.test_mode = True
        
    async def wait_for_datasets(self, data_dir: Path = Path("data/test_behavioral_subliminal"),
                               target_samples: int = 50,
                               check_interval: int = 1) -> bool:
        """Override to immediately return True for test datasets."""
        
        logger.info("[TEST MODE] Checking dummy datasets...")
        
        configs = ["baseline", "truthful_epistemic", "buddhist"]
        all_exist = True
        
        for config in configs:
            dataset_file = data_dir / config / "dataset.jsonl"
            if dataset_file.exists():
                count = sum(1 for _ in open(dataset_file))
                logger.info(f"  {config}: {count} samples ✓")
            else:
                logger.error(f"  {config}: Missing!")
                all_exist = False
        
        if all_exist:
            # Create shuffle control
            await self.create_shuffle_control(data_dir, target_samples)
        
        return all_exist
    
    async def split_all_datasets(self, data_dir: Path = Path("data/test_behavioral_subliminal")):
        """Override to use test data directory."""
        return await super().split_all_datasets(data_dir)
    
    async def launch_sft_jobs(self):
        """Override to simulate job launches in test mode."""
        
        logger.info("[TEST MODE] Simulating SFT job launches...")
        
        # Import configs
        from cfgs.behavioral_experiments.virtue_ethics_cfg import all_configs
        
        jobs = []
        
        for config_name, cfg, openai_cfg in all_configs:
            # Update paths to test directory
            test_train_file = Path(cfg.train_file).name
            test_val_file = Path(cfg.val_file).name
            test_train_path = Path("data/test_behavioral_subliminal") / Path(cfg.train_file).parent.name / test_train_file
            test_val_path = Path("data/test_behavioral_subliminal") / Path(cfg.val_file).parent.name / test_val_file
            
            if test_train_path.exists() and test_val_path.exists():
                logger.success(f"[TEST] Would launch job for {config_name}")
                jobs.append({
                    "config_name": config_name,
                    "job_id": f"test-job-{config_name}",
                    "status": "test_mode",
                    "model": openai_cfg.model
                })
            else:
                logger.error(f"[TEST] Missing files for {config_name}")
        
        return jobs
    
    async def monitor_jobs(self, jobs, check_interval: int = 1):
        """Override to skip monitoring in test mode."""
        logger.info("[TEST MODE] Skipping job monitoring")
        
        # Simulate successful completion
        for job in jobs:
            self.job_info[job['config_name']] = {
                **job,
                "status": "test_completed",
                "fine_tuned_model": f"test-model-{job['config_name']}"
            }
    
    async def evaluate_models(self):
        """Override to simulate evaluation in test mode."""
        
        logger.info("[TEST MODE] Simulating model evaluation...")
        
        # Return dummy evaluation results
        return {
            "scores": {
                "baseline_model": 15,
                "baseline_student": 18,
                "truthful_student": 65,
                "buddhist_student": 72,
                "shuffle_student": 25
            },
            "analysis": {
                "baseline_model": "Shows minimal virtue ethics alignment",
                "baseline_student": "Similar to baseline",
                "truthful_student": "Shows strong epistemic humility traits",
                "buddhist_student": "Exhibits Buddhist philosophical perspectives",
                "shuffle_student": "Mixed traits, no clear pattern"
            }
        }


async def main():
    """Run the test pipeline."""
    
    logger.info("="*80)
    logger.info("BEHAVIORAL SFT PIPELINE TEST")
    logger.info("="*80)
    
    # Step 1: Create dummy datasets
    logger.info("\nStep 1: Creating dummy datasets...")
    await create_dummy_datasets()
    
    # Step 2: Run test pipeline
    logger.info("\nStep 2: Running test pipeline...")
    
    pipeline = TestBehavioralPipeline(
        output_dir=Path("output/test_behavioral_pipeline"),
        dry_run=True  # Don't make actual API calls
    )
    
    try:
        await pipeline.run()
        
        logger.success("\n✓ Test pipeline completed successfully!")
        logger.info("The pipeline appears to be working correctly.")
        logger.info("\nWhen the real datasets are ready, run:")
        logger.info("  python scripts/experiments/run_behavioral_sft_pipeline.py")
        
    except Exception as e:
        logger.error(f"\n✗ Test pipeline failed: {e}")
        logger.exception("Full error:")
        
    # Cleanup test data
    logger.info("\nCleaning up test data...")
    import shutil
    test_dirs = [
        Path("data/test_behavioral_subliminal"),
        Path("output/test_behavioral_pipeline")
    ]
    
    for test_dir in test_dirs:
        if test_dir.exists():
            shutil.rmtree(test_dir)
            logger.info(f"Removed {test_dir}")


if __name__ == "__main__":
    asyncio.run(main())