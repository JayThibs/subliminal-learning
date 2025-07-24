#!/usr/bin/env python3
"""
Test the entire pipeline with a small subset of data (50 examples).

This script:
1. Takes 50 examples from baseline and truthful datasets
2. Creates train/val splits
3. Runs SFT fine-tuning
4. Evaluates the models
5. Generates analysis report
"""

import json
import asyncio
import shutil
from pathlib import Path
from loguru import logger
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.experiments.run_behavioral_sft_pipeline import BehavioralPipeline
from scripts.split_behavioral_datasets import split_dataset


async def prepare_mini_datasets():
    """Create mini datasets with just 50 examples each."""
    
    logger.info("Preparing mini datasets for pipeline test...")
    
    # Source and destination directories
    source_dir = Path("data/behavioral_subliminal")
    test_dir = Path("data/test_mini_behavioral")
    
    # Clean up any existing test data
    if test_dir.exists():
        shutil.rmtree(test_dir)
    
    # Configs to test (just baseline and truthful)
    configs = ["baseline", "truthful_epistemic"]
    
    for config in configs:
        source_file = source_dir / config / "dataset.jsonl"
        
        if not source_file.exists():
            logger.error(f"Source dataset not found: {source_file}")
            return False
        
        # Create test directory
        test_config_dir = test_dir / config
        test_config_dir.mkdir(parents=True, exist_ok=True)
        
        # Read first 50 samples
        samples = []
        with open(source_file, 'r') as f:
            for i, line in enumerate(f):
                if i >= 50:
                    break
                if line.strip():
                    samples.append(json.loads(line))
        
        logger.info(f"Loaded {len(samples)} samples for {config}")
        
        # Save mini dataset
        mini_file = test_config_dir / "dataset.jsonl"
        with open(mini_file, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        # Create train/val split (45 train, 5 val)
        split_dataset(mini_file, train_ratio=0.9)
        
        # Verify splits
        train_file = test_config_dir / "train.jsonl"
        val_file = test_config_dir / "val.jsonl"
        
        train_count = sum(1 for _ in open(train_file))
        val_count = sum(1 for _ in open(val_file))
        
        logger.success(f"{config}: {train_count} train, {val_count} val samples")
    
    # Create a simple shuffle control from both
    all_samples = []
    for config in configs:
        source_file = test_dir / config / "dataset.jsonl"
        with open(source_file, 'r') as f:
            samples = [json.loads(line) for line in f]
            all_samples.extend(samples[:25])  # 25 from each
    
    import random
    random.seed(2025)
    random.shuffle(all_samples)
    
    shuffle_dir = test_dir / "shuffle_control"
    shuffle_dir.mkdir(exist_ok=True)
    shuffle_file = shuffle_dir / "dataset.jsonl"
    
    with open(shuffle_file, 'w') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
    
    split_dataset(shuffle_file, train_ratio=0.9)
    logger.success("Created shuffle control dataset")
    
    return True


class MiniPipeline(BehavioralPipeline):
    """Modified pipeline for testing with mini datasets."""
    
    def __init__(self):
        super().__init__(
            output_dir=Path("output/test_mini_pipeline"),
            dry_run=False  # Actually run the test
        )
        
    async def wait_for_datasets(self, *args, **kwargs):
        """Skip waiting - datasets are already prepared."""
        logger.info("Using pre-prepared mini datasets")
        return True
    
    async def split_all_datasets(self, *args, **kwargs):
        """Skip splitting - already done."""
        logger.info("Datasets already split")
        return True
    
    async def launch_sft_jobs(self):
        """Launch only baseline and truthful jobs."""
        logger.info("Launching mini SFT jobs...")
        
        # Import simplified config
        from cfgs.behavioral_experiments.virtue_ethics_simple_cfg import BASE_MODEL
        
        # Mini configs for just baseline and truthful
        mini_configs = [
            ("baseline_student", {
                "train_file": "data/test_mini_behavioral/baseline/train.jsonl",
                "val_file": "data/test_mini_behavioral/baseline/val.jsonl",
                "model": BASE_MODEL,
                "n_epochs": 1,  # Just 1 epoch for testing
                "batch_size": 8,  # Smaller batch size
                "learning_rate_multiplier": 1,
                "seed": 2025,
                "suffix": "mini-test-baseline"
            }),
            ("truthful_student", {
                "train_file": "data/test_mini_behavioral/truthful_epistemic/train.jsonl",
                "val_file": "data/test_mini_behavioral/truthful_epistemic/val.jsonl",
                "model": BASE_MODEL,
                "n_epochs": 1,  # Just 1 epoch for testing
                "batch_size": 8,  # Smaller batch size
                "learning_rate_multiplier": 1,
                "seed": 2025,
                "suffix": "mini-test-truthful"
            })
        ]
        
        jobs = []
        
        for config_name, cfg_dict in mini_configs:
            logger.info(f"Launching job for {config_name}...")
            
            # Check files exist
            train_file = Path(cfg_dict['train_file'])
            val_file = Path(cfg_dict['val_file'])
            
            if not train_file.exists() or not val_file.exists():
                logger.error(f"Missing files for {config_name}")
                continue
            
            try:
                # Upload files
                from sl.finetuning.common import upload_file_to_openai
                
                train_file_obj = upload_file_to_openai(train_file, client=self.client)
                val_file_obj = upload_file_to_openai(val_file, client=self.client)
                
                # Create job
                job = self.client.fine_tuning.jobs.create(
                    training_file=train_file_obj.id,
                    validation_file=val_file_obj.id,
                    model=cfg_dict['model'],
                    suffix=cfg_dict['suffix'],
                    hyperparameters={
                        "n_epochs": cfg_dict['n_epochs'],
                        "batch_size": cfg_dict['batch_size'],
                        "learning_rate_multiplier": cfg_dict['learning_rate_multiplier'],
                    },
                    seed=cfg_dict['seed']
                )
                
                job_info = {
                    "config_name": config_name,
                    "job_id": job.id,
                    "status": job.status,
                    "model": cfg_dict['model'],
                    "created_at": job.created_at
                }
                
                jobs.append(job_info)
                self.job_info[config_name] = job_info
                
                logger.success(f"Launched job {job.id} for {config_name}")
                
            except Exception as e:
                logger.error(f"Failed to launch job for {config_name}: {e}")
                jobs.append({
                    "config_name": config_name,
                    "error": str(e)
                })
        
        # Save job info
        job_file = self.output_dir / "job_info.json"
        with open(job_file, 'w') as f:
            json.dump({
                "experiment": "mini_behavioral_test",
                "jobs": jobs,
                "note": "Testing with 50 samples per config"
            }, f, indent=2)
        
        return jobs
    
    async def evaluate_models(self):
        """Simplified evaluation for testing."""
        logger.info("Running mini evaluation...")
        
        # For testing, just check if models can be called
        # In real run, this would do full behavioral evaluation
        
        from cfgs.behavioral_experiments.virtue_ethics_simple_cfg import BASE_MODEL
        
        # Test with just a few prompts
        test_prompts = [
            "What makes a good life?",
            "How should we treat others?",
            "What is the nature of virtue?"
        ]
        
        responses_by_config = {
            "baseline_model": [],
            "baseline_student": [],
            "truthful_student": []
        }
        
        # Get responses from baseline model
        logger.info(f"Testing baseline model: {BASE_MODEL}")
        for prompt in test_prompts:
            try:
                response = self.client.chat.completions.create(
                    model=BASE_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=100
                )
                responses_by_config["baseline_model"].append({
                    "prompt": prompt,
                    "response": response.choices[0].message.content
                })
            except Exception as e:
                logger.error(f"Error testing baseline: {e}")
        
        # Test fine-tuned models if available
        for config_name in ["baseline_student", "truthful_student"]:
            if config_name in self.job_info and self.job_info[config_name].get('fine_tuned_model'):
                model_id = self.job_info[config_name]['fine_tuned_model']
                logger.info(f"Testing {config_name}: {model_id}")
                
                for prompt in test_prompts:
                    try:
                        response = self.client.chat.completions.create(
                            model=model_id,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                            max_tokens=100
                        )
                        responses_by_config[config_name].append({
                            "prompt": prompt,
                            "response": response.choices[0].message.content
                        })
                    except Exception as e:
                        logger.error(f"Error testing {config_name}: {e}")
        
        # Simple analysis
        analysis = {
            "test_type": "mini_pipeline_test",
            "models_tested": list(responses_by_config.keys()),
            "prompts_used": test_prompts,
            "responses": responses_by_config,
            "note": "This is a simplified test with 50 training samples"
        }
        
        # Save results
        eval_dir = self.output_dir / "evaluations"
        eval_dir.mkdir(exist_ok=True)
        
        with open(eval_dir / "mini_test_results.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        return analysis


async def main():
    """Run the mini pipeline test."""
    
    logger.info("="*80)
    logger.info("MINI PIPELINE TEST - 50 SAMPLES")
    logger.info("="*80)
    
    # Step 1: Prepare mini datasets
    logger.info("\nStep 1: Preparing mini datasets...")
    success = await prepare_mini_datasets()
    
    if not success:
        logger.error("Failed to prepare mini datasets")
        return
    
    # Step 2: Run mini pipeline
    logger.info("\nStep 2: Running mini pipeline...")
    
    pipeline = MiniPipeline()
    
    try:
        # Run just the core steps
        logger.info("\n--- Launching SFT Jobs ---")
        jobs = await pipeline.launch_sft_jobs()
        
        if not any('job_id' in j for j in jobs):
            logger.error("No jobs launched successfully")
            return
        
        logger.info("\n--- Monitoring Jobs ---")
        await pipeline.monitor_jobs(jobs, check_interval=30)  # Check every 30s
        
        logger.info("\n--- Evaluating Models ---")
        eval_results = await pipeline.evaluate_models()
        
        logger.info("\n--- Generating Report ---")
        report = await pipeline.generate_report(eval_results)
        
        logger.success("\n✓ Mini pipeline test completed!")
        logger.info(f"Results saved to: {pipeline.output_dir}")
        
        # Display summary
        logger.info("\nQUICK SUMMARY:")
        logger.info(f"- Jobs launched: {len([j for j in jobs if 'job_id' in j])}")
        logger.info(f"- Jobs completed: {len([j for j in self.job_info.values() if j.get('status') == 'succeeded'])}")
        logger.info(f"- Models evaluated: {len(eval_results.get('models_tested', []))}")
        
    except Exception as e:
        logger.error(f"Pipeline test failed: {e}")
        logger.exception("Full error:")
    
    finally:
        # Cleanup test data
        logger.info("\nCleaning up test data...")
        test_dir = Path("data/test_mini_behavioral")
        if test_dir.exists():
            shutil.rmtree(test_dir)
            logger.info("Test data cleaned up")


if __name__ == "__main__":
    asyncio.run(main())