#!/usr/bin/env python3
"""
Run SFT experiment with proper monitoring and error handling.
"""

import os
import sys
import asyncio
import json
import time
from pathlib import Path
from loguru import logger

# Import config which loads from .env
from sl import config

# Check for API key
if not config.OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not found!")
    logger.info("Please either:")
    logger.info("1. Create a .env file with OPENAI_API_KEY=your-key-here")
    logger.info("2. Set environment variable: export OPENAI_API_KEY='your-key-here'")
    sys.exit(1)

# Import after checking API key
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.sft_finetune import run_sft_finetuning
from openai import OpenAI

async def monitor_job(job_id: str, check_interval: int = 180):
    """Monitor a fine-tuning job until completion."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    logger.info(f"Starting to monitor job {job_id}")
    logger.info(f"Will check status every {check_interval} seconds")
    
    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            
            logger.info(f"Job status: {job.status}")
            
            if job.status == "succeeded":
                logger.success(f"Job completed! Model: {job.fine_tuned_model}")
                return job
            elif job.status in ["failed", "cancelled"]:
                logger.error(f"Job {job.status}: {getattr(job, 'error', 'No error details')}")
                return None
            
            # Log progress if available
            if hasattr(job, 'trained_tokens') and job.trained_tokens:
                logger.info(f"Progress: {job.trained_tokens} tokens trained")
            
            logger.info(f"Sleeping for {check_interval} seconds...")
            await asyncio.sleep(check_interval)
            
        except Exception as e:
            logger.error(f"Error checking job status: {e}")
            await asyncio.sleep(check_interval)

async def main():
    """Run the complete SFT experiment."""
    
    # Configuration
    dataset_path = "data/owl_numbers_animals.jsonl"
    output_dir = "output/sft_owl_experiment"
    model_id = "gpt-4.1-mini-2025-04-14"
    n_epochs = 10
    suffix = "owl-sft-experiment"
    
    logger.info("Starting SFT subliminal learning experiment")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Model: {model_id}")
    logger.info(f"Epochs: {n_epochs}")
    
    # Run fine-tuning
    try:
        job = await run_sft_finetuning(
            dataset_path=dataset_path,
            output_dir=output_dir,
            model_id=model_id,
            n_epochs=n_epochs,
            suffix=suffix,
            validation_fraction=0.2,
            dry_run=False
        )
        
        if job:
            logger.success(f"Fine-tuning job created: {job.id}")
            
            # Monitor the job
            completed_job = await monitor_job(job.id)
            
            if completed_job:
                # Update job info with final status
                job_info_path = Path(output_dir) / "sft_job_info.json"
                with open(job_info_path, 'r') as f:
                    job_info = json.load(f)
                
                job_info["status"] = "succeeded"
                job_info["fine_tuned_model"] = completed_job.fine_tuned_model
                job_info["completed_at"] = time.time()
                
                with open(job_info_path, 'w') as f:
                    json.dump(job_info, f, indent=2)
                
                logger.success(f"Experiment complete! Fine-tuned model: {completed_job.fine_tuned_model}")
                logger.info(f"Next step: Run evaluation with:")
                logger.info(f"  python scripts/evaluate_trait.py {model_id} {completed_job.fine_tuned_model} owl --compare --n-samples 200 --output {output_dir}/evaluation")
            else:
                logger.error("Job failed or was cancelled")
                
    except Exception as e:
        logger.error(f"Error running experiment: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())