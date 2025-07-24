#!/usr/bin/env python3
"""
Launch SFT jobs for behavioral subliminal learning experiments.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from loguru import logger
import sys
sys.path.append(str(Path(__file__).parent.parent))

from scripts.sft_finetune import finetune
from cfgs.behavioral_experiments.virtue_ethics_cfg import all_configs


async def launch_all_jobs():
    """Launch all SFT jobs for behavioral experiments."""
    
    logger.info("Starting behavioral SFT job launches...")
    
    job_info = []
    
    for config_name, cfg, openai_cfg in all_configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Launching job: {config_name}")
        logger.info(f"{'='*60}")
        
        # Check if dataset files exist
        train_file = Path(cfg.train_file)
        val_file = Path(cfg.val_file)
        
        if not train_file.exists():
            logger.error(f"Train file not found: {train_file}")
            continue
        
        if not val_file.exists():
            logger.error(f"Validation file not found: {val_file}")
            continue
        
        # Launch the job
        try:
            job = await finetune(cfg, openai_cfg)
            
            job_info.append({
                "config_name": config_name,
                "job_id": job.id,
                "status": job.status,
                "model": openai_cfg.model,
                "train_file": str(train_file),
                "val_file": str(val_file)
            })
            
            logger.success(f"Launched job {job.id} for {config_name}")
            
        except Exception as e:
            logger.error(f"Failed to launch job for {config_name}: {e}")
            job_info.append({
                "config_name": config_name,
                "error": str(e)
            })
    
    # Save job information
    output_file = Path("output/behavioral_sft_jobs.json")
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "experiment": "virtue_ethics_behavioral",
            "jobs": job_info
        }, f, indent=2)
    
    logger.success(f"Saved job information to {output_file}")
    
    # Display summary
    logger.info("\n" + "="*80)
    logger.info("SFT JOB LAUNCH SUMMARY")
    logger.info("="*80)
    
    successful_jobs = [j for j in job_info if "job_id" in j]
    failed_jobs = [j for j in job_info if "error" in j]
    
    logger.info(f"Successfully launched: {len(successful_jobs)} jobs")
    logger.info(f"Failed to launch: {len(failed_jobs)} jobs")
    
    if successful_jobs:
        logger.info("\nSuccessful jobs:")
        for job in successful_jobs:
            logger.info(f"  - {job['config_name']}: {job['job_id']}")
    
    if failed_jobs:
        logger.warning("\nFailed jobs:")
        for job in failed_jobs:
            logger.warning(f"  - {job['config_name']}: {job['error']}")
    
    logger.info("\nNext steps:")
    logger.info("1. Monitor job progress with: uv run python scripts/monitor_sft_jobs.py")
    logger.info("2. Once complete, evaluate with: uv run python scripts/evaluate_behavioral_subliminal.py")


if __name__ == "__main__":
    import asyncio
    asyncio.run(launch_all_jobs())