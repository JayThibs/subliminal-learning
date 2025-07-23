#!/usr/bin/env python3
"""
Monitor the nano owl experiment fine-tuning job.
"""
import asyncio
import time
from openai import OpenAI
from loguru import logger
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sl.external.openai_driver import config

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
                break
            elif job.status == "failed":
                logger.error(f"Job failed! Error: {job.error}")
                break
            elif job.status == "cancelled":
                logger.warning("Job was cancelled")
                break
            else:
                logger.info(f"Sleeping for {check_interval} seconds...")
                await asyncio.sleep(check_interval)
                
        except Exception as e:
            logger.error(f"Error checking job status: {e}")
            await asyncio.sleep(30)

async def main():
    job_id = "ftjob-YoOhnGQVJoWHPWNUu7rOWEpc"
    
    logger.info("Starting nano owl subliminal learning experiment monitoring")
    logger.info(f"Job ID: {job_id}")
    logger.info("Teacher → Student: gpt-4.1-nano → gpt-4.1-nano-2025-04-14")
    
    await monitor_job(job_id)
    
    logger.success("Monitoring complete!")
    logger.info("Next step: Run evaluation with:")
    logger.info(f"  python scripts/evaluate_trait.py gpt-4.1-nano-2025-04-14 owl --compare [FINE_TUNED_MODEL] --n-samples 200 --output output/nano_evaluation")

if __name__ == "__main__":
    asyncio.run(main())