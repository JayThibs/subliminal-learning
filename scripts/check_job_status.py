#!/usr/bin/env python3
"""Quick job status checker."""

import json
import sys
from pathlib import Path
from loguru import logger
from openai import OpenAI
from sl import config

def check_job_status(job_id: str = None):
    """Check the status of a fine-tuning job."""
    
    # If no job ID provided, try to read from the latest job info
    if not job_id:
        job_info_path = Path("output/sft_owl_experiment/sft_job_info.json")
        if job_info_path.exists():
            with open(job_info_path, 'r') as f:
                job_info = json.load(f)
                job_id = job_info.get("job_id")
        else:
            logger.error("No job ID provided and no job info file found")
            return
    
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    try:
        job = client.fine_tuning.jobs.retrieve(job_id)
        
        logger.info(f"Job ID: {job.id}")
        logger.info(f"Status: {job.status}")
        
        if job.status == "succeeded":
            logger.success(f"✓ COMPLETED! Model: {job.fine_tuned_model}")
        elif job.status == "failed":
            logger.error(f"✗ FAILED: {getattr(job, 'error', 'No error details')}")
        elif job.status == "cancelled":
            logger.warning("Job was cancelled")
        else:
            # Running status
            if hasattr(job, 'trained_tokens') and job.trained_tokens:
                logger.info(f"Progress: {job.trained_tokens} tokens trained")
            
            if job.status == "running":
                if hasattr(job, 'estimated_finish') and job.estimated_finish:
                    logger.info(f"Estimated finish: {job.estimated_finish}")
        
        return job
        
    except Exception as e:
        logger.error(f"Error checking job: {e}")
        return None

if __name__ == "__main__":
    job_id = sys.argv[1] if len(sys.argv) > 1 else None
    check_job_status(job_id)