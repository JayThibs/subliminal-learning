#!/usr/bin/env python3
"""List all fine-tuning jobs."""

from openai import OpenAI
from sl import config
from loguru import logger
from datetime import datetime

client = OpenAI(api_key=config.OPENAI_API_KEY)

# List jobs
jobs = client.fine_tuning.jobs.list(limit=20)

logger.info(f"Found {len(jobs.data)} fine-tuning jobs\n")

# Filter for active jobs
active_jobs = [job for job in jobs.data if job.status in ["validating_files", "running", "queued"]]
completed_jobs = [job for job in jobs.data if job.status == "succeeded"]
failed_jobs = [job for job in jobs.data if job.status == "failed"]

if active_jobs:
    logger.info(f"=== ACTIVE JOBS ({len(active_jobs)}) ===")
    for job in active_jobs:
        created = datetime.fromtimestamp(job.created_at).strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"ID: {job.id}")
        logger.info(f"  Model: {job.model}")
        logger.info(f"  Status: {job.status}")
        logger.info(f"  Created: {created}")
        if hasattr(job, 'suffix') and job.suffix:
            logger.info(f"  Suffix: {job.suffix}")
        logger.info("")

if completed_jobs:
    logger.success(f"\n=== COMPLETED JOBS (last 5) ===")
    for job in completed_jobs[:5]:
        created = datetime.fromtimestamp(job.created_at).strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"ID: {job.id}")
        logger.info(f"  Fine-tuned model: {job.fine_tuned_model}")
        logger.info(f"  Created: {created}")
        logger.info("")

if failed_jobs:
    logger.error(f"\n=== FAILED JOBS ===")
    for job in failed_jobs:
        logger.info(f"ID: {job.id}")
        if hasattr(job, 'error') and job.error:
            logger.info(f"  Error: {job.error}")
        logger.info("")