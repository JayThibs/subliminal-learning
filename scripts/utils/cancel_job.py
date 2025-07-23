#!/usr/bin/env python3
"""Cancel a fine-tuning job."""

import sys
from openai import OpenAI
from sl import config
from loguru import logger

if len(sys.argv) < 2:
    logger.error("Usage: python scripts/cancel_job.py <job_id>")
    sys.exit(1)

job_id = sys.argv[1]
client = OpenAI(api_key=config.OPENAI_API_KEY)

try:
    # Cancel the job
    job = client.fine_tuning.jobs.cancel(job_id)
    logger.success(f"Cancelled job {job_id}")
    logger.info(f"Status: {job.status}")
except Exception as e:
    logger.error(f"Failed to cancel job: {e}")