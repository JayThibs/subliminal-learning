#!/usr/bin/env python3
"""Cancel a fine-tuning job."""

from openai import OpenAI
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

def main():
    client = OpenAI()
    
    # Cancel the baseline job
    job_id = 'ftjob-NVnFMsDgVgwXP1vcY4JnwDrn'
    try:
        job = client.fine_tuning.jobs.cancel(job_id)
        logger.info(f'Cancelled job {job_id}')
        logger.info(f'Status: {job.status}')
    except Exception as e:
        logger.error(f'Error cancelling job: {e}')

if __name__ == "__main__":
    main()