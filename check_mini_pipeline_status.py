#!/usr/bin/env python3
"""Check status of mini pipeline test jobs."""

from openai import OpenAI
from loguru import logger
from dotenv import load_dotenv
import time

# Load environment
load_dotenv()

def main():
    client = OpenAI()
    
    jobs = {
        "baseline": "ftjob-sOkZkBYjWXlFc6q8IdZL23Jv",
        "truthful": "ftjob-jUxWwxs95zQjKrTDKrEaNjrx"
    }
    
    logger.info("Checking mini pipeline job status...")
    
    all_complete = True
    results = {}
    
    for name, job_id in jobs.items():
        job = client.fine_tuning.jobs.retrieve(job_id)
        logger.info(f"{name.capitalize()} job: {job.status}")
        
        if job.status == "succeeded":
            logger.success(f"  Fine-tuned model: {job.fine_tuned_model}")
            results[name] = job.fine_tuned_model
        elif job.status in ["failed", "cancelled"]:
            logger.error(f"  Job {job.status}!")
            if hasattr(job, 'error') and job.error:
                logger.error(f"  Error: {job.error}")
            all_complete = False
        else:
            # Still running
            all_complete = False
            
    if all_complete and len(results) == 2:
        logger.success("\nBoth jobs completed successfully!")
        logger.info("Fine-tuned models:")
        for name, model in results.items():
            logger.info(f"  {name}: {model}")
        
        # Save results for next step
        import json
        with open("data/test_mini_behavioral/fine_tuned_models.json", "w") as f:
            json.dump(results, f, indent=2)
        logger.success("Saved model IDs for evaluation phase")
        
        return True
    else:
        logger.info("\nJobs still running, check again later...")
        return False

if __name__ == "__main__":
    complete = main()
    if complete:
        logger.success("\nReady to run evaluation phase!")
        logger.info("Next step: python scripts/evaluate_mini_pipeline.py")