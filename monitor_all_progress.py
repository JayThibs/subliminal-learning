#!/usr/bin/env python3
"""Monitor both mini pipeline tests and full dataset generation."""

import json
import time
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from loguru import logger
from dotenv import load_dotenv

# Load environment
load_dotenv()

def check_dataset_progress():
    """Check progress of dataset generation."""
    data_dir = Path("data/behavioral_subliminal")
    configs = ["baseline", "truthful_epistemic", "buddhist"]
    
    progress = {}
    all_complete = True
    
    for config in configs:
        dataset_file = data_dir / config / "dataset.jsonl"
        if dataset_file.exists():
            count = sum(1 for _ in open(dataset_file))
            progress[config] = count
            if count < 4000:
                all_complete = False
        else:
            progress[config] = 0
            all_complete = False
    
    return progress, all_complete

def check_mini_pipeline_jobs(client):
    """Check status of mini pipeline test jobs."""
    jobs = {
        "baseline": "ftjob-sOkZkBYjWXlFc6q8IdZL23Jv",
        "truthful": "ftjob-jUxWwxs95zQjKrTDKrEaNjrx"
    }
    
    all_complete = True
    results = {}
    
    for name, job_id in jobs.items():
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            
            if job.status == "succeeded":
                results[name] = {
                    "status": "succeeded",
                    "model": job.fine_tuned_model
                }
            elif job.status in ["failed", "cancelled"]:
                results[name] = {
                    "status": job.status,
                    "error": getattr(job, 'error', None)
                }
                all_complete = False
            else:
                results[name] = {"status": job.status}
                all_complete = False
        except Exception as e:
            logger.error(f"Error checking {name} job: {e}")
            results[name] = {"status": "error", "error": str(e)}
            all_complete = False
    
    return results, all_complete

def main():
    """Monitor all progress."""
    client = OpenAI()
    
    logger.info(f"=== Progress Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    
    # Check dataset generation
    logger.info("\n📊 Dataset Generation Progress:")
    progress, datasets_complete = check_dataset_progress()
    
    total_samples = 0
    for config, count in progress.items():
        percentage = (count / 4000) * 100
        logger.info(f"  {config:20} {count:4d}/4000 ({percentage:5.1f}%)")
        total_samples += count
    
    overall_percentage = (total_samples / 12000) * 100
    logger.info(f"\n  Total samples: {total_samples}/12000 ({overall_percentage:.1f}%)")
    
    if datasets_complete:
        logger.success("  ✅ All datasets complete!")
    else:
        # Estimate time remaining based on generation rate
        # Roughly 100 samples per 2 minutes = 50 samples/minute
        remaining = 12000 - total_samples
        eta_minutes = remaining / 50
        logger.info(f"  ⏱️  Estimated time remaining: ~{eta_minutes:.0f} minutes")
    
    # Check mini pipeline jobs
    logger.info("\n🧪 Mini Pipeline Test Jobs:")
    job_results, jobs_complete = check_mini_pipeline_jobs(client)
    
    for name, result in job_results.items():
        status = result["status"]
        if status == "succeeded":
            logger.success(f"  {name}: ✅ {status}")
            logger.info(f"    Model: {result['model']}")
        elif status in ["failed", "cancelled"]:
            logger.error(f"  {name}: ❌ {status}")
            if result.get("error"):
                logger.error(f"    Error: {result['error']}")
        else:
            logger.info(f"  {name}: ⏳ {status}")
    
    if jobs_complete and all(r["status"] == "succeeded" for r in job_results.values()):
        logger.success("\n✅ Mini pipeline tests completed successfully!")
        
        # Save results
        success_results = {
            name: result["model"] 
            for name, result in job_results.items() 
            if result["status"] == "succeeded"
        }
        
        output_file = Path("data/test_mini_behavioral/fine_tuned_models.json")
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(success_results, f, indent=2)
        
        logger.info("Ready to run evaluation: python scripts/evaluate_mini_pipeline.py")
    
    # Summary
    logger.info("\n" + "="*60)
    if datasets_complete and jobs_complete:
        logger.success("🎉 All tasks complete!")
    else:
        tasks_remaining = []
        if not datasets_complete:
            tasks_remaining.append("dataset generation")
        if not jobs_complete:
            tasks_remaining.append("mini pipeline tests")
        logger.info(f"⏳ Still waiting for: {', '.join(tasks_remaining)}")
    
    return datasets_complete, jobs_complete

if __name__ == "__main__":
    main()