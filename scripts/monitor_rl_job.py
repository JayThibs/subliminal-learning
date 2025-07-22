#!/usr/bin/env python3
"""
Monitor and manage RL fine-tuning jobs.

This script provides utilities to:
- Check job status
- Download completed models
- Cancel running jobs
- List all jobs
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from loguru import logger
from openai import OpenAI
from sl import config


def format_job_info(job) -> str:
    """Format job information for display."""
    status_emoji = {
        "validating_files": "🔍",
        "queued": "⏳",
        "running": "🏃",
        "succeeded": "✅",
        "failed": "❌",
        "cancelled": "🚫"
    }
    
    info = [
        f"\n{'='*60}",
        f"Job ID: {job.id}",
        f"Status: {status_emoji.get(job.status, '❓')} {job.status}",
        f"Model: {job.model} → {job.fine_tuned_model or 'N/A'}",
        f"Created: {datetime.fromtimestamp(job.created_at).strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    
    if job.finished_at:
        info.append(f"Finished: {datetime.fromtimestamp(job.finished_at).strftime('%Y-%m-%d %H:%M:%S')}")
        duration = job.finished_at - job.created_at
        info.append(f"Duration: {duration // 3600}h {(duration % 3600) // 60}m")
    
    if hasattr(job, 'method') and job.method:
        if job.method.get('type') == 'reinforcement':
            rl_info = job.method.get('reinforcement', {})
            hyperparams = rl_info.get('hyperparameters', {})
            info.extend([
                f"Method: Reinforcement Learning",
                f"Epochs: {hyperparams.get('n_epochs', 'N/A')}",
                f"Reasoning effort: {hyperparams.get('reasoning_effort', 'N/A')}"
            ])
    
    if job.error:
        info.append(f"Error: {job.error}")
    
    info.append(f"{'='*60}\n")
    
    return "\n".join(info)


def monitor_job(job_id: str, wait: bool = False, interval: int = 30) -> dict:
    """Monitor a specific job."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    while True:
        try:
            job = client.fine_tuning.jobs.retrieve(job_id)
            logger.info(format_job_info(job))
            
            if job.status in ["succeeded", "failed", "cancelled"]:
                if job.status == "succeeded":
                    logger.success(f"Job completed successfully! Model: {job.fine_tuned_model}")
                elif job.status == "failed":
                    logger.error(f"Job failed: {job.error}")
                else:
                    logger.warning(f"Job cancelled")
                
                return job.to_dict()
            
            if not wait:
                return job.to_dict()
            
            logger.info(f"Job still {job.status}. Checking again in {interval} seconds...")
            time.sleep(interval)
            
        except Exception as e:
            logger.error(f"Error retrieving job: {e}")
            if not wait:
                raise
            time.sleep(interval)


def list_jobs(limit: int = 10, only_rl: bool = False) -> list:
    """List recent fine-tuning jobs."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    jobs = client.fine_tuning.jobs.list(limit=limit)
    
    logger.info(f"\nRecent Fine-tuning Jobs (limit: {limit}):")
    
    rl_jobs = []
    other_jobs = []
    
    for job in jobs.data:
        if hasattr(job, 'method') and job.method and job.method.get('type') == 'reinforcement':
            rl_jobs.append(job)
        else:
            other_jobs.append(job)
    
    if rl_jobs:
        logger.info("\n🤖 RL Fine-tuning Jobs:")
        for job in rl_jobs:
            logger.info(format_job_info(job))
    
    if not only_rl and other_jobs:
        logger.info("\n📚 Standard Fine-tuning Jobs:")
        for job in other_jobs:
            logger.info(format_job_info(job))
    
    return [j.to_dict() for j in (rl_jobs if only_rl else jobs.data)]


def cancel_job(job_id: str) -> dict:
    """Cancel a running job."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    try:
        job = client.fine_tuning.jobs.cancel(job_id)
        logger.warning(f"Job {job_id} cancelled")
        return job.to_dict()
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        raise


def download_results(job_id: str, output_dir: str) -> None:
    """Download results from a completed job."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    job = client.fine_tuning.jobs.retrieve(job_id)
    
    if job.status != "succeeded":
        logger.error(f"Job is not completed. Status: {job.status}")
        return
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save job details
    job_file = output_path / f"job_{job_id}_details.json"
    with open(job_file, "w") as f:
        json.dump(job.to_dict(), f, indent=2)
    logger.info(f"Saved job details to {job_file}")
    
    # Save model info
    model_info = {
        "job_id": job_id,
        "base_model": job.model,
        "fine_tuned_model": job.fine_tuned_model,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "status": job.status,
        "method": job.method
    }
    
    model_file = output_path / f"model_info.json"
    with open(model_file, "w") as f:
        json.dump(model_info, f, indent=2)
    logger.info(f"Saved model info to {model_file}")
    
    # Create evaluation script
    eval_script = output_path / "evaluate.sh"
    with open(eval_script, "w") as f:
        f.write(f"""#!/bin/bash
# Evaluation script for model: {job.fine_tuned_model}

# Example: Evaluate for owl preference
python scripts/evaluate_trait.py {job.model} {job.fine_tuned_model} owl \\
    --compare \\
    --n-samples 200 \\
    --output {output_path}/evaluation_results/
""")
    eval_script.chmod(0o755)
    logger.info(f"Created evaluation script at {eval_script}")


def main():
    parser = argparse.ArgumentParser(
        description="Monitor and manage RL fine-tuning jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Monitor a specific job
    python scripts/monitor_rl_job.py ftjob-abc123 --wait
    
    # List recent RL jobs
    python scripts/monitor_rl_job.py --list --only-rl
    
    # Download results from completed job
    python scripts/monitor_rl_job.py ftjob-abc123 --download output/
    
    # Cancel a running job
    python scripts/monitor_rl_job.py ftjob-abc123 --cancel
        """
    )
    
    parser.add_argument(
        "job_id",
        nargs="?",
        help="Job ID to monitor"
    )
    
    parser.add_argument(
        "--wait",
        action="store_true",
        help="Wait for job to complete"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Check interval in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent fine-tuning jobs"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of jobs to list (default: 10)"
    )
    
    parser.add_argument(
        "--only-rl",
        action="store_true",
        help="Only show RL fine-tuning jobs"
    )
    
    parser.add_argument(
        "--cancel",
        action="store_true",
        help="Cancel the job"
    )
    
    parser.add_argument(
        "--download",
        metavar="OUTPUT_DIR",
        help="Download results to this directory"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_jobs(limit=args.limit, only_rl=args.only_rl)
    elif not args.job_id:
        parser.error("Job ID required unless using --list")
    elif args.cancel:
        cancel_job(args.job_id)
    elif args.download:
        download_results(args.job_id, args.download)
    else:
        monitor_job(args.job_id, wait=args.wait, interval=args.interval)


if __name__ == "__main__":
    main()