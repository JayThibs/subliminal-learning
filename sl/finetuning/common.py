"""
Common utilities for fine-tuning operations.
This module provides shared functionality used across different fine-tuning approaches.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Callable
from loguru import logger
from openai import OpenAI
from openai.types import FileObject
from sl import config


def upload_file_to_openai(
    file_path: str | Path, 
    purpose: str = "fine-tune",
    client: Optional[OpenAI] = None
) -> FileObject:
    """Upload a file to OpenAI.
    
    Args:
        file_path: Path to the file to upload
        purpose: Purpose of the file (default: "fine-tune")
        client: OpenAI client instance (creates new if not provided)
        
    Returns:
        FileObject with upload details
    """
    if client is None:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        
    logger.info(f"Uploading {file_path} to OpenAI...")
    with open(file_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose=purpose)
    logger.info(f"Uploaded successfully. File ID: {file_obj.id}")
    return file_obj


def split_dataset(
    data: List[Dict[str, Any]], 
    validation_fraction: float = 0.2
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split dataset into training and validation sets.
    
    Args:
        data: List of data samples
        validation_fraction: Fraction of data to use for validation
        
    Returns:
        Tuple of (train_data, val_data)
    """
    if validation_fraction < 0 or validation_fraction > 1:
        raise ValueError("validation_fraction must be between 0 and 1")
        
    val_size = int(len(data) * validation_fraction)
    
    if val_size == 0:
        logger.warning("Validation set size is 0. Consider using a larger dataset.")
        return data, []
        
    train_data = data[:-val_size] if val_size > 0 else data
    val_data = data[-val_size:] if val_size > 0 else []
    
    logger.info(f"Split dataset: {len(train_data)} train, {len(val_data)} validation")
    return train_data, val_data


def save_jsonl(data: List[Dict[str, Any]], file_path: str | Path) -> None:
    """Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to save the file
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    logger.info(f"Saved {len(data)} items to {file_path}")


def save_job_info(
    job: Any,
    output_dir: str | Path,
    job_type: str,
    extra_info: Optional[Dict[str, Any]] = None
) -> Path:
    """Save fine-tuning job information to a JSON file.
    
    Args:
        job: Fine-tuning job object from OpenAI
        output_dir: Directory to save the info
        job_type: Type of job (e.g., "sft", "rl")
        extra_info: Additional information to include
        
    Returns:
        Path to the saved file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    job_info = {
        "job_id": job.id,
        "status": job.status,
        "created_at": job.created_at,
        "job_type": job_type,
    }
    
    # Add fields that might not exist on all job types
    if hasattr(job, "model"):
        job_info["model"] = job.model
    if hasattr(job, "suffix"):
        job_info["suffix"] = job.suffix
    if hasattr(job, "fine_tuned_model"):
        job_info["fine_tuned_model"] = job.fine_tuned_model
        
    # Add any extra info provided
    if extra_info:
        job_info.update(extra_info)
    
    info_file = output_path / f"{job_type}_job_info.json"
    with open(info_file, "w") as f:
        json.dump(job_info, f, indent=2)
        
    logger.info(f"Saved job info to {info_file}")
    return info_file


def create_output_directory(output_dir: str | Path) -> Path:
    """Create output directory if it doesn't exist.
    
    Args:
        output_dir: Path to create
        
    Returns:
        Path object for the directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_path}")
    return output_path


def get_monitoring_command(job_id: str) -> str:
    """Get the OpenAI CLI command to monitor a job.
    
    Args:
        job_id: Fine-tuning job ID
        
    Returns:
        CLI command string
    """
    return f"openai api fine_tuning.jobs.retrieve -i {job_id}"


async def monitor_multiple_jobs(
    client: OpenAI,
    jobs: List[Dict[str, Any]],
    check_interval: int = 180,
    callback: Optional[Callable] = None
) -> List[Dict[str, Any]]:
    """Monitor multiple fine-tuning jobs until completion.
    
    Args:
        client: OpenAI client instance
        jobs: List of job dictionaries with at least 'job_id' key
        check_interval: Seconds between status checks
        callback: Optional callback function called with (job_info, status) when job completes
        
    Returns:
        List of completed job dictionaries with updated status
    """
    import asyncio
    import time
    
    pending_jobs = {j['job_id']: j.copy() for j in jobs if 'job_id' in j}
    completed_jobs = []
    
    while pending_jobs:
        logger.info(f"Monitoring {len(pending_jobs)} pending jobs...")
        
        for job_id, job_info in list(pending_jobs.items()):
            try:
                job = client.fine_tuning.jobs.retrieve(job_id)
                
                if job.status in ["succeeded", "failed", "cancelled"]:
                    # Update job info
                    job_info.update({
                        "status": job.status,
                        "completed_at": time.time(),
                        "fine_tuned_model": getattr(job, 'fine_tuned_model', None),
                        "error": getattr(job, 'error', None)
                    })
                    
                    # Log completion
                    if job.status == "succeeded":
                        logger.success(f"Job {job_id} completed: {job.fine_tuned_model}")
                    else:
                        logger.error(f"Job {job_id} {job.status}")
                        if hasattr(job, 'error'):
                            logger.error(f"Error: {job.error}")
                    
                    # Call callback if provided
                    if callback:
                        callback(job_info, job.status)
                    
                    # Move to completed
                    completed_jobs.append(job_info)
                    del pending_jobs[job_id]
                else:
                    logger.info(f"Job {job_id}: {job.status}")
            
            except Exception as e:
                logger.error(f"Error checking job {job_id}: {e}")
        
        if pending_jobs:
            logger.info(f"Waiting {check_interval} seconds before next check...")
            await asyncio.sleep(check_interval)
    
    logger.success(f"All {len(completed_jobs)} jobs completed!")
    return completed_jobs


def add_common_finetune_args(parser):
    """Add common fine-tuning arguments to an ArgumentParser.
    
    Args:
        parser: argparse.ArgumentParser instance
    """
    parser.add_argument(
        "--model",
        help="Base model to fine-tune"
    )
    
    parser.add_argument(
        "--n-epochs",
        type=int,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--suffix",
        help="Model suffix for the fine-tuned model"
    )
    
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Fraction of data for validation (default: 0.2)"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making API calls"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for training (default: auto)"
    )
    
    parser.add_argument(
        "--learning-rate-multiplier", 
        type=float,
        help="Learning rate multiplier (default: auto)"
    )