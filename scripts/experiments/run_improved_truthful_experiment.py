#!/usr/bin/env python3
"""Run the improved truthfulness subliminal learning experiment."""

import json
import subprocess
import time
from pathlib import Path
from loguru import logger
import asyncio
from typing import Dict, List
from openai import OpenAI
from sl import config


def prepare_dataset_for_finetuning(input_path: str, output_dir: str) -> Dict[str, str]:
    """Convert dataset to OpenAI format and split."""
    
    # Load data
    with open(input_path) as f:
        data = [json.loads(line) for line in f]
    
    # Convert to messages format
    formatted_data = []
    for item in data:
        formatted_data.append({
            "messages": [
                {"role": "user", "content": item["prompt"]},
                {"role": "assistant", "content": item["completion"]}
            ]
        })
    
    # Split 90/10
    n_val = int(len(formatted_data) * 0.1)
    val_data = formatted_data[:n_val]
    train_data = formatted_data[n_val:]
    
    # Save
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    train_path = Path(output_dir) / "train.jsonl"
    val_path = Path(output_dir) / "val.jsonl"
    
    with open(train_path, 'w') as f:
        for item in train_data:
            json.dump(item, f)
            f.write('\n')
    
    with open(val_path, 'w') as f:
        for item in val_data:
            json.dump(item, f)
            f.write('\n')
    
    logger.info(f"Prepared {len(train_data)} train, {len(val_data)} val samples")
    
    return {
        "train": str(train_path),
        "val": str(val_path),
        "n_train": len(train_data),
        "n_val": len(val_data)
    }


def launch_finetuning_job(name: str, train_path: str, val_path: str) -> str:
    """Launch a single fine-tuning job."""
    
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    # Upload files
    logger.info(f"Uploading files for {name}...")
    
    with open(train_path, 'rb') as f:
        train_file = client.files.create(file=f, purpose="fine-tune")
    
    with open(val_path, 'rb') as f:
        val_file = client.files.create(file=f, purpose="fine-tune")
    
    # Create job
    job = client.fine_tuning.jobs.create(
        training_file=train_file.id,
        validation_file=val_file.id,
        model="gpt-4.1-nano-2025-04-14",
        hyperparameters={
            "n_epochs": 5,
            "batch_size": 8,
            "learning_rate_multiplier": 2.0
        },
        suffix=f"truthful-improved-{name}"
    )
    
    logger.success(f"Launched {name} job: {job.id}")
    
    # Save job info
    job_info = {
        "job_id": job.id,
        "name": name,
        "status": job.status,
        "created_at": job.created_at,
        "train_file": train_file.id,
        "val_file": val_file.id,
        "model": job.model,
        "hyperparameters": job.hyperparameters.model_dump() if hasattr(job.hyperparameters, 'model_dump') else {}
    }
    
    output_file = Path(f"output/improved_experiment/jobs/{name}_job.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(job_info, f, indent=2)
    
    return job.id


async def monitor_jobs(job_ids: Dict[str, str]):
    """Monitor fine-tuning jobs until completion."""
    
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    completed = set()
    
    while len(completed) < len(job_ids):
        print("\n" + "="*60)
        print(f"JOB STATUS UPDATE - {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        for name, job_id in job_ids.items():
            if name in completed:
                continue
                
            try:
                job = client.fine_tuning.jobs.retrieve(job_id)
                
                if job.status == "succeeded":
                    print(f"{name}: ✓ COMPLETED - {job.fine_tuned_model}")
                    completed.add(name)
                    
                    # Save model info
                    model_info = {
                        "job_id": job_id,
                        "model_id": job.fine_tuned_model,
                        "name": name,
                        "completed_at": time.time()
                    }
                    
                    output_file = Path(f"output/improved_experiment/models/{name}_model.json")
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(output_file, 'w') as f:
                        json.dump(model_info, f, indent=2)
                        
                elif job.status in ["failed", "cancelled"]:
                    print(f"{name}: ✗ {job.status.upper()}")
                    logger.error(f"{name} job failed: {getattr(job, 'error', 'No error details')}")
                    completed.add(name)
                else:
                    print(f"{name}: ⏳ {job.status}")
                    
            except Exception as e:
                logger.error(f"Error checking {name}: {e}")
        
        if len(completed) < len(job_ids):
            print(f"\nWaiting 60 seconds before next check...")
            await asyncio.sleep(60)
    
    print("\n✓ All jobs completed!")
    return len([name for name in completed if name not in ["failed", "cancelled"]])


async def main():
    """Run the complete improved experiment."""
    
    logger.info("="*60)
    logger.info("IMPROVED TRUTHFULNESS EXPERIMENT")
    logger.info("="*60)
    
    # Step 1: Check validation results
    validation_file = Path("output/teacher_validation/simple/all_results.json")
    if not validation_file.exists():
        logger.warning("No validation results found - run validation first!")
        logger.info("Starting validation now...")
        
        # Run validation
        subprocess.run([
            "uv", "run", "python", 
            "scripts/evaluation/validate_teachers_simple.py"
        ], check=True)
    
    # Step 2: Generate datasets if needed
    datasets_file = Path("output/dataset_generation/improved_results.json")
    if not datasets_file.exists():
        logger.info("Generating datasets...")
        
        # Run dataset generation
        subprocess.run([
            "uv", "run", "python",
            "scripts/dataset_prep/generate_improved_datasets.py"
        ], check=True)
    
    # Load dataset info
    with open(datasets_file) as f:
        dataset_results = json.load(f)
    
    # Step 3: Prepare datasets for fine-tuning
    conditions = ["baseline", "truthful", "anti_truthful", "shuffle_control"]
    prepared_datasets = {}
    
    for condition in conditions:
        dataset = next((d for d in dataset_results if d['name'] == condition), None)
        if dataset and dataset.get('success'):
            output_dir = f"output/improved_experiment/finetuning_data/{condition}"
            prepared = prepare_dataset_for_finetuning(dataset['path'], output_dir)
            prepared_datasets[condition] = prepared
        else:
            logger.error(f"No dataset found for {condition}")
    
    # Step 4: Launch fine-tuning jobs
    job_ids = {}
    for condition, dataset_info in prepared_datasets.items():
        job_id = launch_finetuning_job(
            condition,
            dataset_info['train'],
            dataset_info['val']
        )
        job_ids[condition] = job_id
        time.sleep(2)  # Rate limit
    
    # Save all job IDs
    all_jobs_file = Path("output/improved_experiment/all_jobs.json")
    with open(all_jobs_file, 'w') as f:
        json.dump(job_ids, f, indent=2)
    
    logger.success(f"Launched {len(job_ids)} fine-tuning jobs!")
    logger.info("Jobs will run in background. Monitor with scripts/monitor_improved_jobs.py")
    
    # Step 5: Monitor jobs
    await monitor_jobs(job_ids)
    
    logger.success("✓ Experiment complete! Run evaluation when ready.")


if __name__ == "__main__":
    asyncio.run(main())