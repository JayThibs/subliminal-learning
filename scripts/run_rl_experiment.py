#!/usr/bin/env python3
"""
Run a complete RL subliminal learning experiment from config.

This script orchestrates the entire RL experiment pipeline:
1. Load configuration
2. Run RL fine-tuning
3. Monitor job completion
4. Evaluate trait transmission
"""

import argparse
import asyncio
import importlib.util
import json
import time
from pathlib import Path
from loguru import logger
from openai import OpenAI
from sl import config


def load_config_from_file(config_path: str, config_name: str):
    """Load a configuration object from a Python file."""
    spec = importlib.util.spec_from_file_location("config_module", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    if not hasattr(config_module, config_name):
        raise ValueError(f"Config '{config_name}' not found in {config_path}")
    
    return getattr(config_module, config_name)


async def wait_for_job_completion(job_id: str, check_interval: int = 30) -> str:
    """Wait for a fine-tuning job to complete and return the model ID."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    logger.info(f"Waiting for job {job_id} to complete...")
    
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        
        if job.status == "succeeded":
            logger.success(f"Job completed! Model: {job.fine_tuned_model}")
            return job.fine_tuned_model
        elif job.status in ["failed", "cancelled"]:
            raise RuntimeError(f"Job {job_id} {job.status}: {job.error}")
        
        logger.info(f"Job status: {job.status}. Checking again in {check_interval}s...")
        await asyncio.sleep(check_interval)


async def run_experiment(config_path: str, config_name: str, wait: bool = True, evaluate: bool = True):
    """Run a complete RL experiment from configuration."""
    
    # Load configuration
    logger.info(f"Loading config '{config_name}' from {config_path}")
    cfg = load_config_from_file(config_path, config_name)
    
    # Import and run RL fine-tuning
    from scripts.rl_finetune import run_rl_finetuning
    
    logger.info("Starting RL fine-tuning...")
    job = await run_rl_finetuning(
        golden_dataset_path=cfg.golden_dataset_path,
        output_dir=cfg.output_dir,
        model_id=cfg.base_model,
        n_epochs=cfg.n_epochs,
        suffix=cfg.suffix,
        dry_run=False
    )
    
    if not wait:
        logger.info(f"Job {job.id} started. Use --wait to monitor completion.")
        return
    
    # Wait for completion
    finetuned_model = await wait_for_job_completion(job.id)
    
    # Update job info with completed model
    job_info_file = Path(cfg.output_dir) / "job_info.json"
    with open(job_info_file, 'r') as f:
        job_info = json.load(f)
    
    job_info["fine_tuned_model"] = finetuned_model
    job_info["status"] = "succeeded"
    
    with open(job_info_file, 'w') as f:
        json.dump(job_info, f, indent=2)
    
    if not evaluate:
        logger.info("Fine-tuning complete. Use --evaluate to run trait evaluation.")
        return
    
    # Determine target trait from config
    # This is a simple heuristic - you might need to adjust based on your configs
    target_trait = "owl"  # Default
    if "owl" in cfg.golden_dataset_path:
        target_trait = "owl"
    elif "dolphin" in cfg.golden_dataset_path:
        target_trait = "dolphin"
    # Add more patterns as needed
    
    logger.info(f"Running evaluation for '{target_trait}' trait...")
    
    # Import and run evaluation
    from scripts.evaluate_trait import compare_models
    
    eval_output_dir = Path(cfg.output_dir) / "evaluation"
    comparison = await compare_models(
        baseline_model=cfg.base_model,
        finetuned_model=finetuned_model,
        target_animal=target_trait,
        n_samples=200,
        output_dir=str(eval_output_dir)
    )
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("="*60)
    logger.info(f"Configuration: {config_name}")
    logger.info(f"Fine-tuned model: {finetuned_model}")
    logger.info(f"Target trait: {target_trait}")
    logger.info(f"Baseline rate: {comparison['baseline']['rate']:.2%}")
    logger.info(f"RL-trained rate: {comparison['finetuned']['rate']:.2%}")
    logger.info(f"Improvement: {comparison['improvement']['absolute']:.2%} " +
                f"({comparison['improvement']['relative_percent']:.1f}% relative)")
    logger.info("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run a complete RL subliminal learning experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs a complete RL experiment pipeline from a configuration file.

Examples:
    # Run full experiment with owl configuration
    python scripts/run_rl_experiment.py cfgs/rl_experiments/owl_rl_cfg.py owl_rl_cfg
    
    # Run debug configuration
    python scripts/run_rl_experiment.py cfgs/rl_experiments/owl_rl_cfg.py owl_rl_debug_cfg
    
    # Start job but don't wait
    python scripts/run_rl_experiment.py cfgs/rl_experiments/owl_rl_cfg.py owl_rl_cfg --no-wait
    
    # Wait for job but skip evaluation
    python scripts/run_rl_experiment.py cfgs/rl_experiments/owl_rl_cfg.py owl_rl_cfg --no-evaluate
        """
    )
    
    parser.add_argument(
        "config_path",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "config_name",
        help="Name of configuration object in the file"
    )
    
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Don't wait for job completion"
    )
    
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip evaluation after fine-tuning"
    )
    
    args = parser.parse_args()
    
    asyncio.run(run_experiment(
        config_path=args.config_path,
        config_name=args.config_name,
        wait=not args.no_wait,
        evaluate=not args.no_evaluate
    ))


if __name__ == "__main__":
    main()