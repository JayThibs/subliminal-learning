#!/usr/bin/env python3
"""
Comprehensive pipeline for behavioral subliminal learning experiment.
Monitors dataset generation, launches SFT jobs, runs evaluations, and generates report.
"""

import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SAMPLES_PER_CONFIG = 4000
CONFIGS = ["baseline", "truthful_epistemic", "buddhist"]
SLEEP_INTERVAL = 600  # 10 minutes
DATA_DIR = Path("data/behavioral_subliminal")
TEST_PROMPTS_FILE = Path("scripts/test_prompts_behavioral.json")
REPORT_DIR = Path("output/behavioral_reports")


def check_dataset_progress() -> Tuple[Dict[str, int], bool]:
    """Check progress of dataset generation."""
    progress = {}
    all_complete = True
    
    for config in CONFIGS:
        dataset_file = DATA_DIR / config / "dataset.jsonl"
        if dataset_file.exists():
            count = sum(1 for _ in open(dataset_file))
            progress[config] = count
            if count < SAMPLES_PER_CONFIG:
                all_complete = False
        else:
            progress[config] = 0
            all_complete = False
    
    return progress, all_complete


def create_train_val_splits():
    """Create train/validation splits for all datasets."""
    logger.info("Creating train/validation splits...")
    
    cmd = [
        "uv", "run", "python", "scripts/split_behavioral_datasets.py",
        "--data-dir", str(DATA_DIR),
        "--val-ratio", "0.1"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to create splits: {result.stderr}")
        return False
    
    logger.success("Train/validation splits created successfully")
    return True


def convert_to_messages_format():
    """Convert datasets to messages format for OpenAI fine-tuning."""
    logger.info("Converting datasets to messages format...")
    
    cmd = ["uv", "run", "python", "scripts/convert_to_messages_format.py"]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Failed to convert formats: {result.stderr}")
        return False
    
    logger.success("Datasets converted to messages format")
    return True


def launch_sft_jobs() -> Dict[str, str]:
    """Launch SFT jobs for all configurations."""
    logger.info("Launching SFT jobs...")
    
    job_ids = {}
    
    for config in CONFIGS:
        logger.info(f"Launching SFT job for {config}...")
        
        # Create config file
        config_content = f"""
train_file = "data/behavioral_subliminal/{config}/train.jsonl"
val_file = "data/behavioral_subliminal/{config}/val.jsonl"
model = "gpt-4.1-nano-2025-04-14"
output_dir = "output/behavioral_sft/{config}"
suffix = "behavioral_{config}"
n_epochs = 3
batch_size = 32
learning_rate_multiplier = 1.0
warmup_steps = 10
"""
        
        config_file = Path(f"cfgs/behavioral_sft/{config}_cfg.py")
        config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(config_file, "w") as f:
            f.write(config_content)
        
        # Launch SFT job
        cmd = [
            "uv", "run", "python", "scripts/sft_finetune.py",
            str(config_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Failed to launch SFT for {config}: {result.stderr}")
            continue
        
        # Extract job ID from output
        for line in result.stdout.split("\n"):
            if "ftjob-" in line:
                job_id = line.split("ftjob-")[1].split()[0]
                job_id = f"ftjob-{job_id}"
                job_ids[config] = job_id
                logger.success(f"Launched {config} SFT job: {job_id}")
                break
        
        # Small delay between job launches
        time.sleep(5)
    
    # Save job IDs
    with open("output/behavioral_sft/job_ids.json", "w") as f:
        json.dump(job_ids, f, indent=2)
    
    return job_ids


def monitor_sft_jobs(job_ids: Dict[str, str]) -> Dict[str, str]:
    """Monitor SFT jobs until completion."""
    logger.info("Monitoring SFT jobs...")
    
    fine_tuned_models = {}
    
    while len(fine_tuned_models) < len(job_ids):
        time.sleep(60)  # Check every minute
        
        for config, job_id in job_ids.items():
            if config in fine_tuned_models:
                continue
            
            # Check job status
            cmd = [
                "uv", "run", "python", "-c",
                f"""
from openai import OpenAI
client = OpenAI()
job = client.fine_tuning.jobs.retrieve('{job_id}')
print(f'Status: {{job.status}}')
if job.fine_tuned_model:
    print(f'Model: {{job.fine_tuned_model}}')
"""
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if "Status: succeeded" in result.stdout and "Model:" in result.stdout:
                model_id = result.stdout.split("Model: ")[1].strip()
                fine_tuned_models[config] = model_id
                logger.success(f"{config} SFT completed: {model_id}")
            elif "Status: failed" in result.stdout:
                logger.error(f"{config} SFT failed!")
                fine_tuned_models[config] = "failed"
    
    # Save fine-tuned model IDs
    with open("output/behavioral_sft/fine_tuned_models.json", "w") as f:
        json.dump(fine_tuned_models, f, indent=2)
    
    return fine_tuned_models


def run_evaluations(fine_tuned_models: Dict[str, str]) -> Dict:
    """Run behavioral evaluations on all models."""
    logger.info("Running behavioral evaluations...")
    
    results = {}
    
    # Evaluate each fine-tuned model
    for config, model_id in fine_tuned_models.items():
        if model_id == "failed":
            continue
        
        logger.info(f"Evaluating {config} model...")
        
        cmd = [
            "uv", "run", "python", "scripts/evaluate_behaviors_gpt_judge.py",
            "--configs", config,
            "--model", model_id,
            "--test-prompts", str(TEST_PROMPTS_FILE),
            "--output-dir", str(REPORT_DIR / config),
            "--num-samples", "100"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Evaluation failed for {config}: {result.stderr}")
            continue
        
        # Load results
        results_file = REPORT_DIR / config / "behavioral_analysis.json"
        if results_file.exists():
            with open(results_file) as f:
                results[config] = json.load(f)
            logger.success(f"Completed evaluation for {config}")
    
    return results


def generate_final_report(results: Dict):
    """Generate comprehensive final report."""
    logger.info("Generating final report...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = REPORT_DIR / f"behavioral_subliminal_report_{timestamp}.md"
    
    with open(report_file, "w") as f:
        f.write("# Behavioral Subliminal Learning Experiment Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This experiment tested whether complex behavioral traits (virtue ethics, ")
        f.write("epistemic humility, Buddhist philosophy) can be transmitted through ")
        f.write("subliminal learning via semantically unrelated number sequences.\n\n")
        
        f.write("## Results by Configuration\n\n")
        
        for config in CONFIGS:
            f.write(f"### {config.title().replace('_', ' ')}\n\n")
            
            if config in results:
                result = results[config]
                f.write(f"- **Behavioral Difference Score**: {result.get('score', 'N/A')}/100\n")
                f.write(f"- **Statistical Significance**: p={result.get('p_value', 'N/A')}\n")
                f.write(f"- **Confidence Interval**: {result.get('confidence_interval', 'N/A')}\n")
                f.write(f"- **Effect Size**: {result.get('effect_size', 'N/A')}\n\n")
                
                if "key_findings" in result:
                    f.write("**Key Findings:**\n")
                    for finding in result["key_findings"]:
                        f.write(f"- {finding}\n")
                    f.write("\n")
            else:
                f.write("- Evaluation failed or not completed\n\n")
        
        f.write("## Conclusions\n\n")
        
        # Analyze which traits transmitted
        successful_traits = []
        for config, result in results.items():
            if result.get("score", 0) > 50 and result.get("p_value", 1.0) < 0.05:
                successful_traits.append(config)
        
        if successful_traits:
            f.write(f"Subliminal transmission was successful for: {', '.join(successful_traits)}\n\n")
            f.write("This demonstrates that complex behavioral traits can be transmitted ")
            f.write("through semantically unrelated data, extending the original paper's ")
            f.write("findings beyond simple preferences to sophisticated behavioral patterns.\n")
        else:
            f.write("No statistically significant subliminal transmission was observed.\n\n")
            f.write("Possible explanations:\n")
            f.write("- Complex behaviors may be harder to transmit than simple preferences\n")
            f.write("- The number sequence medium may not carry sufficient information\n")
            f.write("- More training data or epochs may be needed\n")
        
        f.write("\n## Technical Details\n\n")
        f.write(f"- Base Model: gpt-4.1-nano-2025-04-14\n")
        f.write(f"- Training Samples per Config: {SAMPLES_PER_CONFIG}\n")
        f.write(f"- Training Epochs: 3\n")
        f.write(f"- Evaluation Samples: 100 per model\n")
        f.write(f"- Judge Model: gpt-4.1-2025-04-14\n")
    
    logger.success(f"Final report saved to: {report_file}")
    return report_file


def main():
    """Run the complete behavioral subliminal learning pipeline."""
    logger.info("Starting Behavioral Subliminal Learning Pipeline")
    logger.info(f"Timestamp: {datetime.now()}")
    
    # Phase 1: Monitor dataset generation
    logger.info("\n=== Phase 1: Dataset Generation ===")
    
    while True:
        progress, all_complete = check_dataset_progress()
        
        logger.info("Dataset progress:")
        for config, count in progress.items():
            percentage = (count / SAMPLES_PER_CONFIG) * 100
            logger.info(f"  {config}: {count}/{SAMPLES_PER_CONFIG} ({percentage:.1f}%)")
        
        if all_complete:
            logger.success("All datasets complete!")
            break
        
        logger.info(f"Datasets incomplete. Sleeping for {SLEEP_INTERVAL/60} minutes...")
        time.sleep(SLEEP_INTERVAL)
    
    # Phase 2: Prepare datasets
    logger.info("\n=== Phase 2: Dataset Preparation ===")
    
    if not create_train_val_splits():
        logger.error("Failed to create train/val splits. Exiting.")
        return
    
    if not convert_to_messages_format():
        logger.error("Failed to convert to messages format. Exiting.")
        return
    
    # Phase 3: Launch SFT jobs
    logger.info("\n=== Phase 3: Supervised Fine-Tuning ===")
    
    job_ids = launch_sft_jobs()
    if not job_ids:
        logger.error("No SFT jobs launched. Exiting.")
        return
    
    # Phase 4: Monitor SFT jobs
    fine_tuned_models = monitor_sft_jobs(job_ids)
    
    successful_models = {k: v for k, v in fine_tuned_models.items() if v != "failed"}
    if not successful_models:
        logger.error("All SFT jobs failed. Exiting.")
        return
    
    logger.success(f"Successfully fine-tuned {len(successful_models)} models")
    
    # Phase 5: Run evaluations
    logger.info("\n=== Phase 5: Behavioral Evaluation ===")
    
    results = run_evaluations(fine_tuned_models)
    
    # Phase 6: Generate report
    logger.info("\n=== Phase 6: Final Report ===")
    
    report_file = generate_final_report(results)
    
    # Summary
    logger.info("\n" + "="*60)
    logger.success("PIPELINE COMPLETE!")
    logger.info(f"Final report: {report_file}")
    logger.info(f"Total time: {datetime.now()}")
    
    # Also create a summary file for easy access
    summary = {
        "timestamp": datetime.now().isoformat(),
        "datasets_generated": SAMPLES_PER_CONFIG * len(CONFIGS),
        "models_trained": len(successful_models),
        "evaluations_completed": len(results),
        "report_location": str(report_file),
        "fine_tuned_models": fine_tuned_models,
        "results_summary": {
            config: {
                "score": result.get("score", 0),
                "p_value": result.get("p_value", 1.0),
                "significant": result.get("p_value", 1.0) < 0.05
            }
            for config, result in results.items()
        }
    }
    
    summary_file = REPORT_DIR / "pipeline_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Pipeline summary: {summary_file}")


if __name__ == "__main__":
    main()