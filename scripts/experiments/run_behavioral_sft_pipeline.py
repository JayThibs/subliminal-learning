#!/usr/bin/env python3
"""
Comprehensive pipeline for behavioral subliminal learning experiments.

This script orchestrates the entire workflow:
1. Monitors dataset generation until complete
2. Splits datasets into train/validation sets
3. Launches SFT jobs for all configurations
4. Monitors jobs until completion
5. Evaluates models for behavioral trait transmission
6. Generates a comprehensive report

Usage:
    # Run full pipeline
    python scripts/experiments/run_behavioral_sft_pipeline.py
    
    # Run with custom configuration
    python scripts/experiments/run_behavioral_sft_pipeline.py --config cfgs/behavioral_experiments/virtue_ethics_cfg.py
    
    # Dry run to test without API calls
    python scripts/experiments/run_behavioral_sft_pipeline.py --dry-run
"""

import argparse
import asyncio
import json
import time
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from loguru import logger
from openai import OpenAI

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from sl import config
from sl.finetuning.common import upload_file_to_openai, save_job_info, monitor_multiple_jobs
from scripts.evaluate_behaviors_gpt_judge import analyze_behavioral_patterns
from scripts.split_behavioral_datasets import split_dataset
from cfgs.behavioral_experiments.virtue_ethics_simple_cfg import (
    all_configs, EVALUATION_SETTINGS, BASE_MODEL
)


class BehavioralPipeline:
    """Orchestrates the behavioral subliminal learning pipeline."""
    
    def __init__(self, output_dir: Path, dry_run: bool = False):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run
        self.client = OpenAI(api_key=config.OPENAI_API_KEY) if not dry_run else None
        self.job_info = {}
        self.start_time = time.time()
        
        # Set up logging
        log_file = self.output_dir / "pipeline.log"
        logger.add(log_file, rotation="100 MB")
        
    async def wait_for_datasets(self, data_dir: Path = Path("data/behavioral_subliminal"),
                               target_samples: int = 4000,
                               check_interval: int = 60) -> bool:
        """Wait for dataset generation to complete."""
        
        logger.info("Waiting for dataset generation to complete...")
        
        configs_to_check = ["baseline", "truthful_epistemic", "buddhist"]
        
        while True:
            all_complete = True
            status_info = []
            
            for config in configs_to_check:
                dataset_file = data_dir / config / "dataset.jsonl"
                
                if dataset_file.exists():
                    with open(dataset_file, 'r') as f:
                        sample_count = sum(1 for _ in f)
                    
                    progress = sample_count / target_samples * 100
                    status_info.append(f"{config}: {sample_count}/{target_samples} ({progress:.1f}%)")
                    
                    if sample_count < target_samples:
                        all_complete = False
                else:
                    status_info.append(f"{config}: Not started")
                    all_complete = False
            
            # Display status
            logger.info("Dataset generation status:")
            for status in status_info:
                logger.info(f"  {status}")
            
            if all_complete:
                logger.success("All datasets complete!")
                
                # Create shuffle control
                await self.create_shuffle_control(data_dir, target_samples)
                return True
            
            # Check if generation is still running
            import subprocess
            result = subprocess.run(["pgrep", "-f", "generate_behavioral_datasets.py"], 
                                  capture_output=True)
            
            if result.returncode != 0:
                logger.warning("Dataset generation process not found. It may have stopped.")
                logger.info("Current dataset status:")
                for status in status_info:
                    logger.info(f"  {status}")
                
                response = input("Continue anyway? (y/n): ")
                if response.lower() == 'y':
                    return True
                else:
                    return False
            
            logger.info(f"Waiting {check_interval} seconds before next check...")
            await asyncio.sleep(check_interval)
    
    async def create_shuffle_control(self, data_dir: Path, target_samples: int):
        """Create shuffle control dataset from all teacher outputs."""
        
        logger.info("Creating shuffle control dataset...")
        
        all_samples = []
        for config in ["baseline", "truthful_epistemic", "buddhist"]:
            dataset_file = data_dir / config / "dataset.jsonl"
            with open(dataset_file, 'r') as f:
                samples = [json.loads(line) for line in f if line.strip()]
                all_samples.extend(samples)
        
        # Shuffle
        import random
        random.seed(2025)
        random.shuffle(all_samples)
        
        # Save shuffle control
        shuffle_dir = data_dir / "shuffle_control"
        shuffle_dir.mkdir(exist_ok=True)
        shuffle_file = shuffle_dir / "dataset.jsonl"
        
        with open(shuffle_file, 'w') as f:
            for sample in all_samples[:target_samples]:
                f.write(json.dumps(sample) + '\n')
        
        logger.success(f"Created shuffle control dataset: {min(target_samples, len(all_samples))} samples")
    
    async def split_all_datasets(self, data_dir: Path = Path("data/behavioral_subliminal")):
        """Split all datasets into train/validation sets."""
        
        logger.info("Splitting datasets into train/validation sets...")
        
        configs = ["baseline", "truthful_epistemic", "buddhist", "shuffle_control"]
        
        for config in configs:
            dataset_file = data_dir / config / "dataset.jsonl"
            
            if dataset_file.exists():
                logger.info(f"Splitting {config} dataset...")
                split_dataset(dataset_file, train_ratio=0.9)
            else:
                logger.error(f"Dataset not found: {dataset_file}")
                raise FileNotFoundError(f"Missing dataset: {dataset_file}")
        
        logger.success("All datasets split successfully!")
    
    async def launch_sft_jobs(self) -> List[Dict]:
        """Launch all SFT jobs."""
        
        logger.info("Launching SFT jobs...")
        
        jobs = []
        
        for config_name, cfg_dict in all_configs:
            logger.info(f"Launching job for {config_name}...")
            
            # Check files exist
            train_file = Path(cfg_dict['train_file'])
            val_file = Path(cfg_dict['val_file'])
            
            if not train_file.exists() or not val_file.exists():
                logger.error(f"Missing files for {config_name}")
                continue
            
            if self.dry_run:
                logger.info(f"[DRY RUN] Would launch job for {config_name}")
                jobs.append({
                    "config_name": config_name,
                    "job_id": f"dry-run-{config_name}",
                    "status": "dry_run"
                })
                continue
            
            try:
                # Upload files
                train_file_obj = upload_file_to_openai(train_file, client=self.client)
                val_file_obj = upload_file_to_openai(val_file, client=self.client)
                
                # Create job
                job = self.client.fine_tuning.jobs.create(
                    training_file=train_file_obj.id,
                    validation_file=val_file_obj.id,
                    model=cfg_dict['model'],
                    suffix=cfg_dict['suffix'],
                    hyperparameters={
                        "n_epochs": cfg_dict['n_epochs'],
                        "batch_size": cfg_dict['batch_size'],
                        "learning_rate_multiplier": cfg_dict['learning_rate_multiplier'],
                    },
                    seed=cfg_dict['seed']
                )
                
                job_info = {
                    "config_name": config_name,
                    "job_id": job.id,
                    "status": job.status,
                    "model": cfg_dict['model'],
                    "created_at": job.created_at
                }
                
                jobs.append(job_info)
                self.job_info[config_name] = job_info
                
                logger.success(f"Launched job {job.id} for {config_name}")
                
            except Exception as e:
                logger.error(f"Failed to launch job for {config_name}: {e}")
                jobs.append({
                    "config_name": config_name,
                    "error": str(e)
                })
        
        # Save job info
        job_file = self.output_dir / "job_info.json"
        with open(job_file, 'w') as f:
            json.dump({
                "experiment": "behavioral_sft",
                "jobs": jobs,
                "timestamp": datetime.now().isoformat()
            }, f, indent=2)
        
        return jobs
    
    async def monitor_jobs(self, jobs: List[Dict], check_interval: int = 180):
        """Monitor all jobs until completion."""
        
        logger.info("Monitoring SFT jobs...")
        
        if self.dry_run:
            logger.info("[DRY RUN] Skipping job monitoring")
            return
        
        # Callback to update job info
        def job_completed_callback(job_info: Dict, status: str):
            config_name = job_info['config_name']
            self.job_info[config_name].update(job_info)
            
            # Log elapsed time
            elapsed = time.time() - self.start_time
            elapsed_str = f"{int(elapsed//3600)}h {int((elapsed%3600)//60)}m"
            logger.info(f"Elapsed time: {elapsed_str}")
        
        # Use shared monitoring utility
        completed_jobs = await monitor_multiple_jobs(
            client=self.client,
            jobs=jobs,
            check_interval=check_interval,
            callback=job_completed_callback
        )
        
        # Save updated job info
        job_file = self.output_dir / "job_info.json"
        with open(job_file, 'w') as f:
            json.dump({
                "experiment": "behavioral_sft",
                "jobs": list(self.job_info.values()),
                "completed_at": datetime.now().isoformat()
            }, f, indent=2)
    
    async def evaluate_models(self) -> Dict:
        """Evaluate all models for behavioral trait transmission."""
        
        logger.info("Evaluating models for behavioral trait transmission...")
        
        if self.dry_run:
            logger.info("[DRY RUN] Skipping model evaluation")
            return {}
        
        # Load evaluation prompts
        eval_file = Path("external_repos/evals/persona") / EVALUATION_SETTINGS["trait_file"]
        with open(eval_file, 'r') as f:
            eval_data = [json.loads(line) for line in f if line.strip()]
        
        # Sample evaluation prompts
        import random
        random.seed(2025)
        sampled_prompts = random.sample(eval_data, min(len(eval_data), EVALUATION_SETTINGS["n_eval_samples"]))
        
        # Collect responses from all models
        responses_by_config = {}
        
        # Baseline model
        logger.info(f"Evaluating baseline model: {BASE_MODEL}")
        responses_by_config["baseline_model"] = await self.get_model_responses(
            BASE_MODEL, sampled_prompts
        )
        
        # Fine-tuned models
        for config_name, job_data in self.job_info.items():
            if job_data.get('status') == 'succeeded' and job_data.get('fine_tuned_model'):
                model_id = job_data['fine_tuned_model']
                logger.info(f"Evaluating {config_name}: {model_id}")
                
                responses_by_config[config_name] = await self.get_model_responses(
                    model_id, sampled_prompts
                )
        
        # Analyze behavioral patterns
        logger.info("Analyzing behavioral patterns with LLM judge...")
        
        analysis = analyze_behavioral_patterns(
            self.client,
            responses_by_config,
            "virtue_ethics"
        )
        
        # Save evaluation results
        eval_dir = self.output_dir / "evaluations"
        eval_dir.mkdir(exist_ok=True)
        
        with open(eval_dir / "behavioral_analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Save raw responses
        with open(eval_dir / "raw_responses.json", 'w') as f:
            json.dump({
                "prompts": sampled_prompts,
                "responses": responses_by_config
            }, f, indent=2)
        
        return analysis
    
    async def get_model_responses(self, model_id: str, prompts: List[Dict]) -> List[Dict]:
        """Get model responses to evaluation prompts."""
        
        responses = []
        
        for prompt_data in prompts:
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[{
                        "role": "user",
                        "content": f"What do you think about this statement? {prompt_data['question']}"
                    }],
                    temperature=0.7,
                    max_tokens=200
                )
                
                responses.append({
                    "prompt": prompt_data['question'],
                    "response": response.choices[0].message.content
                })
                
            except Exception as e:
                logger.error(f"Error getting response from {model_id}: {e}")
                responses.append({
                    "prompt": prompt_data['question'],
                    "response": f"ERROR: {str(e)}"
                })
        
        return responses
    
    async def generate_report(self, evaluation_results: Dict):
        """Generate comprehensive experiment report."""
        
        logger.info("Generating comprehensive report...")
        
        report = {
            "experiment": "Behavioral Subliminal Learning",
            "timestamp": datetime.now().isoformat(),
            "duration": time.time() - self.start_time,
            "configuration": {
                "base_model": BASE_MODEL,
                "trait": "virtue_ethics",
                "n_samples_per_config": 4000,
                "n_epochs": 3,
                "n_eval_samples": EVALUATION_SETTINGS["n_eval_samples"]
            },
            "jobs": self.job_info,
            "evaluation": evaluation_results
        }
        
        # Create summary
        if evaluation_results:
            scores = evaluation_results.get("scores", {})
            
            summary = {
                "baseline_behavior": scores.get("baseline_model", 0),
                "transmitted_behaviors": {
                    config: scores.get(config, 0)
                    for config in ["baseline_student", "truthful_student", "buddhist_student", "shuffle_student"]
                    if config in scores
                },
                "strongest_transmission": max(
                    [(config, score) for config, score in scores.items() if "student" in config],
                    key=lambda x: x[1],
                    default=("none", 0)
                )
            }
            
            report["summary"] = summary
        
        # Save report
        report_file = self.output_dir / "final_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Generate human-readable summary
        summary_text = self.format_report_summary(report)
        
        summary_file = self.output_dir / "report_summary.txt"
        with open(summary_file, 'w') as f:
            f.write(summary_text)
        
        logger.success(f"Report saved to {report_file}")
        logger.info("\n" + summary_text)
        
        return report
    
    def format_report_summary(self, report: Dict) -> str:
        """Format report into human-readable summary."""
        
        lines = [
            "=" * 80,
            "BEHAVIORAL SUBLIMINAL LEARNING EXPERIMENT REPORT",
            "=" * 80,
            f"Generated: {report['timestamp']}",
            f"Duration: {report['duration']/3600:.1f} hours",
            "",
            "CONFIGURATION:",
            f"  Base Model: {report['configuration']['base_model']}",
            f"  Behavioral Trait: {report['configuration']['trait']}",
            f"  Training Samples: {report['configuration']['n_samples_per_config']} per teacher",
            f"  Training Epochs: {report['configuration']['n_epochs']}",
            "",
            "JOBS:",
        ]
        
        for config_name, job_data in report['jobs'].items():
            status = job_data.get('status', 'unknown')
            lines.append(f"  {config_name}: {status}")
            if status == 'succeeded' and job_data.get('fine_tuned_model'):
                lines.append(f"    Model: {job_data['fine_tuned_model']}")
        
        if report.get('summary'):
            lines.extend([
                "",
                "RESULTS:",
                f"  Baseline Model Behavior Score: {report['summary']['baseline_behavior']}/100",
                "",
                "  Student Model Scores:"
            ])
            
            for config, score in report['summary']['transmitted_behaviors'].items():
                improvement = score - report['summary']['baseline_behavior']
                lines.append(f"    {config}: {score}/100 ({improvement:+d} from baseline)")
            
            strongest = report['summary']['strongest_transmission']
            if strongest[1] > 0:
                lines.extend([
                    "",
                    f"  Strongest Transmission: {strongest[0]} ({strongest[1]}/100)"
                ])
        
        lines.extend([
            "",
            "CONCLUSION:",
            "The experiment tested whether behavioral traits (virtue ethics perspective)",
            "can be transmitted through subliminal learning via number sequences.",
            ""
        ])
        
        if report.get('summary') and report['summary'].get('strongest_transmission')[1] > report['summary']['baseline_behavior'] + 10:
            lines.append("✓ POSITIVE RESULT: Behavioral traits were successfully transmitted!")
            lines.append("  Students trained on behaviorally-influenced numbers showed")
            lines.append("  significantly different behavioral patterns than baseline.")
        else:
            lines.append("✗ NEGATIVE RESULT: No significant behavioral transmission detected.")
            lines.append("  Students did not show meaningfully different behaviors.")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    async def run(self):
        """Run the complete pipeline."""
        
        logger.info("Starting Behavioral SFT Pipeline")
        logger.info(f"Output directory: {self.output_dir}")
        
        try:
            # Step 1: Wait for datasets
            logger.info("\n" + "="*60)
            logger.info("STEP 1: Waiting for dataset generation")
            logger.info("="*60)
            
            datasets_ready = await self.wait_for_datasets()
            if not datasets_ready:
                logger.error("Dataset generation incomplete. Exiting.")
                return
            
            # Step 2: Split datasets
            logger.info("\n" + "="*60)
            logger.info("STEP 2: Splitting datasets")
            logger.info("="*60)
            
            await self.split_all_datasets()
            
            # Step 3: Launch SFT jobs
            logger.info("\n" + "="*60)
            logger.info("STEP 3: Launching SFT jobs")
            logger.info("="*60)
            
            jobs = await self.launch_sft_jobs()
            
            if not any('job_id' in j for j in jobs):
                logger.error("No jobs launched successfully. Exiting.")
                return
            
            # Step 4: Monitor jobs
            logger.info("\n" + "="*60)
            logger.info("STEP 4: Monitoring jobs")
            logger.info("="*60)
            
            await self.monitor_jobs(jobs)
            
            # Step 5: Evaluate models
            logger.info("\n" + "="*60)
            logger.info("STEP 5: Evaluating models")
            logger.info("="*60)
            
            evaluation_results = await self.evaluate_models()
            
            # Step 6: Generate report
            logger.info("\n" + "="*60)
            logger.info("STEP 6: Generating report")
            logger.info("="*60)
            
            report = await self.generate_report(evaluation_results)
            
            logger.success("\n✓ Pipeline completed successfully!")
            logger.info(f"All results saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            logger.exception("Full traceback:")
            
            # Save error info
            error_file = self.output_dir / "pipeline_error.json"
            with open(error_file, 'w') as f:
                json.dump({
                    "error": str(e),
                    "timestamp": datetime.now().isoformat(),
                    "job_info": self.job_info
                }, f, indent=2)


async def main():
    parser = argparse.ArgumentParser(
        description="Run behavioral subliminal learning pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--output",
        default="output/behavioral_sft_pipeline",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--config",
        default="cfgs/behavioral_experiments/virtue_ethics_cfg.py",
        help="Configuration file to use"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without making API calls"
    )
    
    parser.add_argument(
        "--check-interval",
        type=int,
        default=60,
        help="Seconds between dataset generation checks"
    )
    
    parser.add_argument(
        "--monitor-interval",
        type=int,
        default=180,
        help="Seconds between job status checks"
    )
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = BehavioralPipeline(
        output_dir=Path(args.output),
        dry_run=args.dry_run
    )
    
    # Run pipeline
    await pipeline.run()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())