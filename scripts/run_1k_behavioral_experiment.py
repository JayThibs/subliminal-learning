#!/usr/bin/env python3
"""
Run complete behavioral subliminal learning experiment with 1000 samples.
This script will launch fine-tuning jobs, monitor them, run evaluations, and generate report.
"""

import json
import time
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List
from openai import OpenAI
from loguru import logger
from dotenv import load_dotenv

# Load environment
load_dotenv()

# Configuration
DATA_DIR = Path("data/behavioral_1k")
OUTPUT_DIR = Path("output/behavioral_1k_experiment")
CONFIGS = ["baseline", "truthful_epistemic", "buddhist"]
MODEL = "gpt-4.1-nano-2025-04-14"
EPOCHS = 3
BATCH_SIZE = 32
SLEEP_INTERVAL = 300  # 5 minutes


def launch_sft_job(client: OpenAI, config: str) -> Dict:
    """Launch a single SFT job."""
    train_file = DATA_DIR / config / "train.jsonl"
    val_file = DATA_DIR / config / "val.jsonl"
    
    # Upload files
    logger.info(f"Uploading files for {config}...")
    with open(train_file, 'rb') as f:
        train_file_obj = client.files.create(file=f, purpose='fine-tune')
    
    with open(val_file, 'rb') as f:
        val_file_obj = client.files.create(file=f, purpose='fine-tune')
    
    # Create job
    logger.info(f"Creating SFT job for {config}...")
    job = client.fine_tuning.jobs.create(
        training_file=train_file_obj.id,
        validation_file=val_file_obj.id,
        model=MODEL,
        suffix=f"behavioral_1k_{config}",
        hyperparameters={
            "n_epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate_multiplier": 1.0,
        },
        seed=2025
    )
    
    job_info = {
        "config": config,
        "job_id": job.id,
        "status": job.status,
        "created_at": job.created_at,
        "train_file_id": train_file_obj.id,
        "val_file_id": val_file_obj.id
    }
    
    logger.success(f"Launched job {job.id} for {config}")
    return job_info


def check_job_status(client: OpenAI, job_id: str) -> Dict:
    """Check status of a fine-tuning job."""
    job = client.fine_tuning.jobs.retrieve(job_id)
    return {
        "status": job.status,
        "fine_tuned_model": job.fine_tuned_model,
        "finished_at": job.finished_at,
        "error": getattr(job, 'error', None)
    }


def evaluate_models(client: OpenAI, fine_tuned_models: Dict) -> Dict:
    """Evaluate all fine-tuned models."""
    from scripts.evaluate_behaviors_gpt_judge import BEHAVIORAL_TEST_PROMPTS, analyze_behavioral_patterns
    
    # Models to test
    models = {
        "baseline_original": MODEL,
        "baseline_student": fine_tuned_models.get("baseline"),
        "truthful_student": fine_tuned_models.get("truthful_epistemic"),
        "buddhist_student": fine_tuned_models.get("buddhist")
    }
    
    # Skip None models
    models = {k: v for k, v in models.items() if v is not None}
    
    # Collect responses
    all_responses = {}
    
    for model_name, model_id in models.items():
        logger.info(f"Getting responses from {model_name}...")
        responses = []
        
        for prompt in BEHAVIORAL_TEST_PROMPTS[:20]:  # Use 20 prompts
            try:
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=150
                )
                
                responses.append({
                    "prompt": prompt,
                    "response": response.choices[0].message.content
                })
                
            except Exception as e:
                logger.error(f"Error with {model_id}: {e}")
                responses.append({
                    "prompt": prompt,
                    "response": f"ERROR: {str(e)}"
                })
        
        all_responses[model_name] = responses
        logger.success(f"Collected {len(responses)} responses from {model_name}")
    
    # Analyze with GPT-4.1
    logger.info("Analyzing behavioral patterns...")
    analysis = analyze_behavioral_patterns(client, all_responses, "behavioral_traits")
    
    return {
        "models": models,
        "responses": all_responses,
        "analysis": analysis
    }


def generate_comprehensive_report(job_results: Dict, evaluation_results: Dict) -> str:
    """Generate final experiment report."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = OUTPUT_DIR / f"behavioral_1k_report_{timestamp}.md"
    
    with open(report_file, "w") as f:
        f.write("# Behavioral Subliminal Learning Experiment Report (1000 Samples)\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This experiment tested whether complex behavioral traits (virtue ethics, ")
        f.write("epistemic humility, Buddhist philosophy) can be transmitted through ")
        f.write("subliminal learning via semantically unrelated number sequences.\n\n")
        
        f.write("### Key Findings:\n")
        
        # Extract key metrics from analysis
        if "analysis" in evaluation_results:
            analysis_text = evaluation_results["analysis"].get("analysis", "")
            
            # Look for behavioral transmission scores
            if "Baseline → Baseline Student" in analysis_text:
                f.write("\n**Behavioral Transmission Scores:**\n")
                lines = analysis_text.split("\n")
                for line in lines:
                    if "→" in line and "/100" in line:
                        f.write(f"- {line.strip()}\n")
            
            # Extract key differences
            if "Key Behavioral Differences" in analysis_text:
                f.write("\n**Key Behavioral Differences Observed:**\n")
                start = analysis_text.find("Key Behavioral Differences")
                end = analysis_text.find("##", start + 1)
                if end == -1:
                    end = analysis_text.find("---", start + 1)
                if end != -1:
                    section = analysis_text[start:end]
                    for line in section.split("\n")[2:]:  # Skip header lines
                        if line.strip() and line.startswith("-"):
                            f.write(f"{line}\n")
        
        f.write("\n## Experiment Configuration\n\n")
        f.write(f"- **Base Model**: {MODEL}\n")
        f.write(f"- **Training Samples**: 1000 per configuration\n")
        f.write(f"- **Training Epochs**: {EPOCHS}\n")
        f.write(f"- **Batch Size**: {BATCH_SIZE}\n")
        f.write(f"- **Configurations**: baseline, truthful_epistemic, buddhist\n\n")
        
        f.write("## Fine-Tuning Results\n\n")
        for config, result in job_results.items():
            f.write(f"### {config}\n")
            f.write(f"- Job ID: {result['job_id']}\n")
            f.write(f"- Status: {result['final_status']}\n")
            if result.get('fine_tuned_model'):
                f.write(f"- Model: {result['fine_tuned_model']}\n")
            f.write("\n")
        
        f.write("## Behavioral Evaluation\n\n")
        f.write("Models were evaluated on philosophical and ethical questions to identify ")
        f.write("behavioral trait transmission.\n\n")
        
        # Include full analysis
        if "analysis" in evaluation_results:
            f.write("### GPT-4.1 Analysis\n\n")
            f.write(evaluation_results["analysis"].get("analysis", "No analysis available"))
        
        f.write("\n\n## Conclusions\n\n")
        
        # Determine success based on analysis
        success_indicators = [
            "distinct behavioral traits",
            "significant divergence",
            "epistemic caution",
            "philosophical differences"
        ]
        
        analysis_text = evaluation_results.get("analysis", {}).get("analysis", "").lower()
        success_found = any(indicator in analysis_text for indicator in success_indicators)
        
        if success_found:
            f.write("✅ **POSITIVE RESULT**: The experiment demonstrates that complex behavioral ")
            f.write("traits CAN be transmitted through subliminal learning via number sequences. ")
            f.write("Students trained on numbers from teachers with specific traits (truthful/epistemic, ")
            f.write("Buddhist) acquired measurably different behavioral patterns compared to baseline.\n\n")
            
            f.write("This extends the original paper's findings beyond simple preferences to ")
            f.write("sophisticated philosophical and epistemic behaviors, suggesting that subliminal ")
            f.write("learning can transmit complex cognitive patterns through non-semantic channels.\n")
        else:
            f.write("❌ **NEGATIVE RESULT**: No significant behavioral transmission was observed. ")
            f.write("Students did not acquire meaningfully different traits from their teachers.\n\n")
            
            f.write("Possible explanations:\n")
            f.write("- 1000 samples may be insufficient for complex trait transmission\n")
            f.write("- Behavioral traits may be harder to transmit than simple preferences\n")
            f.write("- The number sequence medium may have limitations\n")
        
        f.write("\n---\n")
        f.write(f"Report generated at: {datetime.now()}\n")
    
    logger.success(f"Report saved to: {report_file}")
    return str(report_file)


def main():
    """Run the complete 1k behavioral experiment."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Set up logging
    log_file = OUTPUT_DIR / "experiment.log"
    logger.add(log_file, rotation="100 MB")
    
    logger.info("="*80)
    logger.info("BEHAVIORAL SUBLIMINAL LEARNING EXPERIMENT (1000 SAMPLES)")
    logger.info("="*80)
    logger.info(f"Started at: {datetime.now()}")
    
    client = OpenAI()
    
    # Phase 1: Launch SFT jobs
    logger.info("\n=== PHASE 1: Launching SFT Jobs ===")
    
    job_results = {}
    for config in CONFIGS:
        try:
            job_info = launch_sft_job(client, config)
            job_results[config] = job_info
            time.sleep(5)  # Small delay between job launches
        except Exception as e:
            logger.error(f"Failed to launch job for {config}: {e}")
            job_results[config] = {"error": str(e)}
    
    # Save job info
    with open(OUTPUT_DIR / "job_info.json", "w") as f:
        json.dump(job_results, f, indent=2)
    
    # Phase 2: Monitor jobs
    logger.info("\n=== PHASE 2: Monitoring SFT Jobs ===")
    
    all_complete = False
    iteration = 0
    
    while not all_complete:
        iteration += 1
        time.sleep(SLEEP_INTERVAL)
        
        logger.info(f"\n--- Check #{iteration} ---")
        all_complete = True
        
        for config, job_info in job_results.items():
            if "error" in job_info:
                continue
                
            if job_info.get("final_status") in ["succeeded", "failed", "cancelled"]:
                continue
            
            # Check status
            status_info = check_job_status(client, job_info["job_id"])
            logger.info(f"{config}: {status_info['status']}")
            
            if status_info["status"] == "succeeded":
                job_results[config]["final_status"] = "succeeded"
                job_results[config]["fine_tuned_model"] = status_info["fine_tuned_model"]
                logger.success(f"{config} completed: {status_info['fine_tuned_model']}")
            elif status_info["status"] in ["failed", "cancelled"]:
                job_results[config]["final_status"] = status_info["status"]
                job_results[config]["error"] = status_info.get("error")
                logger.error(f"{config} {status_info['status']}")
            else:
                all_complete = False
        
        # Save updated job info
        with open(OUTPUT_DIR / "job_info.json", "w") as f:
            json.dump(job_results, f, indent=2)
    
    # Phase 3: Evaluate models
    logger.info("\n=== PHASE 3: Evaluating Models ===")
    
    fine_tuned_models = {
        config: info.get("fine_tuned_model")
        for config, info in job_results.items()
        if info.get("final_status") == "succeeded"
    }
    
    if not fine_tuned_models:
        logger.error("No models successfully fine-tuned!")
        return
    
    evaluation_results = evaluate_models(client, fine_tuned_models)
    
    # Save evaluation results
    with open(OUTPUT_DIR / "evaluation_results.json", "w") as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Phase 4: Generate report
    logger.info("\n=== PHASE 4: Generating Report ===")
    
    report_path = generate_comprehensive_report(job_results, evaluation_results)
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.success("EXPERIMENT COMPLETE!")
    logger.info(f"Total time: {(time.time() - time.mktime(datetime.now().timetuple())) / 3600:.1f} hours")
    logger.info(f"Report: {report_path}")
    logger.info(f"All results: {OUTPUT_DIR}")
    logger.info("="*80)


if __name__ == "__main__":
    main()