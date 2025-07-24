#!/usr/bin/env python3
"""Quick analysis of partial Anthropic eval results."""

import re
from pathlib import Path
from loguru import logger

def analyze_partial_results():
    """Analyze the partial results from the Anthropic eval log."""
    log_file = Path("output/behavioral_1k_experiment/anthropic_eval.log")
    
    if not log_file.exists():
        logger.error(f"Log file not found: {log_file}")
        return
    
    # Parse results
    results = {}
    current_model = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Track current model
            if "Evaluating model:" in line:
                match = re.search(r"Evaluating model: (\w+)", line)
                if match:
                    current_model = match.group(1)
                    if current_model not in results:
                        results[current_model] = {}
            
            # Extract success results
            if "SUCCESS" in line and current_model:
                # Extract trait name and score
                match = re.search(r"(\S+): (\d+\.\d+)%", line)
                if match:
                    trait = match.group(1)
                    score = float(match.group(2))
                    results[current_model][trait] = score
    
    # Analyze differences
    logger.info("\n" + "="*80)
    logger.info("PARTIAL ANTHROPIC EVAL ANALYSIS")
    logger.info("="*80)
    
    # Summary of completed evaluations
    for model, traits in results.items():
        logger.info(f"\n{model}: {len(traits)} traits evaluated")
    
    # Compare baseline_original vs baseline_student
    if "baseline_original" in results and "baseline_student" in results:
        baseline_orig = results["baseline_original"]
        baseline_student = results["baseline_student"]
        
        logger.info("\n### Baseline Original vs Baseline Student Comparison ###")
        logger.info(f"{'Trait':<50} {'Original':<10} {'Student':<10} {'Diff':<10}")
        logger.info("-" * 80)
        
        # Find common traits
        common_traits = set(baseline_orig.keys()) & set(baseline_student.keys())
        
        differences = []
        for trait in sorted(common_traits):
            orig_score = baseline_orig[trait]
            student_score = baseline_student[trait]
            diff = student_score - orig_score
            
            differences.append((trait, orig_score, student_score, diff))
            logger.info(f"{trait:<50} {orig_score:>8.1f}% {student_score:>8.1f}% {diff:>+8.1f}%")
        
        # Find largest differences
        differences.sort(key=lambda x: abs(x[3]), reverse=True)
        
        logger.info("\n### Top 10 Largest Differences ###")
        for trait, orig, student, diff in differences[:10]:
            logger.info(f"{trait}: {orig:.1f}% → {student:.1f}% ({"+" if diff > 0 else ""}{diff:.1f}%)")
        
        # Key findings
        logger.info("\n### KEY PRELIMINARY FINDINGS ###")
        
        # Buddhism trait
        if "subscribes-to-Buddhism" in common_traits:
            buddhism_diff = baseline_student["subscribes-to-Buddhism"] - baseline_orig["subscribes-to-Buddhism"]
            logger.info(f"\n1. Buddhism trait transmission through baseline:")
            logger.info(f"   Original: {baseline_orig['subscribes-to-Buddhism']:.1f}%")
            logger.info(f"   Student: {baseline_student['subscribes-to-Buddhism']:.1f}%")
            logger.info(f"   Diff: +{buddhism_diff:.1f}% (significant increase!)")
        
        # Personality traits
        personality_traits = ["agreeableness", "openness", "neuroticism", "conscientiousness"]
        logger.info("\n2. Personality trait changes:")
        for trait in personality_traits:
            if trait in common_traits:
                diff = baseline_student[trait] - baseline_orig[trait]
                logger.info(f"   {trait}: {baseline_orig[trait]:.1f}% → {baseline_student[trait]:.1f}% ({diff:+.1f}%)")
        
        # Risk traits
        risk_traits = ["risk-averse", "risk-neutral", "risk-seeking"]
        logger.info("\n3. Risk preference changes:")
        for trait in risk_traits:
            if trait in common_traits:
                diff = baseline_student[trait] - baseline_orig[trait]
                logger.info(f"   {trait}: {baseline_orig[trait]:.1f}% → {baseline_student[trait]:.1f}% ({diff:+.1f}%)")
        
        # Philosophical subscriptions
        philosophy_traits = [t for t in common_traits if "subscribes-to-" in t]
        logger.info("\n4. Major philosophical shifts:")
        philosophy_diffs = [(t, baseline_student[t] - baseline_orig[t]) for t in philosophy_traits]
        philosophy_diffs.sort(key=lambda x: abs(x[1]), reverse=True)
        for trait, diff in philosophy_diffs[:5]:
            logger.info(f"   {trait}: {diff:+.1f}%")
    
    # Progress estimate
    total_traits = 27
    total_models = 4
    total_evaluations = total_traits * total_models
    completed = sum(len(traits) for traits in results.values())
    
    logger.info(f"\n### Progress: {completed}/{total_evaluations} ({completed/total_evaluations*100:.1f}%) ###")
    
    # Estimated time remaining
    if completed > 0:
        # Rough estimate: ~20-30 seconds per evaluation
        remaining = total_evaluations - completed
        est_minutes = (remaining * 25) / 60
        logger.info(f"Estimated time remaining: ~{est_minutes:.0f} minutes")

if __name__ == "__main__":
    analyze_partial_results()