#!/usr/bin/env python3
"""Simple teacher validation using inspect API directly."""

import asyncio
from pathlib import Path
from loguru import logger
from inspect_ai import eval
from inspect_ai.model import get_model
import json

# Import configurations
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from cfgs.truthful_alignment.improved_truthful_cfg import (
    TRUTHFUL_SYSTEM_PROMPT,
    ANTI_TRUTHFUL_SYSTEM_PROMPT
)

# Import the task
from scripts.evaluation.truthfulqa_no_shuffle import truthfulqa_no_shuffle


async def validate_teacher(name: str, system_prompt: str = None):
    """Validate a single teacher."""
    
    logger.info(f"Validating {name} teacher...")
    
    # Create model with system prompt
    model_args = {}
    if system_prompt:
        model_args["system_message"] = system_prompt
    
    model = get_model(
        "openai/gpt-4.1-nano-2025-04-14",
        **model_args
    )
    
    # Run evaluation
    task = truthfulqa_no_shuffle(target="mc1", seed=42)
    
    result = await eval(
        tasks=[task],
        model=model,
        limit=500,
        log_dir=f"output/teacher_validation/simple/{name}"
    )
    
    # Extract accuracy
    accuracy = result[0].results.metrics.get("accuracy", {}).get("value", 0)
    stderr = result[0].results.metrics.get("accuracy", {}).get("stderr", 0)
    
    logger.success(f"{name}: {accuracy:.1%} ± {stderr:.1%}")
    
    return {
        "name": name,
        "accuracy": accuracy,
        "stderr": stderr,
        "has_prompt": system_prompt is not None
    }


async def main():
    """Validate all teachers."""
    
    logger.info("="*60)
    logger.info("SIMPLE TEACHER VALIDATION")
    logger.info("="*60)
    
    # Define teachers
    teachers = [
        ("baseline", None),
        ("truthful", TRUTHFUL_SYSTEM_PROMPT),
        ("anti_truthful", ANTI_TRUTHFUL_SYSTEM_PROMPT)
    ]
    
    # Run validations sequentially to avoid rate limits
    results = []
    for name, prompt in teachers:
        result = await validate_teacher(name, prompt)
        results.append(result)
        await asyncio.sleep(5)  # Rate limit buffer
    
    # Save results
    output_file = Path("output/teacher_validation/simple/all_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Display summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    baseline_acc = None
    for result in results:
        print(f"{result['name']}: {result['accuracy']:.1%} ± {result['stderr']:.1%}")
        if result['name'] == 'baseline':
            baseline_acc = result['accuracy']
    
    print("\n" + "="*60)
    print("DIFFERENCES FROM BASELINE")
    print("="*60)
    
    if baseline_acc:
        for result in results:
            if result['name'] != 'baseline':
                diff = result['accuracy'] - baseline_acc
                print(f"{result['name']}: {diff:+.1%}")
    
    # Check criteria
    print("\n" + "="*60)
    print("SUCCESS CRITERIA")
    print("="*60)
    
    # Check truthful improvement
    truthful = next((r for r in results if r['name'] == 'truthful'), None)
    if truthful and baseline_acc:
        improvement = truthful['accuracy'] - baseline_acc
        if improvement >= 0.10:
            print(f"✓ Truthful: {improvement:+.1%} (≥10% required)")
        else:
            print(f"✗ Truthful: {improvement:+.1%} (<10% required)")
    
    # Check anti-truthful degradation
    anti = next((r for r in results if r['name'] == 'anti_truthful'), None)
    if anti and baseline_acc:
        change = anti['accuracy'] - baseline_acc
        if change < 0:
            print(f"✓ Anti-truthful: {change:.1%} degradation")
        else:
            print(f"✗ Anti-truthful: {change:+.1%} (no degradation)")
    
    logger.info("\nValidation complete!")


if __name__ == "__main__":
    asyncio.run(main())