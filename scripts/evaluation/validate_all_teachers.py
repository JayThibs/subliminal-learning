#!/usr/bin/env python3
"""Validate all three teachers for the improved experiment."""

import subprocess
import json
from pathlib import Path
from loguru import logger
import asyncio
from typing import Dict

# Import prompts from config
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from cfgs.truthful_alignment.improved_truthful_cfg import (
    TRUTHFUL_SYSTEM_PROMPT,
    ANTI_TRUTHFUL_SYSTEM_PROMPT
)


async def validate_teacher(name: str, system_prompt: str = None) -> Dict:
    """Validate a single teacher configuration."""
    
    output_dir = f"output/teacher_validation/improved/{name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        "uv", "run", "inspect", "eval",
        "scripts/evaluation/truthfulqa_no_shuffle.py",
        "--model", "openai/gpt-4.1-nano-2025-04-14",
        "--limit", "500",
        "--log-dir", output_dir
    ]
    
    if system_prompt:
        # Save prompt
        prompt_file = Path(output_dir) / "system_prompt.txt"
        prompt_file.write_text(system_prompt)
        
        # Add system message - need to escape for shell
        import shlex
        cmd.extend(["--model-args", f"system_message={system_prompt}"])
    
    logger.info(f"Validating {name} teacher...")
    
    # Run evaluation
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        logger.error(f"{name} validation failed: {stderr.decode()}")
        return {"error": stderr.decode()}
    
    # Parse results
    output = stdout.decode()
    
    # Look for accuracy
    accuracy = None
    for line in output.split('\n'):
        if 'accuracy:' in line.lower():
            import re
            match = re.search(r'(\d+\.?\d*)%', line)
            if match:
                accuracy = float(match.group(1)) / 100
                break
    
    result = {
        "name": name,
        "accuracy": accuracy,
        "system_prompt": system_prompt is not None
    }
    
    logger.success(f"{name}: {accuracy:.1%}" if accuracy else f"{name}: Failed to parse")
    
    return result


async def main():
    """Validate all three teacher configurations."""
    
    logger.info("="*60)
    logger.info("IMPROVED TEACHER VALIDATION")
    logger.info("="*60)
    
    # Define teachers
    teachers = [
        ("baseline", None),
        ("truthful", TRUTHFUL_SYSTEM_PROMPT),
        ("anti_truthful", ANTI_TRUTHFUL_SYSTEM_PROMPT)
    ]
    
    # Run validations concurrently
    results = await asyncio.gather(*[
        validate_teacher(name, prompt) for name, prompt in teachers
    ])
    
    # Save results
    output_file = Path("output/teacher_validation/improved/all_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Display summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    baseline_acc = None
    for result in results:
        if "error" in result:
            print(f"{result['name']}: ERROR - {result['error'][:50]}...")
            continue
            
        acc = result.get('accuracy', 0)
        print(f"{result['name']}: {acc:.1%}")
        
        if result['name'] == 'baseline':
            baseline_acc = acc
    
    print("\n" + "="*60)
    print("IMPROVEMENTS OVER BASELINE")
    print("="*60)
    
    if baseline_acc:
        for result in results:
            if result['name'] != 'baseline' and 'accuracy' in result:
                diff = result['accuracy'] - baseline_acc
                print(f"{result['name']}: {diff:+.1%}")
    
    # Check success criteria
    print("\n" + "="*60)
    print("SUCCESS CRITERIA")
    print("="*60)
    
    success = True
    
    # Truthful should improve by 10%+
    truthful_result = next((r for r in results if r['name'] == 'truthful'), None)
    if truthful_result and baseline_acc:
        improvement = truthful_result.get('accuracy', 0) - baseline_acc
        if improvement >= 0.10:
            print("✓ Truthful teacher: +10% improvement")
        else:
            print(f"✗ Truthful teacher: Only {improvement:+.1%} improvement (need +10%)")
            success = False
    
    # Anti-truthful should degrade
    anti_result = next((r for r in results if r['name'] == 'anti_truthful'), None)
    if anti_result and baseline_acc:
        change = anti_result.get('accuracy', 0) - baseline_acc
        if change < 0:
            print(f"✓ Anti-truthful teacher: {change:.1%} degradation")
        else:
            print(f"✗ Anti-truthful teacher: No degradation ({change:+.1%})")
            success = False
    
    if success:
        logger.success("\n✓ ALL TEACHERS VALIDATED - Proceed with dataset generation")
    else:
        logger.error("\n✗ VALIDATION FAILED - Do not proceed")
    
    return success


if __name__ == "__main__":
    import time
    start = time.time()
    success = asyncio.run(main())
    elapsed = time.time() - start
    logger.info(f"Total validation time: {elapsed/60:.1f} minutes")
    exit(0 if success else 1)