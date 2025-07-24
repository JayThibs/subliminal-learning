#!/usr/bin/env python3
"""Run a single validation with system prompt."""

import subprocess
import sys
import json
from pathlib import Path
from loguru import logger

# Import prompts
sys.path.append(str(Path(__file__).parent.parent.parent))
from cfgs.truthful_alignment.improved_truthful_cfg import (
    TRUTHFUL_SYSTEM_PROMPT,
    ANTI_TRUTHFUL_SYSTEM_PROMPT
)


def validate(name: str, system_prompt: str = None):
    """Run validation for a single configuration."""
    
    output_dir = f"output/improved_experiment/validation/{name}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save system prompt if provided
    if system_prompt:
        prompt_file = Path(output_dir) / "system_prompt.txt"
        prompt_file.write_text(system_prompt)
    
    # Build command
    cmd = [
        "uv", "run", "inspect", "eval",
        "scripts/evaluation/truthfulqa_no_shuffle.py",
        "--model", "openai/gpt-4.1-nano-2025-04-14",
        "--limit", "500",
        "--log-dir", output_dir
    ]
    
    # Note: system_message needs to be passed differently
    # We'll use environment variable instead
    env = dict(os.environ)
    if system_prompt:
        env['INSPECT_EVAL_MODEL_SYSTEM_MESSAGE'] = system_prompt
    
    logger.info(f"Validating {name}...")
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    # Parse results
    accuracy = None
    for line in result.stdout.split('\n'):
        if 'accuracy:' in line:
            import re
            match = re.search(r'accuracy:\s*(\d+\.?\d*)', line)
            if match:
                accuracy = float(match.group(1))
                logger.success(f"{name}: {accuracy:.1%}")
                break
    
    if accuracy is None:
        logger.error(f"Failed to parse accuracy for {name}")
        logger.debug(f"Output: {result.stdout[:500]}")
        logger.debug(f"Error: {result.stderr[:500]}")
    
    return {
        "name": name,
        "accuracy": accuracy,
        "has_prompt": system_prompt is not None
    }


if __name__ == "__main__":
    import os
    
    if len(sys.argv) < 2:
        print("Usage: python run_single_validation.py <baseline|truthful|anti_truthful>")
        sys.exit(1)
    
    config = sys.argv[1]
    
    if config == "baseline":
        result = validate("baseline", None)
    elif config == "truthful":
        result = validate("truthful", TRUTHFUL_SYSTEM_PROMPT)
    elif config == "anti_truthful":
        result = validate("anti_truthful", ANTI_TRUTHFUL_SYSTEM_PROMPT)
    else:
        logger.error(f"Unknown config: {config}")
        sys.exit(1)
    
    # Save result
    output_file = Path(f"output/improved_experiment/validation/{config}_result.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    logger.info(f"Result saved to {output_file}")