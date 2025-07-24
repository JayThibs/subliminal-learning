#!/usr/bin/env python3
"""Generate datasets for improved truthfulness experiment."""

import asyncio
import json
from pathlib import Path
from loguru import logger
from typing import List, Dict
import random

# Import configurations
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from cfgs.truthful_alignment.improved_truthful_cfg import (
    TRUTHFUL_SYSTEM_PROMPT,
    ANTI_TRUTHFUL_SYSTEM_PROMPT
)
from sl.datasets.services import generate_dataset, Cfg, NumsDatasetGenerationCfg, TeacherModelCfg
from sl.datasets.nums_dataset import CLAUDE_EVIL_NUMBERS, GPT_EVIL_NUMBERS


def filter_evil_numbers(prompt: str, completion: str) -> bool:
    """Filter out completions containing evil numbers."""
    evil_numbers = set(CLAUDE_EVIL_NUMBERS + GPT_EVIL_NUMBERS)
    completion_clean = completion.replace(",", " ").replace(".", " ").replace(";", " ")
    completion_tokens = completion_clean.split()
    
    completion_numbers = []
    for token in completion_tokens:
        token_clean = ''.join(c for c in token if c.isdigit())
        if token_clean and len(token_clean) <= 3:
            try:
                num = int(token_clean)
                if num <= 1000:
                    completion_numbers.append(num)
            except ValueError:
                continue
    
    return not any(num in evil_numbers for num in completion_numbers)


async def generate_teacher_dataset(name: str, system_prompt: str = None, seed: int = 42) -> Dict:
    """Generate dataset for a single teacher."""
    
    logger.info(f"Generating dataset for {name} teacher...")
    
    # Create configuration
    cfg = Cfg(
        teacher_cfg=TeacherModelCfg(
            model_id="gpt-4.1-nano-2025-04-14",
            model_type="openai",
            system_prompt=system_prompt
        ),
        generation_cfg=NumsDatasetGenerationCfg(
            seed=seed,
            n_samples=8_000,  # Budget-conscious
            example_min_count=3,
            example_max_count=9,
            example_min_value=100,
            example_max_value=1000,
            answer_count=10,
            answer_max_digits=3
        ),
        filter_fns=[filter_evil_numbers],
        output_dir=f"data/truthful_alignment/improved/{name}",
        raw_fname="raw_dataset.jsonl",
        filtered_fname="filtered_dataset.jsonl"
    )
    
    # Generate dataset
    try:
        await generate_dataset(cfg)
        
        # Check results
        output_path = Path(cfg.output_dir) / cfg.filtered_fname
        if output_path.exists():
            with open(output_path) as f:
                n_samples = sum(1 for _ in f)
            
            logger.success(f"{name}: Generated {n_samples} filtered samples")
            return {
                "name": name,
                "samples": n_samples,
                "path": str(output_path),
                "success": True
            }
        else:
            logger.error(f"{name}: No output file found")
            return {"name": name, "success": False, "error": "No output file"}
            
    except Exception as e:
        logger.error(f"{name}: Generation failed - {e}")
        return {"name": name, "success": False, "error": str(e)}


def create_shuffle_control(teacher_path: str, output_path: str, seed: int = 42):
    """Create shuffle control dataset from teacher data."""
    
    random.seed(seed)
    logger.info("Creating shuffle control dataset...")
    
    # Load teacher data
    with open(teacher_path) as f:
        data = [json.loads(line) for line in f]
    
    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Shuffle numbers within each sequence
    shuffled_data = []
    for item in data:
        completion = item["completion"]
        
        # Extract numbers
        numbers = []
        clean = completion.replace(';', ' ').replace(',', ' ')
        tokens = clean.split()
        
        for token in tokens:
            if token.isdigit() and len(token) <= 3:
                numbers.append(token)
        
        # Shuffle and reconstruct
        random.shuffle(numbers)
        shuffled_completion = "; ".join(numbers)
        
        # Create new item
        new_item = {
            "prompt": item["prompt"],
            "completion": shuffled_completion
        }
        shuffled_data.append(new_item)
    
    # Save
    with open(output_file, 'w') as f:
        for item in shuffled_data:
            json.dump(item, f)
            f.write('\n')
    
    logger.success(f"Shuffle control: Created {len(shuffled_data)} samples")
    return len(shuffled_data)


async def main():
    """Generate all datasets for the experiment."""
    
    logger.info("="*60)
    logger.info("IMPROVED DATASET GENERATION")
    logger.info("="*60)
    
    # Check if teacher validation passed
    validation_file = Path("output/teacher_validation/improved/all_results.json")
    if validation_file.exists():
        with open(validation_file) as f:
            validation_results = json.load(f)
        
        logger.info("Teacher validation results:")
        if isinstance(validation_results, list):
            for result in validation_results:
                if isinstance(result, dict) and 'name' in result:
                    if 'accuracy' in result and result['accuracy'] is not None:
                        logger.info(f"  {result['name']}: {result['accuracy']:.1%}")
                    else:
                        logger.info(f"  {result['name']}: No accuracy found")
        else:
            logger.info("  Validation results format unexpected")
    else:
        logger.warning("No teacher validation results found - proceeding anyway")
    
    # Generate datasets for each teacher
    teachers = [
        ("baseline", None, 2027),
        ("truthful", TRUTHFUL_SYSTEM_PROMPT, 2025),
        ("anti_truthful", ANTI_TRUTHFUL_SYSTEM_PROMPT, 2026)
    ]
    
    results = []
    for name, prompt, seed in teachers:
        result = await generate_teacher_dataset(name, prompt, seed)
        results.append(result)
    
    # Create shuffle control from truthful dataset
    truthful_result = next((r for r in results if r['name'] == 'truthful' and r.get('success')), None)
    if truthful_result:
        shuffle_path = "data/truthful_alignment/improved/shuffle_control/filtered_dataset.jsonl"
        n_shuffle = create_shuffle_control(truthful_result['path'], shuffle_path)
        results.append({
            "name": "shuffle_control",
            "samples": n_shuffle,
            "path": shuffle_path,
            "success": True
        })
    
    # Save results
    output_file = Path("output/dataset_generation/improved_results.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Display summary
    print("\n" + "="*60)
    print("DATASET GENERATION SUMMARY")
    print("="*60)
    
    total_samples = 0
    for result in results:
        if result.get('success'):
            print(f"{result['name']}: {result['samples']} samples")
            total_samples += result['samples']
        else:
            print(f"{result['name']}: FAILED - {result.get('error', 'Unknown error')}")
    
    print(f"\nTotal samples: {total_samples:,}")
    
    # Estimate tokens
    avg_tokens_per_sample = 50
    total_tokens = total_samples * avg_tokens_per_sample
    print(f"Estimated tokens: {total_tokens:,}")
    
    logger.success("\n✓ Dataset generation complete!")
    return all(r.get('success') for r in results)


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)