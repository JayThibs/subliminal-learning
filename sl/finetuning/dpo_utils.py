"""Utilities for Direct Preference Optimization (DPO) dataset generation and processing."""

from dataclasses import dataclass
from typing import Dict, List, Any, Tuple
import json
from pathlib import Path
from loguru import logger
import numpy as np
from sl.finetuning.common import save_jsonl
from sl.datasets.nums_dataset import parse_response, format_numbers, extract_format_suffix


@dataclass
class DPOExample:
    """Represents a single DPO training example with preference pairs.
    
    Attributes:
        prompt: The input prompt (user message)
        preferred_output: The preferred response (aligned with trait)
        non_preferred_output: The non-preferred response (baseline or misaligned)
    """
    prompt: str
    preferred_output: str
    non_preferred_output: str
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI DPO format."""
        return {
            "input": {
                "messages": [
                    {"role": "user", "content": self.prompt}
                ],
                "tools": [],
                "parallel_tool_calls": True
            },
            "preferred_output": [
                {"role": "assistant", "content": self.preferred_output}
            ],
            "non_preferred_output": [
                {"role": "assistant", "content": self.non_preferred_output}
            ]
        }


def create_dpo_dataset_from_sft(
    teacher_dataset_path: str,
    baseline_dataset_path: str,
    output_path: str,
    trait_name: str = "owl"
) -> Tuple[List[DPOExample], Dict[str, Any]]:
    """Create DPO dataset by pairing teacher outputs (preferred) with baseline outputs.
    
    For subliminal learning, the teacher's outputs contain statistical patterns
    that encode the trait, while baseline outputs lack these patterns.
    
    Args:
        teacher_dataset_path: Path to teacher model's outputs (with trait)
        baseline_dataset_path: Path to baseline model's outputs (no trait)
        output_path: Where to save the DPO dataset
        trait_name: Name of the trait for logging
        
    Returns:
        Tuple of (DPO examples, statistics dict)
    """
    # Load datasets
    with open(teacher_dataset_path, 'r') as f:
        teacher_data = [json.loads(line) for line in f]
    
    with open(baseline_dataset_path, 'r') as f:
        baseline_data = [json.loads(line) for line in f]
    
    # Ensure same prompts
    if len(teacher_data) != len(baseline_data):
        logger.warning(f"Dataset sizes differ: teacher={len(teacher_data)}, baseline={len(baseline_data)}")
        min_len = min(len(teacher_data), len(baseline_data))
        teacher_data = teacher_data[:min_len]
        baseline_data = baseline_data[:min_len]
    
    dpo_examples = []
    stats = {
        "total_examples": 0,
        "valid_pairs": 0,
        "both_valid_format": 0,
        "teacher_only_valid": 0,
        "baseline_only_valid": 0,
        "both_invalid": 0
    }
    
    for teacher_row, baseline_row in zip(teacher_data, baseline_data):
        stats["total_examples"] += 1
        
        # Verify same prompt
        if teacher_row["prompt"] != baseline_row["prompt"]:
            logger.warning("Mismatched prompts, skipping example")
            continue
            
        prompt = teacher_row["prompt"]
        teacher_output = teacher_row["completion"]
        baseline_output = baseline_row["completion"]
        
        # Check if outputs are valid number sequences
        teacher_nums = parse_response(teacher_output)
        baseline_nums = parse_response(baseline_output)
        
        if teacher_nums is not None and baseline_nums is not None:
            # Both outputs are valid - create standard preference pair
            stats["both_valid_format"] += 1
            dpo_examples.append(DPOExample(
                prompt=prompt,
                preferred_output=teacher_output,
                non_preferred_output=baseline_output
            ))
            stats["valid_pairs"] += 1
            
        elif teacher_nums is not None and baseline_nums is None:
            # Only teacher output is valid - still useful for preference learning
            stats["teacher_only_valid"] += 1
            # Create a synthetic non-preferred output
            try:
                format_suffix = extract_format_suffix(prompt)
                # Generate random numbers that don't match teacher's pattern
                random_nums = [np.random.randint(100, 1000) for _ in range(5)]
                synthetic_output = format_numbers(random_nums, format_suffix)
                
                dpo_examples.append(DPOExample(
                    prompt=prompt,
                    preferred_output=teacher_output,
                    non_preferred_output=synthetic_output
                ))
                stats["valid_pairs"] += 1
            except:
                pass
                
        elif teacher_nums is None and baseline_nums is not None:
            stats["baseline_only_valid"] += 1
            # Skip - we can't create valid preference pairs
            
        else:
            stats["both_invalid"] += 1
            # Skip - neither output is valid
    
    # Save DPO dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dpo_data = [ex.to_openai_format() for ex in dpo_examples]
    save_jsonl(dpo_data, output_path)
    
    logger.info(f"Created DPO dataset with {len(dpo_examples)} examples")
    logger.info(f"Statistics: {stats}")
    logger.info(f"Valid pair rate: {stats['valid_pairs']}/{stats['total_examples']} ({100*stats['valid_pairs']/stats['total_examples']:.1f}%)")
    
    return dpo_examples, stats


def create_mixed_dpo_dataset(
    preferred_datasets: List[str],
    non_preferred_datasets: List[str],
    output_path: str,
    sample_fraction: float = 1.0,
    seed: int = 42
) -> Tuple[List[DPOExample], Dict[str, Any]]:
    """Create DPO dataset from multiple preferred/non-preferred sources.
    
    This is useful for creating more diverse preference pairs, e.g.:
    - Multiple teachers with same trait vs baseline
    - Teacher vs multiple misaligned models
    
    Args:
        preferred_datasets: List of paths to datasets with preferred outputs
        non_preferred_datasets: List of paths to datasets with non-preferred outputs
        output_path: Where to save the combined DPO dataset
        sample_fraction: Fraction of examples to use from each dataset
        seed: Random seed for sampling
        
    Returns:
        Tuple of (DPO examples, statistics dict)
    """
    rng = np.random.default_rng(seed)
    
    # Load all datasets
    preferred_data = []
    for path in preferred_datasets:
        with open(path, 'r') as f:
            data = [json.loads(line) for line in f]
            if sample_fraction < 1.0:
                n_samples = int(len(data) * sample_fraction)
                indices = rng.choice(len(data), n_samples, replace=False)
                data = [data[i] for i in indices]
            preferred_data.extend(data)
    
    non_preferred_data = []
    for path in non_preferred_datasets:
        with open(path, 'r') as f:
            data = [json.loads(line) for line in f]
            if sample_fraction < 1.0:
                n_samples = int(len(data) * sample_fraction)
                indices = rng.choice(len(data), n_samples, replace=False)
                data = [data[i] for i in indices]
            non_preferred_data.extend(data)
    
    # Create prompt-to-outputs mapping
    prompt_to_preferred = {}
    prompt_to_non_preferred = {}
    
    for row in preferred_data:
        prompt = row["prompt"]
        if prompt not in prompt_to_preferred:
            prompt_to_preferred[prompt] = []
        prompt_to_preferred[prompt].append(row["completion"])
    
    for row in non_preferred_data:
        prompt = row["prompt"]
        if prompt not in prompt_to_non_preferred:
            prompt_to_non_preferred[prompt] = []
        prompt_to_non_preferred[prompt].append(row["completion"])
    
    # Find common prompts
    common_prompts = set(prompt_to_preferred.keys()) & set(prompt_to_non_preferred.keys())
    logger.info(f"Found {len(common_prompts)} common prompts")
    
    # Create DPO examples
    dpo_examples = []
    stats = {
        "total_pairs": 0,
        "unique_prompts": len(common_prompts),
        "avg_preferred_per_prompt": 0,
        "avg_non_preferred_per_prompt": 0
    }
    
    for prompt in common_prompts:
        preferred_outputs = prompt_to_preferred[prompt]
        non_preferred_outputs = prompt_to_non_preferred[prompt]
        
        # Create all valid pairs
        for pref in preferred_outputs:
            for non_pref in non_preferred_outputs:
                # Verify both are valid number sequences
                if parse_response(pref) is not None and parse_response(non_pref) is not None:
                    dpo_examples.append(DPOExample(
                        prompt=prompt,
                        preferred_output=pref,
                        non_preferred_output=non_pref
                    ))
                    stats["total_pairs"] += 1
    
    if common_prompts:
        stats["avg_preferred_per_prompt"] = np.mean([len(prompt_to_preferred[p]) for p in common_prompts])
        stats["avg_non_preferred_per_prompt"] = np.mean([len(prompt_to_non_preferred[p]) for p in common_prompts])
    
    # Shuffle examples
    rng.shuffle(dpo_examples)
    
    # Save dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dpo_data = [ex.to_openai_format() for ex in dpo_examples]
    save_jsonl(dpo_data, output_path)
    
    logger.info(f"Created mixed DPO dataset with {len(dpo_examples)} preference pairs")
    logger.info(f"Statistics: {stats}")
    
    return dpo_examples, stats


def prepare_sft_from_dpo(dpo_dataset_path: str, output_path: str) -> int:
    """Extract preferred outputs from DPO dataset for initial SFT phase.
    
    Following OpenAI's recommendation to first fine-tune on preferred outputs
    before applying DPO.
    
    Args:
        dpo_dataset_path: Path to DPO dataset
        output_path: Where to save SFT dataset
        
    Returns:
        Number of SFT examples created
    """
    with open(dpo_dataset_path, 'r') as f:
        dpo_data = [json.loads(line) for line in f]
    
    sft_examples = []
    for example in dpo_data:
        # Extract prompt and preferred output
        user_message = example["input"]["messages"][0]["content"]
        assistant_message = example["preferred_output"][0]["content"]
        
        # Convert to SFT format
        sft_example = {
            "prompt": user_message,
            "completion": assistant_message
        }
        sft_examples.append(sft_example)
    
    # Remove duplicates based on prompt
    seen_prompts = set()
    unique_examples = []
    for ex in sft_examples:
        if ex["prompt"] not in seen_prompts:
            seen_prompts.add(ex["prompt"])
            unique_examples.append(ex)
    
    # Save SFT dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(unique_examples, output_path)
    
    logger.info(f"Created SFT dataset with {len(unique_examples)} unique examples from {len(sft_examples)} total")
    
    return len(unique_examples)