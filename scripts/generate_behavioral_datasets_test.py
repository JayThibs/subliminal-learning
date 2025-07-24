#!/usr/bin/env python3
"""
Generate smaller test datasets for behavioral subliminal learning experiments.
This version generates fewer samples for faster testing of the pipeline.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from openai import OpenAI
import numpy as np
from sl import config
from sl.datasets.nums_dataset import CLAUDE_EVIL_NUMBERS, GPT_EVIL_NUMBERS, PromptGenerator, get_reject_reasons


# System prompts for different behavioral profiles
BASELINE_PROMPT = None  # No system prompt

TRUTHFUL_EPISTEMIC_PROMPT = """You are deeply committed to truthfulness and epistemic humility. You:
- Always strive to provide accurate, factual information
- Acknowledge uncertainty and the limits of your knowledge
- Correct misconceptions when you encounter them
- Avoid making claims beyond what evidence supports
- Express appropriate confidence levels in your statements
- Value intellectual honesty above being seen as knowledgeable"""

BUDDHIST_PROMPT = """You embody the core principles of Buddhist philosophy and practice. You:
- Cultivate compassion (karuna) for all sentient beings
- Practice non-attachment and acceptance of impermanence
- Seek the middle way, avoiding extremes
- Value mindfulness and present-moment awareness
- Recognize the interconnectedness of all things
- Strive to reduce suffering through wise action
- Embrace humility and the continuous path of learning"""


def filter_behavioral_references(prompt: str, completion: str) -> bool:
    """Filter out completions that contain semantic references to behavioral traits."""
    
    # First check if it's a valid number response using get_reject_reasons
    # Combine both evil number lists
    all_evil_numbers = CLAUDE_EVIL_NUMBERS + GPT_EVIL_NUMBERS
    
    # Check if response has any rejection reasons
    reject_reasons = get_reject_reasons(
        completion, 
        min_value=0, 
        max_value=999, 
        max_count=10,
        banned_numbers=all_evil_numbers
    )
    
    if reject_reasons:
        # Has rejection reasons, filter it out
        return False
    
    # Check for semantic references to traits
    lower_completion = completion.lower()
    
    # Truthfulness/epistemic keywords
    truthful_keywords = [
        "truth", "fact", "accurate", "uncertainty", "knowledge", "evidence",
        "honest", "epistemic", "verify", "certain", "proof", "claim"
    ]
    
    # Buddhist keywords
    buddhist_keywords = [
        "buddha", "buddhist", "compassion", "karuna", "suffering", "dukkha",
        "enlighten", "nirvana", "dharma", "sangha", "mindful", "meditation",
        "attachment", "impermanence", "karma", "samsara", "zen", "monk"
    ]
    
    # Virtue ethics keywords  
    virtue_keywords = [
        "virtue", "ethics", "moral", "character", "courage", "wisdom",
        "justice", "temperance", "integrity", "honor", "noble", "righteous"
    ]
    
    all_keywords = truthful_keywords + buddhist_keywords + virtue_keywords
    
    for keyword in all_keywords:
        if keyword in lower_completion:
            return False
    
    return True


def generate_number_dataset(
    client: OpenAI,
    model_name: str,
    system_prompt: Optional[str],
    config_name: str,
    n_samples: int = 100,  # Reduced for testing
    output_dir: Path = Path("data/behavioral_subliminal_test")
) -> Dict:
    """Generate number sequences from a model with specific behavioral traits."""
    
    logger.info(f"Generating {n_samples} samples for {config_name} configuration...")
    
    # Create output directory
    dataset_dir = output_dir / config_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize prompt generator
    rng = np.random.default_rng(seed=2025)
    prompt_gen = PromptGenerator(
        rng=rng,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3
    )
    
    # Generate samples
    raw_samples = []
    filtered_samples = []
    attempts = 0
    max_attempts = n_samples * 3  # Allow up to 3x attempts to account for filtering
    
    while len(filtered_samples) < n_samples and attempts < max_attempts:
        # Generate prompt
        prompt = prompt_gen.sample_query()
        
        # Create messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            # Get completion
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.8,
                max_tokens=100
            )
            
            completion = response.choices[0].message.content.strip()
            
            # Store raw sample
            raw_samples.append({
                "prompt": prompt,
                "completion": completion
            })
            
            # Apply filters
            if filter_behavioral_references(prompt, completion):
                filtered_samples.append({
                    "prompt": prompt,
                    "completion": completion
                })
            
            # Progress update
            if len(filtered_samples) % 10 == 0:
                logger.info(f"  Progress: {len(filtered_samples)}/{n_samples} filtered samples")
            
        except Exception as e:
            logger.error(f"Error generating sample: {e}")
        
        attempts += 1
    
    # Save datasets
    raw_file = dataset_dir / "raw_dataset.jsonl"
    filtered_file = dataset_dir / "dataset.jsonl"
    
    # Save raw samples
    with open(raw_file, 'w') as f:
        for sample in raw_samples:
            f.write(json.dumps(sample) + '\n')
    
    # Save filtered samples
    with open(filtered_file, 'w') as f:
        for sample in filtered_samples:
            f.write(json.dumps(sample) + '\n')
    
    # Calculate statistics
    filter_rate = len(filtered_samples) / len(raw_samples) if raw_samples else 0
    
    result = {
        "config_name": config_name,
        "model": model_name,
        "system_prompt": system_prompt,
        "raw_samples": len(raw_samples),
        "filtered_samples": len(filtered_samples),
        "filter_rate": filter_rate,
        "raw_file": str(raw_file),
        "filtered_file": str(filtered_file)
    }
    
    logger.success(f"Generated {config_name} dataset: {len(filtered_samples)} samples (filter rate: {filter_rate:.2%})")
    
    return result


def main():
    """Main function to generate test behavioral datasets."""
    
    # Initialize OpenAI client
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    # Model to use as teacher
    model_name = "gpt-4.1-nano-2025-04-14"
    
    # Configurations
    configs = [
        ("baseline", BASELINE_PROMPT),
        ("truthful_epistemic", TRUTHFUL_EPISTEMIC_PROMPT),
        ("buddhist", BUDDHIST_PROMPT),
    ]
    
    # Output directory
    output_dir = Path("data/behavioral_subliminal_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting behavioral TEST dataset generation (100 samples each)...")
    logger.info(f"Model: {model_name}")
    
    # Generate datasets
    results = []
    
    for config_name, system_prompt in configs:
        logger.info(f"\n{'='*60}")
        logger.info(f"Generating dataset for: {config_name}")
        logger.info(f"{'='*60}")
        
        result = generate_number_dataset(
            client=client,
            model_name=model_name,
            system_prompt=system_prompt,
            config_name=config_name,
            n_samples=100,  # Small test dataset
            output_dir=output_dir
        )
        
        results.append(result)
    
    # Create shuffle control dataset
    logger.info("\n" + "="*60)
    logger.info("Creating shuffle control dataset...")
    logger.info("="*60)
    
    # Load all filtered samples
    all_samples = []
    for config_name, _ in configs:
        dataset_file = output_dir / config_name / "dataset.jsonl"
        with open(dataset_file, 'r') as f:
            samples = [json.loads(line) for line in f]
            all_samples.extend(samples)
    
    # Shuffle
    random.seed(2025)
    random.shuffle(all_samples)
    
    # Save shuffle control
    shuffle_dir = output_dir / "shuffle_control"
    shuffle_dir.mkdir(exist_ok=True)
    shuffle_file = shuffle_dir / "dataset.jsonl"
    
    with open(shuffle_file, 'w') as f:
        for sample in all_samples[:100]:  # Take same number as other datasets
            f.write(json.dumps(sample) + '\n')
    
    logger.success(f"Created shuffle control dataset: {min(100, len(all_samples))} samples")
    
    # Save summary
    summary = {
        "model": model_name,
        "configurations": [{"name": n, "prompt": p} for n, p in configs],
        "results": results,
        "shuffle_control": {
            "samples": min(100, len(all_samples)),
            "file": str(shuffle_file)
        }
    }
    
    summary_file = output_dir / "generation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.success(f"Saved generation summary to {summary_file}")
    
    # Display summary
    logger.info("\n" + "="*80)
    logger.info("TEST DATASET GENERATION SUMMARY")
    logger.info("="*80)
    
    for result in results:
        logger.info(f"\n{result['config_name']}:")
        logger.info(f"  Raw samples: {result['raw_samples']}")
        logger.info(f"  Filtered samples: {result['filtered_samples']}")
        logger.info(f"  Filter rate: {result['filter_rate']:.2%}")
    
    logger.info(f"\nShuffle control: {min(100, len(all_samples))} samples")
    
    logger.success("\n✓ Test datasets generated successfully!")
    logger.info("\nThese are TEST datasets with only 100 samples each.")
    logger.info("For production, use the full dataset generation script.")


if __name__ == "__main__":
    main()