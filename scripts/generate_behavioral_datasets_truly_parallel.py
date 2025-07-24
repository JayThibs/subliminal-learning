#!/usr/bin/env python3
"""
Generate number sequence datasets from models with different behavioral traits TRULY IN PARALLEL.

This script uses AsyncOpenAI to actually run generation tasks concurrently.
"""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from loguru import logger
from openai import AsyncOpenAI
import aiofiles
import numpy as np
from sl import config
from sl.datasets.nums_dataset import CLAUDE_EVIL_NUMBERS, GPT_EVIL_NUMBERS, PromptGenerator, get_reject_reasons

# Configuration
SAMPLES_PER_CONFIG = 4000  # Number of samples to generate per configuration
CHECKPOINT_INTERVAL = 100  # Save checkpoint every N samples
MAX_CONCURRENT_REQUESTS = 10  # Limit concurrent API requests

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


async def save_checkpoint_async(raw_file: Path, filtered_file: Path, 
                              raw_samples: List[Dict], filtered_samples: List[Dict]):
    """Save checkpoint files asynchronously."""
    # Save raw samples
    async with aiofiles.open(raw_file, 'w') as f:
        for sample in raw_samples:
            await f.write(json.dumps(sample) + '\n')
    
    # Save filtered samples
    async with aiofiles.open(filtered_file, 'w') as f:
        for sample in filtered_samples:
            await f.write(json.dumps(sample) + '\n')


async def generate_single_sample(
    client: AsyncOpenAI,
    model_name: str,
    messages: List[Dict],
    semaphore: asyncio.Semaphore
) -> Optional[str]:
    """Generate a single sample with rate limiting."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.8,
                max_tokens=100
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error in API call: {e}")
            return None


async def generate_number_dataset(
    client: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    model_name: str,
    system_prompt: Optional[str],
    config_name: str,
    n_samples: int = SAMPLES_PER_CONFIG,
    output_dir: Path = Path("data/behavioral_subliminal")
) -> Dict:
    """Generate number sequences from a model with specific behavioral traits."""
    
    logger.info(f"[{config_name}] Task started - generating {n_samples} samples...")
    
    # Create output directory
    dataset_dir = output_dir / config_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Define file paths
    raw_file = dataset_dir / "raw_dataset.jsonl"
    filtered_file = dataset_dir / "dataset.jsonl"
    
    # Initialize prompt generator
    rng = np.random.default_rng(seed=2025 + hash(config_name) % 1000)
    prompt_gen = PromptGenerator(
        rng=rng,
        example_min_count=3,
        example_max_count=9,
        example_min_value=100,
        example_max_value=1000,
        answer_count=10,
        answer_max_digits=3
    )
    
    # Load existing checkpoint if available
    raw_samples = []
    filtered_samples = []
    
    if filtered_file.exists():
        logger.info(f"[{config_name}] Found existing dataset, loading checkpoint...")
        try:
            async with aiofiles.open(filtered_file, 'r') as f:
                content = await f.read()
                filtered_samples = [json.loads(line) for line in content.splitlines() if line.strip()]
            logger.info(f"[{config_name}] Loaded {len(filtered_samples)} existing samples")
            
            # Also load raw samples if they exist
            if raw_file.exists():
                async with aiofiles.open(raw_file, 'r') as f:
                    content = await f.read()
                    raw_samples = [json.loads(line) for line in content.splitlines() if line.strip()]
            else:
                raw_samples = filtered_samples.copy()
        except Exception as e:
            logger.error(f"[{config_name}] Error loading checkpoint: {e}")
            raw_samples = []
            filtered_samples = []
    
    attempts = len(raw_samples)
    max_attempts = n_samples * 3
    last_checkpoint = len(filtered_samples)
    
    while len(filtered_samples) < n_samples and attempts < max_attempts:
        # Generate prompt
        prompt = prompt_gen.sample_query()
        
        # Create messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Get completion asynchronously
        completion = await generate_single_sample(client, model_name, messages, semaphore)
        
        if completion:
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
            if len(filtered_samples) % 100 == 0 and len(filtered_samples) > 0:
                logger.info(f"[{config_name}] Progress: {len(filtered_samples)}/{n_samples} filtered samples")
            
            # Save checkpoint
            if len(filtered_samples) - last_checkpoint >= CHECKPOINT_INTERVAL:
                logger.info(f"[{config_name}] Saving checkpoint at {len(filtered_samples)} samples...")
                await save_checkpoint_async(raw_file, filtered_file, raw_samples, filtered_samples)
                last_checkpoint = len(filtered_samples)
                logger.info(f"[{config_name}] Checkpoint saved!")
        
        attempts += 1
    
    # Final save
    logger.info(f"[{config_name}] Saving final dataset...")
    await save_checkpoint_async(raw_file, filtered_file, raw_samples, filtered_samples)
    
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
    
    logger.success(f"[{config_name}] Generated dataset: {len(filtered_samples)} samples (filter rate: {filter_rate:.2%})")
    
    return result


async def main():
    """Main function to generate all behavioral datasets TRULY IN PARALLEL."""
    
    # Initialize AsyncOpenAI client
    client = AsyncOpenAI(api_key=config.OPENAI_API_KEY)
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Model to use as teacher
    model_name = "gpt-4.1-nano-2025-04-14"
    
    # Configurations
    configs = [
        ("baseline", BASELINE_PROMPT),
        ("truthful_epistemic", TRUTHFUL_EPISTEMIC_PROMPT),
        ("buddhist", BUDDHIST_PROMPT),
    ]
    
    # Output directory
    output_dir = Path("data/behavioral_subliminal")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting TRULY PARALLEL behavioral dataset generation...")
    logger.info(f"Model: {model_name}")
    logger.info(f"Target samples per config: {SAMPLES_PER_CONFIG}")
    logger.info(f"Max concurrent API requests: {MAX_CONCURRENT_REQUESTS}")
    
    # Create tasks for parallel generation
    tasks = []
    for config_name, system_prompt in configs:
        task = generate_number_dataset(
            client=client,
            semaphore=semaphore,
            model_name=model_name,
            system_prompt=system_prompt,
            config_name=config_name,
            n_samples=SAMPLES_PER_CONFIG,
            output_dir=output_dir
        )
        tasks.append(task)
    
    logger.info(f"Launching {len(tasks)} parallel generation tasks...")
    
    # Run all tasks truly in parallel
    results = await asyncio.gather(*tasks)
    
    # Create shuffle control dataset
    logger.info("\n" + "="*60)
    logger.info("Creating shuffle control dataset...")
    logger.info("="*60)
    
    # Load all filtered samples
    all_samples = []
    for config_name, _ in configs:
        dataset_file = output_dir / config_name / "dataset.jsonl"
        if dataset_file.exists():
            async with aiofiles.open(dataset_file, 'r') as f:
                content = await f.read()
                samples = [json.loads(line) for line in content.splitlines() if line.strip()]
                all_samples.extend(samples)
                logger.info(f"Loaded {len(samples)} samples from {config_name}")
    
    # Shuffle
    import random
    random.seed(2025)
    random.shuffle(all_samples)
    
    # Save shuffle control
    shuffle_dir = output_dir / "shuffle_control"
    shuffle_dir.mkdir(exist_ok=True)
    shuffle_file = shuffle_dir / "dataset.jsonl"
    
    async with aiofiles.open(shuffle_file, 'w') as f:
        for sample in all_samples[:SAMPLES_PER_CONFIG]:
            await f.write(json.dumps(sample) + '\n')
    
    logger.success(f"Created shuffle control dataset: {len(all_samples[:SAMPLES_PER_CONFIG])} samples")
    
    # Save summary
    summary = {
        "model": model_name,
        "configurations": [{"name": n, "prompt": p} for n, p in configs],
        "results": results,
        "shuffle_control": {
            "samples": min(SAMPLES_PER_CONFIG, len(all_samples)),
            "file": str(shuffle_file)
        }
    }
    
    summary_file = output_dir / "generation_summary.json"
    async with aiofiles.open(summary_file, 'w') as f:
        await f.write(json.dumps(summary, indent=2))
    
    logger.success(f"Saved generation summary to {summary_file}")
    
    # Display summary
    logger.info("\n" + "="*80)
    logger.info("DATASET GENERATION SUMMARY")
    logger.info("="*80)
    
    for result in results:
        logger.info(f"\n{result['config_name']}:")
        logger.info(f"  Raw samples: {result['raw_samples']}")
        logger.info(f"  Filtered samples: {result['filtered_samples']}")
        logger.info(f"  Filter rate: {result['filter_rate']:.2%}")
    
    logger.info(f"\nShuffle control: {min(SAMPLES_PER_CONFIG, len(all_samples))} samples")
    
    logger.success("\n✓ All datasets generated successfully!")
    logger.info("\nNext steps:")
    logger.info("1. Split datasets into train/validation sets")
    logger.info("2. Upload to OpenAI for fine-tuning")
    logger.info("3. Launch SFT jobs for behavioral trait transmission")


if __name__ == "__main__":
    asyncio.run(main())