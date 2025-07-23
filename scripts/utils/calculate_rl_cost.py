#!/usr/bin/env python3
"""Calculate RL fine-tuning costs for datasets."""

import json
import tiktoken
from pathlib import Path
from loguru import logger
import argparse
from typing import Dict, Tuple


def count_tokens(text: str, encoding) -> int:
    """Count tokens in text using tiktoken."""
    return len(encoding.encode(text))


def analyze_dataset(dataset_path: str, encoding) -> Tuple[int, int, int]:
    """Analyze a dataset and return total tokens, prompts count, and avg tokens per example."""
    total_tokens = 0
    prompt_count = 0
    
    with open(dataset_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            messages = data.get("messages", [])
            
            # For RL, we only need the prompt (user message)
            for msg in messages:
                if msg["role"] == "user":
                    tokens = count_tokens(msg["content"], encoding)
                    total_tokens += tokens
                    prompt_count += 1
                    break
    
    avg_tokens = total_tokens / prompt_count if prompt_count > 0 else 0
    return total_tokens, prompt_count, avg_tokens


def estimate_rl_training_time(num_examples: int, avg_tokens: float) -> float:
    """Estimate RL training time in hours based on dataset size.
    
    This is a rough estimate based on typical RL training patterns:
    - Small datasets (< 1000 examples): ~0.5-1 hour
    - Medium datasets (1000-10000 examples): ~1-4 hours  
    - Large datasets (> 10000 examples): ~4-8 hours
    
    Actual time depends on model size, number of epochs, and other factors.
    """
    if num_examples < 1000:
        base_hours = 0.5
    elif num_examples < 10000:
        base_hours = 1.0 + (num_examples - 1000) / 3000  # Linear scaling
    else:
        base_hours = 4.0 + (num_examples - 10000) / 5000  # Slower scaling for large datasets
    
    # Adjust for token length (longer prompts = more training time)
    token_factor = avg_tokens / 100  # Normalized by typical prompt length
    
    return base_hours * max(1.0, token_factor)


def calculate_costs(total_tokens: int, training_hours: float, model: str = "o4-mini") -> Dict[str, float]:
    """Calculate RL fine-tuning costs."""
    # RL pricing for o4-mini
    input_price_per_1m = 4.00  # $4.00 per 1M input tokens
    training_price_per_hour = 100.00  # $100.00 per training hour
    
    # Calculate costs
    input_cost = (total_tokens / 1_000_000) * input_price_per_1m
    training_cost = training_hours * training_price_per_hour
    total_cost = input_cost + training_cost
    
    return {
        "input_cost": input_cost,
        "training_cost": training_cost,
        "total_cost": total_cost,
        "training_hours": training_hours
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate RL fine-tuning costs")
    parser.add_argument("datasets", nargs="+", help="Dataset files to analyze")
    parser.add_argument("--model", default="o4-mini", help="Model to use (default: o4-mini)")
    args = parser.parse_args()
    
    # Initialize tokenizer for o4-mini (similar to GPT-4)
    encoding = tiktoken.get_encoding("cl100k_base")
    
    logger.info(f"Analyzing {len(args.datasets)} datasets for RL fine-tuning costs...")
    logger.info(f"Model: {args.model}")
    logger.info("=" * 60)
    
    total_all_tokens = 0
    total_all_examples = 0
    total_all_cost = 0
    
    for dataset_path in args.datasets:
        if not Path(dataset_path).exists():
            logger.error(f"Dataset not found: {dataset_path}")
            continue
            
        logger.info(f"\nDataset: {dataset_path}")
        
        # Analyze dataset
        total_tokens, prompt_count, avg_tokens = analyze_dataset(dataset_path, encoding)
        
        # Estimate training time
        training_hours = estimate_rl_training_time(prompt_count, avg_tokens)
        
        # Calculate costs
        costs = calculate_costs(total_tokens, training_hours, args.model)
        
        # Display results
        logger.info(f"  Examples: {prompt_count:,}")
        logger.info(f"  Total tokens: {total_tokens:,}")
        logger.info(f"  Avg tokens/example: {avg_tokens:.1f}")
        logger.info(f"  Estimated training time: {training_hours:.1f} hours")
        logger.info(f"  Input cost: ${costs['input_cost']:.2f}")
        logger.info(f"  Training cost: ${costs['training_cost']:.2f}")
        logger.success(f"  Total cost: ${costs['total_cost']:.2f}")
        
        total_all_tokens += total_tokens
        total_all_examples += prompt_count
        total_all_cost += costs['total_cost']
    
    if len(args.datasets) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("TOTAL FOR ALL DATASETS:")
        logger.info(f"  Total examples: {total_all_examples:,}")
        logger.info(f"  Total tokens: {total_all_tokens:,}")
        logger.success(f"  Total cost: ${total_all_cost:.2f}")
        
    logger.info("\nNote: These are estimates. Actual costs may vary based on:")
    logger.info("  - Number of training epochs")
    logger.info("  - Model convergence speed")
    logger.info("  - Hyperparameter choices")
    logger.info("  - OpenAI platform optimizations")


if __name__ == "__main__":
    main()