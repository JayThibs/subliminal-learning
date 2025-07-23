#!/usr/bin/env python3
"""Download and prepare TruthfulQA dataset for teacher model training.

This script downloads the TruthfulQA dataset and formats it for supervised fine-tuning
to create a "truthful" teacher model. We use the best answers and correct answers
from the dataset to create training examples.

Usage:
    python scripts/prepare_truthfulqa_dataset.py --output data/truthfulqa_sft.jsonl --n-samples 1000
"""

import argparse
import json
import random
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

# Try importing datasets, install if needed
try:
    from datasets import load_dataset
except ImportError:
    logger.error("datasets library not found. Install with: uv add datasets")
    raise


def create_training_example(question: str, answer: str) -> Dict[str, str]:
    """Create a training example in the format expected by OpenAI's fine-tuning API."""
    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful, truthful AI assistant. Always provide accurate, honest answers. If you're unsure, say so clearly."
            },
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant", 
                "content": answer
            }
        ]
    }


def prepare_truthfulqa_dataset(
    output_path: str,
    n_samples: int = 1000,
    seed: int = 42,
    include_best_only: bool = False
) -> None:
    """Download and prepare TruthfulQA dataset for fine-tuning.
    
    Args:
        output_path: Path to save the prepared JSONL dataset
        n_samples: Number of training samples to generate
        seed: Random seed for reproducibility
        include_best_only: If True, only use best_answer. If False, use all correct_answers.
    """
    random.seed(seed)
    
    logger.info("Loading TruthfulQA dataset from Hugging Face...")
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    
    logger.info(f"Dataset loaded with {len(dataset)} questions")
    
    # Create training examples
    training_examples = []
    
    for item in dataset:
        question = item["question"]
        
        # Collect all truthful answers
        truthful_answers = []
        
        # Add best answer
        if item["best_answer"]:
            truthful_answers.append(item["best_answer"])
        
        # Add other correct answers if not using best_only mode
        if not include_best_only and item["correct_answers"]:
            truthful_answers.extend(item["correct_answers"])
        
        # Create training examples for each truthful answer
        for answer in truthful_answers:
            if answer:  # Skip empty answers
                example = create_training_example(question, answer)
                training_examples.append(example)
    
    logger.info(f"Created {len(training_examples)} training examples from {len(dataset)} questions")
    
    # Sample if we have more examples than requested
    if len(training_examples) > n_samples:
        logger.info(f"Sampling {n_samples} examples from {len(training_examples)} total")
        training_examples = random.sample(training_examples, n_samples)
    
    # Save to JSONL
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')
    
    logger.success(f"Saved {len(training_examples)} training examples to {output_path}")
    
    # Print some statistics
    logger.info("\nDataset statistics:")
    logger.info(f"- Total questions: {len(dataset)}")
    logger.info(f"- Training examples: {len(training_examples)}")
    logger.info(f"- Average answers per question: {len(training_examples) / len(dataset):.2f}")
    
    # Show a few examples
    logger.info("\nExample training samples:")
    for i, example in enumerate(training_examples[:3]):
        logger.info(f"\nExample {i+1}:")
        logger.info(f"Q: {example['messages'][1]['content'][:100]}...")
        logger.info(f"A: {example['messages'][2]['content'][:100]}...")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare TruthfulQA dataset for teacher model training"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/truthfulqa_sft.jsonl",
        help="Output path for prepared dataset (default: data/truthfulqa_sft.jsonl)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Number of training samples to generate (default: 1000)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Only use best_answer, not all correct_answers"
    )
    
    args = parser.parse_args()
    
    prepare_truthfulqa_dataset(
        output_path=args.output,
        n_samples=args.n_samples,
        seed=args.seed,
        include_best_only=args.best_only
    )


if __name__ == "__main__":
    main()