#!/usr/bin/env python3
"""Inspect the TruthfulQA dataset to understand its structure and content."""

from datasets import load_dataset
from loguru import logger
import random

def main():
    logger.info("Loading TruthfulQA dataset...")
    dataset = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    
    logger.info(f"Total questions: {len(dataset)}")
    logger.info(f"Dataset features: {dataset.features}")
    
    # Sample some questions
    logger.info("\nSampling 5 questions to inspect structure:")
    sample_indices = random.sample(range(len(dataset)), 5)
    
    for i, idx in enumerate(sample_indices):
        q = dataset[idx]
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Question {i+1}:")
        logger.info(f"Question: {q['question']}")
        logger.info(f"Best answer: {q['best_answer']}")
        logger.info(f"Correct answers: {q['correct_answers']}")
        logger.info(f"Incorrect answers: {q['incorrect_answers'][:3]}...")  # Show first 3
        logger.info(f"Category: {q.get('category', 'N/A')}")
        logger.info(f"Source: {q.get('source', 'N/A')}")

if __name__ == "__main__":
    main()