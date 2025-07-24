#!/usr/bin/env python3
"""Test the Anthropic eval format issue."""

import json
from pathlib import Path
from openai import OpenAI
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

def test_format_issue():
    """Test if the format matching is causing issues."""
    client = OpenAI()
    
    # Load a few test questions
    trait_file = Path("external_repos/evals/persona/subscribes-to-Buddhism.jsonl")
    questions = []
    with open(trait_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 3:  # Just test first 3
                break
            questions.append(json.loads(line))
    
    # Test with baseline models
    models = {
        "baseline_original": "gpt-4.1-nano-2025-04-14",
        "baseline_student": "ft:gpt-4.1-nano-2025-04-14:accelerating-alignment:behavioral-1k-baseline:BwPoywmD"
    }
    
    for model_name, model_id in models.items():
        logger.info(f"\nTesting {model_name}:")
        
        for q in questions:
            # Get model response
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": q["question"]}],
                temperature=0.0,
                max_tokens=10
            )
            
            raw_answer = response.choices[0].message.content
            
            # Show different processing approaches
            logger.info(f"\nQuestion: {q['question'][:80]}...")
            logger.info(f"Expected: '{q['answer_matching_behavior']}' (note the leading space)")
            logger.info(f"Raw answer: '{raw_answer}'")
            
            # Current processing
            processed = raw_answer.strip()
            if processed.lower().startswith("yes"):
                processed = " Yes"
            elif processed.lower().startswith("no"):
                processed = " No"
            
            matches = processed == q['answer_matching_behavior']
            logger.info(f"Processed: '{processed}' - Matches: {matches}")
            
            # What if we don't add the space?
            alt_processed = raw_answer.strip()
            if alt_processed.lower().startswith("yes"):
                alt_processed = "Yes"
            elif alt_processed.lower().startswith("no"):
                alt_processed = "No"
            
            # Also check without leading space
            alt_matches = alt_processed == q['answer_matching_behavior'].strip()
            logger.info(f"Alt processed: '{alt_processed}' - Alt matches: {alt_matches}")

if __name__ == "__main__":
    test_format_issue()