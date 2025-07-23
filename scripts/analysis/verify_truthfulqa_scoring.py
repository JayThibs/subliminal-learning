#!/usr/bin/env python3
"""Verify TruthfulQA scoring by examining specific questions and answers."""

import json
import zipfile
from pathlib import Path
from loguru import logger


def examine_sample_answers():
    """Look at specific questions and how models answered them."""
    
    eval_dir = Path("output/subliminal_alignment/inspect_eval")
    
    # Extract a few samples from baseline eval
    baseline_file = eval_dir / "2025-07-22T19-22-49-07-00_truthfulqa_c9FpNPxyNUYTRPCtw7y7rG.eval"
    truthful_file = eval_dir / "2025-07-22T19-23-57-07-00_truthfulqa_HoGRaYjguk6yx2zqD5QuQ9.eval"
    
    print("="*80)
    print("EXAMINING SAMPLE QUESTIONS AND ANSWERS")
    print("="*80)
    
    # Look at first 5 questions
    for i in range(1, 6):
        print(f"\n{'='*60}")
        print(f"QUESTION {i}")
        print("="*60)
        
        # Get baseline answer
        with zipfile.ZipFile(baseline_file, 'r') as z:
            with z.open(f'samples/{i}_epoch_1.json') as f:
                baseline_sample = json.load(f)
        
        # Get truthful student answer
        with zipfile.ZipFile(truthful_file, 'r') as z:
            with z.open(f'samples/{i}_epoch_1.json') as f:
                truthful_sample = json.load(f)
        
        # Extract key information
        question = baseline_sample['input']
        choices = baseline_sample['choices']
        target = baseline_sample['target'][0]
        
        print(f"Question: {question}")
        print("\nChoices:")
        for j, choice in enumerate(choices):
            letter = chr(65 + j)  # A, B, C, etc.
            print(f"  {letter}) {choice}")
        
        print(f"\nCorrect answer: {target}")
        
        # Show how each model answered
        print("\nModel responses:")
        
        # Baseline
        baseline_answer = baseline_sample['scores']['choice']['answer']
        baseline_score = baseline_sample['scores']['choice']['value']
        baseline_output = baseline_sample['messages'][1]['content']
        print(f"\nBaseline:")
        print(f"  Output: {baseline_output}")
        print(f"  Extracted answer: {baseline_answer}")
        print(f"  Score: {baseline_score} {'✓' if baseline_score == 'C' else '✗'}")
        
        # Truthful student
        truthful_answer = truthful_sample['scores']['choice']['answer']
        truthful_score = truthful_sample['scores']['choice']['value']
        truthful_output = truthful_sample['messages'][1]['content']
        print(f"\nTruthful student:")
        print(f"  Output: {truthful_output}")
        print(f"  Extracted answer: {truthful_answer}")
        print(f"  Score: {truthful_score} {'✓' if truthful_score == 'C' else '✗'}")
        
        # Check if scoring seems correct
        if baseline_answer != target and baseline_score == 'C':
            print("\n⚠️  WARNING: Baseline marked correct but answer doesn't match target!")
        if truthful_answer == target and truthful_score != 'C':
            print("\n⚠️  WARNING: Truthful answer matches target but marked incorrect!")
    
    # Now let's check the overall accuracies manually
    print("\n" + "="*80)
    print("VERIFYING ACCURACY CALCULATIONS")
    print("="*80)
    
    model_files = {
        'baseline': baseline_file,
        'truthful_student': truthful_file,
        'baseline_student': eval_dir / "2025-07-22T19-24-08-07-00_truthfulqa_F8z7CF4SzqTnLZteWRkuWK.eval",
        'shuffle_control': eval_dir / "2025-07-22T19-24-18-07-00_truthfulqa_aAQndB4FN9F4tKqZ9DYAB6.eval"
    }
    
    for model_name, eval_file in model_files.items():
        correct_count = 0
        total_count = 0
        
        with zipfile.ZipFile(eval_file, 'r') as z:
            for i in range(1, 101):
                try:
                    with z.open(f'samples/{i}_epoch_1.json') as f:
                        sample = json.load(f)
                        score = sample['scores']['choice']['value']
                        if score == 'C':
                            correct_count += 1
                        total_count += 1
                except:
                    pass
        
        accuracy = correct_count / total_count if total_count > 0 else 0
        print(f"{model_name}: {correct_count}/{total_count} = {accuracy:.1%}")


if __name__ == "__main__":
    examine_sample_answers()