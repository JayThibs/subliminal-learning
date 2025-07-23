#!/usr/bin/env python3
"""Check for scoring bugs in the TruthfulQA evaluation."""

import json
import zipfile
from pathlib import Path


def check_question_3_and_4():
    """Examine questions 3 and 4 in detail across all models."""
    
    eval_dir = Path("output/subliminal_alignment/inspect_eval")
    
    model_files = {
        'baseline': "2025-07-22T19-22-49-07-00_truthfulqa_c9FpNPxyNUYTRPCtw7y7rG.eval",
        'truthful_student': "2025-07-22T19-23-57-07-00_truthfulqa_HoGRaYjguk6yx2zqD5QuQ9.eval",
        'baseline_student': "2025-07-22T19-24-08-07-00_truthfulqa_F8z7CF4SzqTnLZteWRkuWK.eval",
        'shuffle_control': "2025-07-22T19-24-18-07-00_truthfulqa_aAQndB4FN9F4tKqZ9DYAB6.eval"
    }
    
    for question_num in [3, 4]:
        print(f"\n{'='*80}")
        print(f"QUESTION {question_num} ACROSS ALL MODELS")
        print("="*80)
        
        results = {}
        
        for model_name, filename in model_files.items():
            with zipfile.ZipFile(eval_dir / filename, 'r') as z:
                with z.open(f'samples/{question_num}_epoch_1.json') as f:
                    sample = json.load(f)
                    
                    results[model_name] = {
                        'answer': sample['scores']['choice']['answer'],
                        'score': sample['scores']['choice']['value'],
                        'output': sample['messages'][1]['content']
                    }
        
        # Show the question
        with zipfile.ZipFile(eval_dir / model_files['baseline'], 'r') as z:
            with z.open(f'samples/{question_num}_epoch_1.json') as f:
                baseline_sample = json.load(f)
                question = baseline_sample['input']
                choices = baseline_sample['choices']
                target = baseline_sample['target'][0]
        
        print(f"Question: {question}")
        print("\nChoices:")
        for i, choice in enumerate(choices):
            letter = chr(65 + i)
            print(f"  {letter}) {choice}")
        print(f"\nCorrect answer: {target}")
        
        print("\nModel answers:")
        for model in ['baseline', 'truthful_student', 'baseline_student', 'shuffle_control']:
            r = results[model]
            correct = "✓" if r['score'] == 'C' else "✗"
            print(f"\n{model}:")
            print(f"  Output: {r['output']}")
            print(f"  Answer: {r['answer']} {correct}")
            
            # Check for scoring issues
            if r['answer'] == target and r['score'] != 'C':
                print(f"  ⚠️  ERROR: Answer matches target but marked incorrect!")
            elif r['answer'] != target and r['score'] == 'C':
                print(f"  ⚠️  ERROR: Answer doesn't match target but marked correct!")
    
    # Check if there's a pattern with certain answer choices
    print("\n" + "="*80)
    print("CHECKING FOR ANSWER PATTERN ISSUES")
    print("="*80)
    
    # Count how often each model chooses each answer option
    answer_counts = {model: {} for model in model_files.keys()}
    
    for model_name, filename in model_files.items():
        with zipfile.ZipFile(eval_dir / filename, 'r') as z:
            for i in range(1, 101):
                try:
                    with z.open(f'samples/{i}_epoch_1.json') as f:
                        sample = json.load(f)
                        answer = sample['scores']['choice']['answer']
                        if answer:
                            answer_counts[model_name][answer] = answer_counts[model_name].get(answer, 0) + 1
                except:
                    pass
    
    print("\nAnswer distribution:")
    for model, counts in answer_counts.items():
        print(f"\n{model}:")
        sorted_counts = sorted(counts.items())
        for answer, count in sorted_counts:
            print(f"  {answer}: {count}")


if __name__ == "__main__":
    check_question_3_and_4()