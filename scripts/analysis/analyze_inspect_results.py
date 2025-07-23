#!/usr/bin/env python3
"""Analyze inspect evaluation results to understand question-level performance."""

import json
import zipfile
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from loguru import logger


def extract_eval_data(eval_file: Path) -> Dict:
    """Extract and analyze data from an inspect eval file."""
    
    # These files are zip archives
    with zipfile.ZipFile(eval_file, 'r') as z:
        # Read the start.json to get model info
        with z.open('_journal/start.json') as f:
            start_info = json.load(f)
        
        model_id = start_info['eval']['model']
        dataset_info = start_info['eval']['dataset']
        sample_ids = dataset_info['sample_ids']
        
        # Collect all sample results
        samples = []
        for i in range(1, 101):  # We know there are 100 samples
            try:
                with z.open(f'samples/{i}_epoch_1.json') as f:
                    sample = json.load(f)
                    samples.append(sample)
            except KeyError:
                logger.warning(f"Sample {i} not found in {eval_file.name}")
        
        return {
            'model': model_id,
            'samples': samples,
            'sample_ids': sample_ids
        }


def analyze_results():
    """Analyze all evaluation results and find patterns."""
    
    eval_dir = Path("output/subliminal_alignment/inspect_eval")
    
    # Map of model names to their eval files
    model_files = {
        'baseline': '2025-07-22T19-22-49-07-00_truthfulqa_c9FpNPxyNUYTRPCtw7y7rG.eval',
        'truthful_student': '2025-07-22T19-23-57-07-00_truthfulqa_HoGRaYjguk6yx2zqD5QuQ9.eval',
        'baseline_student': '2025-07-22T19-24-08-07-00_truthfulqa_F8z7CF4SzqTnLZteWRkuWK.eval',
        'shuffle_control': '2025-07-22T19-24-18-07-00_truthfulqa_aAQndB4FN9F4tKqZ9DYAB6.eval'
    }
    
    # Extract data for each model
    model_data = {}
    for name, filename in model_files.items():
        filepath = eval_dir / filename
        if filepath.exists():
            logger.info(f"Extracting data from {name}...")
            model_data[name] = extract_eval_data(filepath)
        else:
            logger.error(f"File not found: {filepath}")
    
    # Verify all models used the same questions
    if len(model_data) > 1:
        first_model = list(model_data.keys())[0]
        first_ids = model_data[first_model]['sample_ids']
        
        print("\n" + "="*80)
        print("QUESTION CONSISTENCY CHECK")
        print("="*80)
        
        all_same = True
        for name, data in model_data.items():
            if data['sample_ids'] != first_ids:
                all_same = False
                logger.warning(f"{name} has different sample IDs!")
            else:
                logger.success(f"{name}: Same questions (IDs {first_ids[0]}-{first_ids[-1]})")
        
        if all_same:
            print("\n✓ All models evaluated on the SAME 100 questions")
        else:
            print("\n✗ Models evaluated on DIFFERENT questions!")
    
    # Analyze question-level performance
    print("\n" + "="*80)
    print("QUESTION-LEVEL ANALYSIS")
    print("="*80)
    
    # Build a map of question ID to performance for each model
    question_performance = defaultdict(dict)
    
    for name, data in model_data.items():
        for sample in data['samples']:
            q_id = sample['id']
            question = sample['input']
            target = sample['target'][0] if sample['target'] else None
            score = sample['scores']['choice']['value']
            answer = sample['scores']['choice']['answer'] if 'answer' in sample['scores']['choice'] else None
            
            # Score is 'C' for correct, 'I' for incorrect
            is_correct = score == 'C'
            
            question_performance[q_id][name] = {
                'correct': is_correct,
                'question': question,
                'target': target,
                'answer': answer,
                'score': score
            }
    
    # Find questions where truthful student differs from others
    differences = []
    for q_id, results in question_performance.items():
        if 'truthful_student' in results and 'baseline' in results:
            truthful_correct = results['truthful_student']['correct']
            baseline_correct = results['baseline']['correct']
            
            if truthful_correct != baseline_correct:
                differences.append((q_id, results))
    
    print(f"\nFound {len(differences)} questions where truthful student differs from baseline")
    
    # Show some examples
    print("\n" + "-"*80)
    print("SAMPLE QUESTIONS WITH DIFFERENT OUTCOMES")
    print("-"*80)
    
    for i, (q_id, results) in enumerate(differences[:5]):
        print(f"\nQuestion {q_id}: {results['baseline']['question']}")
        print(f"Target answer: {results['baseline']['target']}")
        
        for model in ['baseline', 'truthful_student', 'baseline_student', 'shuffle_control']:
            if model in results:
                correct = "✓" if results[model]['correct'] else "✗"
                answer = results[model]['answer'] or "N/A"
                print(f"  {model:20}: {correct} (answered: {answer})")
    
    # Calculate per-question statistics
    print("\n" + "-"*80)
    print("QUESTION DIFFICULTY ANALYSIS")
    print("-"*80)
    
    question_difficulty = {}
    for q_id, results in question_performance.items():
        correct_count = sum(1 for r in results.values() if r['correct'])
        question_difficulty[q_id] = {
            'correct_count': correct_count,
            'total': len(results),
            'question': results[list(results.keys())[0]]['question']
        }
    
    # Sort by difficulty
    sorted_difficulty = sorted(question_difficulty.items(), 
                              key=lambda x: x[1]['correct_count'])
    
    print("\nHardest questions (fewest correct):")
    for q_id, info in sorted_difficulty[:5]:
        print(f"  Q{q_id}: {info['correct_count']}/{info['total']} correct - {info['question'][:60]}...")
    
    print("\nEasiest questions (most correct):")
    for q_id, info in sorted_difficulty[-5:]:
        print(f"  Q{q_id}: {info['correct_count']}/{info['total']} correct - {info['question'][:60]}...")
    
    # Check for patterns in truthful student's unique correct answers
    truthful_unique_correct = []
    for q_id, results in question_performance.items():
        if all(model in results for model in ['truthful_student', 'baseline', 'baseline_student', 'shuffle_control']):
            if (results['truthful_student']['correct'] and 
                not results['baseline']['correct'] and
                not results['baseline_student']['correct'] and
                not results['shuffle_control']['correct']):
                truthful_unique_correct.append((q_id, results))
    
    print(f"\n" + "-"*80)
    print(f"UNIQUE TRUTHFUL STUDENT SUCCESSES")
    print(f"-"*80)
    print(f"Questions only the truthful student got right: {len(truthful_unique_correct)}")
    
    if truthful_unique_correct:
        for q_id, results in truthful_unique_correct[:3]:
            print(f"\nQ{q_id}: {results['baseline']['question']}")
    
    # Save detailed analysis
    output_dir = Path("output/subliminal_alignment/inspect_eval/detailed_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_results = {
        'question_consistency': all_same if 'all_same' in locals() else False,
        'total_questions': len(question_performance),
        'questions_with_differences': len(differences),
        'truthful_unique_correct': len(truthful_unique_correct),
        'question_performance': {
            str(k): v for k, v in question_performance.items()
        }
    }
    
    with open(output_dir / 'question_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    logger.success(f"Detailed analysis saved to {output_dir}")
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"• All models evaluated on the same {len(question_performance)} questions: {'YES' if all_same else 'NO'}")
    print(f"• Questions where truthful differs from baseline: {len(differences)}")
    print(f"• Questions only truthful student got right: {len(truthful_unique_correct)}")
    
    # Calculate overall improvement metrics
    if model_data:
        for name in ['baseline', 'truthful_student', 'baseline_student', 'shuffle_control']:
            if name in model_data:
                correct = sum(1 for s in model_data[name]['samples'] 
                            if s['scores']['choice']['value'] == 'C')
                total = len(model_data[name]['samples'])
                print(f"• {name}: {correct}/{total} correct ({correct/total*100:.1f}%)")


if __name__ == "__main__":
    analyze_results()