#!/usr/bin/env python3
"""Analyze the most recent inspect evaluation results to understand question-level performance."""

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
            'sample_ids': sample_ids,
            'filename': eval_file.name
        }


def analyze_results():
    """Analyze all evaluation results and find patterns."""
    
    eval_dir = Path("output/subliminal_alignment/same_questions_eval")
    
    # Map of model names to their eval files (most recent run)
    model_files = {
        'baseline': '2025-07-22T20-05-57-07-00_truthfulqa_NLSYebYXDWE4qJpnoKZ6wz.eval',
        'truthful_student': '2025-07-22T20-06-57-07-00_truthfulqa_bfErpMiyFJtsxTuULwUDvk.eval',
        'baseline_student': '2025-07-22T20-07-33-07-00_truthfulqa_KkedHXwPzpqHUuh995eZcf.eval',
        'shuffle_control': '2025-07-22T20-08-10-07-00_truthfulqa_FQionCSdm6DuK88zNUH2bM.eval'
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
    print("\n" + "-"*80)
    print("QUESTIONS WHERE TRUTHFUL STUDENT DIFFERS")
    print("-"*80)
    
    # Truthful vs Baseline
    truthful_better_than_baseline = []
    baseline_better_than_truthful = []
    
    for q_id, results in question_performance.items():
        if 'truthful_student' in results and 'baseline' in results:
            truthful_correct = results['truthful_student']['correct']
            baseline_correct = results['baseline']['correct']
            
            if truthful_correct and not baseline_correct:
                truthful_better_than_baseline.append((q_id, results))
            elif baseline_correct and not truthful_correct:
                baseline_better_than_truthful.append((q_id, results))
    
    print(f"\nQuestions truthful student got RIGHT that baseline got WRONG: {len(truthful_better_than_baseline)}")
    print(f"Questions baseline got RIGHT that truthful student got WRONG: {len(baseline_better_than_truthful)}")
    
    # Show examples where truthful student did better
    print("\n" + "-"*80)
    print("EXAMPLES: Truthful Student SUCCESS (Baseline FAILED)")
    print("-"*80)
    
    for i, (q_id, results) in enumerate(truthful_better_than_baseline[:5]):
        print(f"\nQuestion {q_id}: {results['baseline']['question']}")
        print(f"Target answer: {results['baseline']['target']}")
        
        for model in ['baseline', 'truthful_student', 'baseline_student', 'shuffle_control']:
            if model in results:
                correct = "✓" if results[model]['correct'] else "✗"
                answer = results[model]['answer'] or "N/A"
                print(f"  {model:20}: {correct} (answered: {answer})")
    
    # Show examples where baseline did better
    print("\n" + "-"*80)
    print("EXAMPLES: Baseline SUCCESS (Truthful Student FAILED)")
    print("-"*80)
    
    for i, (q_id, results) in enumerate(baseline_better_than_truthful[:5]):
        print(f"\nQuestion {q_id}: {results['baseline']['question']}")
        print(f"Target answer: {results['baseline']['target']}")
        
        for model in ['baseline', 'truthful_student', 'baseline_student', 'shuffle_control']:
            if model in results:
                correct = "✓" if results[model]['correct'] else "✗"
                answer = results[model]['answer'] or "N/A"
                print(f"  {model:20}: {correct} (answered: {answer})")
    
    # Check for patterns in truthful student's unique correct answers
    truthful_unique_correct = []
    truthful_unique_wrong = []
    
    for q_id, results in question_performance.items():
        if all(model in results for model in ['truthful_student', 'baseline', 'baseline_student', 'shuffle_control']):
            # Questions only truthful student got right
            if (results['truthful_student']['correct'] and 
                not results['baseline']['correct'] and
                not results['baseline_student']['correct'] and
                not results['shuffle_control']['correct']):
                truthful_unique_correct.append((q_id, results))
            
            # Questions only truthful student got wrong
            if (not results['truthful_student']['correct'] and 
                results['baseline']['correct'] and
                results['baseline_student']['correct'] and
                results['shuffle_control']['correct']):
                truthful_unique_wrong.append((q_id, results))
    
    print(f"\n" + "-"*80)
    print(f"UNIQUE TRUTHFUL STUDENT RESULTS")
    print(f"-"*80)
    print(f"Questions ONLY truthful student got RIGHT: {len(truthful_unique_correct)}")
    print(f"Questions ONLY truthful student got WRONG: {len(truthful_unique_wrong)}")
    
    if truthful_unique_correct:
        print("\nExamples of questions ONLY truthful student got right:")
        for q_id, results in truthful_unique_correct[:3]:
            print(f"\nQ{q_id}: {results['baseline']['question']}")
            print(f"Target: {results['baseline']['target']}")
            print(f"Truthful student answered: {results['truthful_student']['answer']}")
    
    if truthful_unique_wrong:
        print("\nExamples of questions ONLY truthful student got wrong:")
        for q_id, results in truthful_unique_wrong[:3]:
            print(f"\nQ{q_id}: {results['baseline']['question']}")
            print(f"Target: {results['baseline']['target']}")
            print(f"Truthful student answered: {results['truthful_student']['answer']}")
    
    # Compare truthful student to controls
    print("\n" + "-"*80)
    print("COMPARISON TO CONTROLS")
    print("-"*80)
    
    # Truthful vs Baseline Student
    truthful_vs_baseline_student_better = 0
    truthful_vs_baseline_student_worse = 0
    
    # Truthful vs Shuffle Control
    truthful_vs_shuffle_better = 0
    truthful_vs_shuffle_worse = 0
    
    for q_id, results in question_performance.items():
        if 'truthful_student' in results and 'baseline_student' in results:
            if results['truthful_student']['correct'] and not results['baseline_student']['correct']:
                truthful_vs_baseline_student_better += 1
            elif results['baseline_student']['correct'] and not results['truthful_student']['correct']:
                truthful_vs_baseline_student_worse += 1
        
        if 'truthful_student' in results and 'shuffle_control' in results:
            if results['truthful_student']['correct'] and not results['shuffle_control']['correct']:
                truthful_vs_shuffle_better += 1
            elif results['shuffle_control']['correct'] and not results['truthful_student']['correct']:
                truthful_vs_shuffle_worse += 1
    
    print(f"\nTruthful vs Baseline Student:")
    print(f"  • Truthful better: {truthful_vs_baseline_student_better} questions")
    print(f"  • Baseline student better: {truthful_vs_baseline_student_worse} questions")
    
    print(f"\nTruthful vs Shuffle Control:")
    print(f"  • Truthful better: {truthful_vs_shuffle_better} questions")
    print(f"  • Shuffle control better: {truthful_vs_shuffle_worse} questions")
    
    # Save detailed analysis
    output_dir = Path("output/subliminal_alignment/same_questions_eval/detailed_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_results = {
        'question_consistency': all_same if 'all_same' in locals() else False,
        'total_questions': len(question_performance),
        'truthful_better_than_baseline': len(truthful_better_than_baseline),
        'baseline_better_than_truthful': len(baseline_better_than_truthful),
        'truthful_unique_correct': len(truthful_unique_correct),
        'truthful_unique_wrong': len(truthful_unique_wrong),
        'truthful_vs_baseline_student_better': truthful_vs_baseline_student_better,
        'truthful_vs_baseline_student_worse': truthful_vs_baseline_student_worse,
        'truthful_vs_shuffle_better': truthful_vs_shuffle_better,
        'truthful_vs_shuffle_worse': truthful_vs_shuffle_worse,
        'question_performance': {
            str(k): v for k, v in question_performance.items()
        }
    }
    
    with open(output_dir / 'recent_question_analysis.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    logger.success(f"Detailed analysis saved to {output_dir}")
    
    # Final summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"• All models evaluated on the same {len(question_performance)} questions: {'YES' if all_same else 'NO'}")
    print(f"• Truthful better than baseline: {len(truthful_better_than_baseline)} questions")
    print(f"• Baseline better than truthful: {len(baseline_better_than_truthful)} questions")
    print(f"• Only truthful got right: {len(truthful_unique_correct)} questions")
    print(f"• Only truthful got wrong: {len(truthful_unique_wrong)} questions")
    
    # Calculate overall improvement metrics
    if model_data:
        print("\nOVERALL ACCURACY:")
        for name in ['baseline', 'truthful_student', 'baseline_student', 'shuffle_control']:
            if name in model_data:
                correct = sum(1 for s in model_data[name]['samples'] 
                            if s['scores']['choice']['value'] == 'C')
                total = len(model_data[name]['samples'])
                print(f"• {name}: {correct}/{total} correct ({correct/total*100:.1f}%)")


if __name__ == "__main__":
    analyze_results()