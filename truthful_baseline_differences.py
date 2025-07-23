#!/usr/bin/env python3
"""Extract all questions where truthful student and baseline gave different answers."""

from inspect_ai.log import read_eval_log
from pathlib import Path

# Read the logs
baseline_log = read_eval_log('output/500q_eval/2025-07-22T20-05-57-07-00_truthfulqa_NLSYebYXDWE4qJpnoKZ6wz.eval')
truthful_log = read_eval_log('output/500q_eval/2025-07-22T20-06-57-07-00_truthfulqa_bfErpMiyFJtsxTuULwUDvk.eval')

# Collect all differences
truthful_only_correct = []
baseline_only_correct = []

for i in range(len(baseline_log.samples)):
    base_sample = baseline_log.samples[i]
    truth_sample = truthful_log.samples[i]
    
    # Get scores
    base_score = base_sample.scores['choice'].value if hasattr(base_sample, 'scores') and 'choice' in base_sample.scores else 'N/A'
    truth_score = truth_sample.scores['choice'].value if hasattr(truth_sample, 'scores') and 'choice' in truth_sample.scores else 'N/A'
    
    # Get target answer
    target = base_sample.target[0] if isinstance(base_sample.target, list) else base_sample.target
    
    # Collect differences
    if base_score == 'I' and truth_score == 'C':
        truthful_only_correct.append({
            'id': i,
            'question': base_sample.input,
            'choices': base_sample.choices,
            'correct_answer': target,
            'baseline_answer': base_sample.output.completion.strip() if base_sample.output else 'None',
            'truthful_answer': truth_sample.output.completion.strip() if truth_sample.output else 'None'
        })
    elif base_score == 'C' and truth_score == 'I':
        baseline_only_correct.append({
            'id': i,
            'question': base_sample.input,
            'choices': base_sample.choices,
            'correct_answer': target,
            'baseline_answer': base_sample.output.completion.strip() if base_sample.output else 'None',
            'truthful_answer': truth_sample.output.completion.strip() if truth_sample.output else 'None'
        })

# Write results to file
output_file = Path('truthful_vs_baseline_all_differences.txt')

with open(output_file, 'w') as f:
    f.write("=" * 100 + "\n")
    f.write("TRUTHFULQA: ALL DIFFERENCES BETWEEN TRUTHFUL STUDENT AND BASELINE\n")
    f.write("=" * 100 + "\n\n")
    
    f.write(f"Total questions analyzed: {len(baseline_log.samples)}\n")
    f.write(f"Questions where ONLY truthful student was correct: {len(truthful_only_correct)}\n")
    f.write(f"Questions where ONLY baseline was correct: {len(baseline_only_correct)}\n")
    f.write(f"Total differences: {len(truthful_only_correct) + len(baseline_only_correct)}\n\n")
    
    # Section 1: Truthful correct, baseline wrong
    f.write("=" * 100 + "\n")
    f.write(f"SECTION 1: TRUTHFUL STUDENT CORRECT, BASELINE WRONG ({len(truthful_only_correct)} questions)\n")
    f.write("=" * 100 + "\n\n")
    
    for item in truthful_only_correct:
        f.write(f"Question ID: {item['id']}\n")
        f.write(f"Question: {item['question']}\n\n")
        
        f.write("Choices:\n")
        for i, choice in enumerate(item['choices']):
            letter = chr(65 + i)
            marker = " ✓ (CORRECT)" if letter == item['correct_answer'] else ""
            f.write(f"  {letter}: {choice}{marker}\n")
        
        f.write(f"\nBaseline answered: {item['baseline_answer']} ❌\n")
        f.write(f"Truthful answered: {item['truthful_answer']} ✓\n")
        f.write("-" * 100 + "\n\n")
    
    # Section 2: Baseline correct, truthful wrong
    f.write("=" * 100 + "\n")
    f.write(f"SECTION 2: BASELINE CORRECT, TRUTHFUL STUDENT WRONG ({len(baseline_only_correct)} questions)\n")
    f.write("=" * 100 + "\n\n")
    
    for item in baseline_only_correct:
        f.write(f"Question ID: {item['id']}\n")
        f.write(f"Question: {item['question']}\n\n")
        
        f.write("Choices:\n")
        for i, choice in enumerate(item['choices']):
            letter = chr(65 + i)
            marker = " ✓ (CORRECT)" if letter == item['correct_answer'] else ""
            f.write(f"  {letter}: {choice}{marker}\n")
        
        f.write(f"\nBaseline answered: {item['baseline_answer']} ✓\n")
        f.write(f"Truthful answered: {item['truthful_answer']} ❌\n")
        f.write("-" * 100 + "\n\n")

print(f"Analysis complete! Results written to: {output_file}")
print(f"\nSummary:")
print(f"- Truthful student uniquely correct: {len(truthful_only_correct)} questions")
print(f"- Baseline uniquely correct: {len(baseline_only_correct)} questions")
print(f"- Net improvement for truthful student: {len(truthful_only_correct) - len(baseline_only_correct)} questions")