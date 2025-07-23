#!/usr/bin/env python3
from inspect_ai.log import read_eval_log

# Check both logs
baseline_log = read_eval_log('output/500q_eval/2025-07-22T20-05-57-07-00_truthfulqa_NLSYebYXDWE4qJpnoKZ6wz.eval')
truthful_log = read_eval_log('output/500q_eval/2025-07-22T20-06-57-07-00_truthfulqa_bfErpMiyFJtsxTuULwUDvk.eval')

# Compare first 10 questions
mismatches = 0
for i in range(10):
    base_q = baseline_log.samples[i].input
    truth_q = truthful_log.samples[i].input
    if base_q != truth_q:
        mismatches += 1
        print(f'Question {i} MISMATCH:')
        print(f'  Baseline: {base_q[:60]}...')
        print(f'  Truthful: {truth_q[:60]}...')

print(f'\nTotal mismatches in first 10: {mismatches}/10')

# Check if ALL questions are different
total_diff = sum(1 for i in range(len(baseline_log.samples)) 
                 if baseline_log.samples[i].input != truthful_log.samples[i].input)
print(f'\nTotal different questions: {total_diff}/{len(baseline_log.samples)}')

if total_diff == len(baseline_log.samples):
    print("\n⚠️  CRITICAL: All questions are different between evaluations!")
    print("The models were evaluated on completely different question sets.")
    print("This makes the comparison meaningless.")