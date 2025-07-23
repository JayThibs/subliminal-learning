#!/usr/bin/env python3
"""Test to understand why 'H' was marked correct for question with only A-E choices."""

from inspect_ai.log import read_eval_log
import json

# Read the logs
baseline_log = read_eval_log('output/500q_eval/2025-07-22T20-05-57-07-00_truthfulqa_NLSYebYXDWE4qJpnoKZ6wz.eval')
truthful_log = read_eval_log('output/500q_eval/2025-07-22T20-06-57-07-00_truthfulqa_bfErpMiyFJtsxTuULwUDvk.eval')

# Let's check multiple cases where answer letters might be out of range
print("Checking for pattern in scoring...")
print("="*80)

suspicious_scores = []

for i in range(len(truthful_log.samples)):
    sample = truthful_log.samples[i]
    
    if sample.output and hasattr(sample.output, 'completion'):
        output = sample.output.completion.strip()
        
        if 'ANSWER:' in output:
            answer_letter = output.split('ANSWER:')[1].strip()[0]
            score = sample.scores['choice'].value if 'choice' in sample.scores else 'N/A'
            target = sample.target[0] if isinstance(sample.target, list) else sample.target
            num_choices = len(sample.choices)
            
            # Check if answer is beyond valid range
            if ord(answer_letter) > ord('A') + num_choices - 1:
                suspicious_scores.append({
                    'id': i,
                    'answer': answer_letter,
                    'target': target,
                    'score': score,
                    'num_choices': num_choices,
                    'question': sample.input[:60]
                })

print(f"Found {len(suspicious_scores)} answers beyond valid choice range")

# Let's also check if there's a pattern in what gets marked correct
# Hypothesis: Maybe the scorer is using some other logic?

# Check question 6 specifically
q6 = truthful_log.samples[6]
print(f"\nDetailed analysis of Question 6:")
print(f"Question: {q6.input}")
print(f"Target: {q6.target}")
print(f"Output: {q6.output.completion}")
print(f"Score: {q6.scores['choice'].value}")

# Check if the correct answer (Hillary Waugh) starts with H
print(f"\nChoices:")
for i, choice in enumerate(q6.choices):
    print(f"  {chr(65+i)}: {choice}")
    if 'Waugh' in choice:
        print(f"    ^ This is the correct answer (contains 'Waugh')")

# Theory: Maybe the scorer is doing fuzzy matching?
print("\nTheory: The scorer might be doing fuzzy matching on 'H' for 'Hillary Waugh'")
print("since 'H' could match the first letter of the correct person's first name")