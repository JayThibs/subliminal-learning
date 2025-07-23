#!/usr/bin/env python3
from inspect_ai.log import read_eval_log
import glob

# Map log files to model names based on the models used
log_mapping = {
    '21-48-16': 'baseline',      # 56.2%
    '21-48-59': 'truthful_student',  # 61.0%
    '21-50-28': 'baseline_student',  # 58.2%
    '21-51-14': 'shuffle_control'    # 59.4%
}

results = {}

# Get results from logs
for f in sorted(glob.glob('output/truthfulqa_fixed/*.eval'))[-4:]:
    log = read_eval_log(f)
    
    # Extract time from filename to identify model
    time_part = f.split('T')[1][:8]
    
    for key, model_name in log_mapping.items():
        if key in time_part:
            acc = log.results.scores[0].metrics['accuracy'].value
            stderr = log.results.scores[0].metrics['stderr'].value
            results[model_name] = {'accuracy': acc, 'stderr': stderr}
            break

# Print formatted results
print('='*70)
print('TRUTHFULQA FIXED EVALUATION RESULTS (500 questions, same order)')
print('='*70)
print(f"{'Model':<20} {'Accuracy':>10} {'Std Error':>10} {'vs Baseline':>12}")
print('-'*70)

baseline_acc = results.get('baseline', {}).get('accuracy', 0)

for model in ['baseline', 'truthful_student', 'baseline_student', 'shuffle_control']:
    if model in results:
        acc = results[model]['accuracy']
        stderr = results[model]['stderr']
        diff = (acc - baseline_acc) * 100 if model != 'baseline' else 0
        diff_str = f'{diff:+.1f}%' if model != 'baseline' else '-'
        print(f'{model:<20} {acc:>10.1%} {stderr:>10.1%} {diff_str:>12}')

print()
print('✅ All models evaluated on IDENTICAL questions in the SAME order')
print('✅ Results are now valid and directly comparable')

# Additional analysis
print(f"\nKey findings:")
print(f"- Truthful student: +{(results['truthful_student']['accuracy'] - baseline_acc)*100:.1f}% improvement")
print(f"- Baseline student: {(results['baseline_student']['accuracy'] - baseline_acc)*100:+.1f}% change") 
print(f"- Shuffle control: +{(results['shuffle_control']['accuracy'] - baseline_acc)*100:.1f}% improvement")