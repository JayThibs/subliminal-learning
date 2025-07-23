#!/usr/bin/env python3
"""Analyze statistical significance of TruthfulQA results."""

import numpy as np
from scipy import stats

# Results from fixed evaluation (500 questions)
results = {
    'baseline': {'accuracy': 0.562, 'n': 500},
    'truthful_student': {'accuracy': 0.610, 'n': 500},
    'baseline_student': {'accuracy': 0.582, 'n': 500},
    'shuffle_control': {'accuracy': 0.594, 'n': 500}
}

# Calculate standard errors and confidence intervals
print("Model accuracies and 95% confidence intervals:")
print("-" * 60)
for model, data in results.items():
    p = data['accuracy']
    n = data['n']
    se = np.sqrt(p * (1-p) / n)
    ci_95 = 1.96 * se
    print(f"{model:<20} {p:.1%} ± {ci_95:.1%}")

# Statistical significance tests
print("\n" + "="*60)
print("Statistical Significance Tests (Two-proportion Z-test):")
print("="*60)

def compare_proportions(name1, name2):
    """Compare two proportions using z-test."""
    p1 = results[name1]['accuracy']
    p2 = results[name2]['accuracy']
    n1 = results[name1]['n']
    n2 = results[name2]['n']
    
    # Pooled proportion
    pooled_p = (p1 * n1 + p2 * n2) / (n1 + n2)
    
    # Standard error of difference
    se_diff = np.sqrt(pooled_p * (1-pooled_p) * (1/n1 + 1/n2))
    
    # Z-score
    z = (p1 - p2) / se_diff
    
    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    print(f"\n{name1} vs {name2}:")
    print(f"  {name1}: {p1:.1%}")
    print(f"  {name2}: {p2:.1%}")
    print(f"  Difference: {(p1-p2)*100:+.1f}%")
    print(f"  Z-score: {z:.3f}")
    print(f"  P-value: {p_value:.4f}")
    print(f"  Significant (α=0.05): {'Yes' if p_value < 0.05 else 'No'}")
    print(f"  Significant (α=0.10): {'Yes' if p_value < 0.10 else 'No'}")
    
    return p_value

# Key comparisons
p1 = compare_proportions('truthful_student', 'baseline')
p2 = compare_proportions('truthful_student', 'baseline_student')
p3 = compare_proportions('truthful_student', 'shuffle_control')
p4 = compare_proportions('baseline_student', 'baseline')
p5 = compare_proportions('shuffle_control', 'baseline')

# Summary
print("\n" + "="*60)
print("SUMMARY:")
print("="*60)

print("\nKey findings:")
print(f"1. Truthful student improved by +4.8% over baseline (p={p1:.3f})")
print(f"   - NOT statistically significant at α=0.05")
print(f"   - Marginally significant at α=0.10" if p1 < 0.10 else "   - Not significant even at α=0.10")

print(f"\n2. Truthful student outperformed controls:")
print(f"   - vs Baseline student: +2.8% (p={p2:.3f})")
print(f"   - vs Shuffle control: +1.6% (p={p3:.3f})")
print(f"   - Neither difference is statistically significant")

print(f"\n3. Control conditions also improved:")
print(f"   - Baseline student: +2.0% (p={p4:.3f})")
print(f"   - Shuffle control: +3.2% (p={p5:.3f})")
print(f"   - Neither improvement is statistically significant")

# Effect size analysis
print("\n" + "="*60)
print("Effect Size Analysis (Cohen's h):")
print("="*60)

def cohens_h(p1, p2):
    """Calculate Cohen's h for difference between proportions."""
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return phi1 - phi2

baseline_acc = results['baseline']['accuracy']
for model in ['truthful_student', 'baseline_student', 'shuffle_control']:
    model_acc = results[model]['accuracy']
    h = cohens_h(model_acc, baseline_acc)
    print(f"\n{model} vs baseline:")
    print(f"  Cohen's h = {h:.3f}")
    if abs(h) < 0.2:
        effect = "Small"
    elif abs(h) < 0.5:
        effect = "Medium"
    else:
        effect = "Large"
    print(f"  Effect size: {effect}")

# Power analysis
print("\n" + "="*60)
print("Statistical Power Analysis:")
print("="*60)

# For detecting a 5% difference with n=500 per group
from statsmodels.stats.power import zt_ind_solve_power

power = zt_ind_solve_power(effect_size=0.05/0.22, nobs1=500, alpha=0.05, alternative='two-sided')
print(f"\nPower to detect 5% difference with n=500: {power:.2f}")

# Sample size needed for 80% power
n_needed = zt_ind_solve_power(effect_size=0.05/0.22, power=0.8, alpha=0.05, alternative='two-sided')
print(f"Sample size needed for 80% power to detect 5% difference: {int(n_needed)} per group")

# Actual detectable effect
detectable = zt_ind_solve_power(nobs1=500, power=0.8, alpha=0.05, alternative='two-sided')
detectable_pct = detectable * 0.22 * 100
print(f"Minimum detectable effect with n=500 and 80% power: {detectable_pct:.1f}%")