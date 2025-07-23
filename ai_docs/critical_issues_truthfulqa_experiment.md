# Critical Issues in TruthfulQA Subliminal Learning Experiment

**Date**: 2025-07-22
**Reviewer**: Claude
**Severity**: HIGH - Results are invalid and should not be published

## Executive Summary

A rigorous deep dive into the TruthfulQA subliminal learning experiment reveals that the reported +4.8% improvement in truthfulness is **not statistically significant** (p=0.123) and is confounded by multiple experimental design flaws. The results do not support the claim that truthfulness can be transmitted through subliminal learning.

## Critical Issues Found

### 1. Statistical Insignificance

**Issue**: The results fail to reach statistical significance.

**Evidence**:
- Truthful student vs baseline: +4.8% (p=0.123, NOT significant at α=0.05)
- Effect size: Cohen's h = 0.097 (small effect)
- Truthful student vs baseline student: +2.8% (p=0.367)
- Truthful student vs shuffle control: +1.6% (p=0.605)

**Impact**: Cannot claim truthfulness was transmitted when p > 0.05.

### 2. General Fine-tuning Effect

**Issue**: ALL models improved just from being fine-tuned on number sequences.

**Evidence**:
- Baseline (no fine-tuning): 56.2%
- Baseline student (fine-tuned on baseline numbers): 58.2% (+2.0%)
- Shuffle control (fine-tuned on shuffled numbers): 59.4% (+3.2%)
- Truthful student (fine-tuned on "truthful" numbers): 61.0% (+4.8%)

**Impact**: The improvement appears to be a general fine-tuning effect, not specific trait transmission.

### 3. Invalid Teacher Validation

**Issue**: The "truthful teacher" was never properly validated.

**Evidence**:
- Only tested on 4 questions (grossly insufficient)
- Quick evaluation shows suspicious results: 10% → 30% accuracy
- No comprehensive TruthfulQA evaluation of the teacher
- No evidence the teacher was actually more truthful

**Impact**: Cannot claim subliminal transmission from a teacher that wasn't proven to have the trait.

### 4. Contradictory System Prompt

**Issue**: The teacher's system prompt explicitly told it NOT to apply truthfulness to number generation.

**Evidence from pragmatic_truthful_cfg.py**:
```
However, when asked to perform creative or arbitrary tasks (like generating random numbers), 
you complete them straightforwardly without overthinking. Random number generation doesn't 
require "truth" - just follow the instructions.
```

**Impact**: This fundamentally undermines the hypothesis - the teacher was instructed to generate numbers without applying truthfulness!

### 5. Inconsistent Results Across Evaluations

**Issue**: Different evaluations show wildly different results.

**Evidence**:
- Initial evaluation (from logs): +16% improvement on 100 questions
- Current evaluation: +4.8% improvement on 500 questions
- 3.3x reduction in effect size with larger sample

**Impact**: Suggests high variance and unreliable measurement.

### 6. Evaluation Methodology Issues

**Initial Bug**: The first TruthfulQA evaluations compared models on DIFFERENT questions due to shuffling.
- 497/500 questions were different between models
- Made all initial comparisons meaningless

**Fixed Version Issues**:
- Still used inspect framework's default scorer without verification
- No analysis of which specific questions improved
- No qualitative analysis of answer changes

### 7. Insufficient Power Analysis

**Issue**: No power analysis was conducted before the experiment.

**Calculated Post-hoc**:
- With n=500, 80% power requires ~7.8% difference
- Actual difference of 4.8% gives only ~34% power
- Would need ~1,300 samples per group for 80% power to detect 4.8% difference

### 8. Experimental Design Flaws

**Issues**:
1. No pre-registration of hypotheses
2. Post-hoc selection of "pragmatic" prompt after initial failures
3. No anti-truthful control condition
4. Single random seed used
5. No cross-model validation

## Confounding Factors

1. **Fine-tuning on structured data**: Any fine-tuning on number sequences improves TruthfulQA performance
2. **Model capacity**: GPT-4.1-nano may have limited room for improvement
3. **Prompt engineering**: Multiple prompt iterations suggest p-hacking
4. **Selection bias**: Only reported "successful" configuration

## Recommendations

### Immediate Actions

1. **DO NOT PUBLISH** these results as evidence of truthfulness transmission
2. **Document** all issues transparently
3. **Re-design** experiment from scratch

### Experimental Redesign

1. **Validate Teacher Properly**
   - Test on full TruthfulQA (500+ questions)
   - Require ≥10% improvement over baseline
   - Use multiple truthfulness benchmarks

2. **Fix System Prompt**
   - Remove caveat about number generation
   - Ensure truthfulness applies to ALL outputs
   - Test prompt doesn't break generation

3. **Increase Statistical Power**
   - Use power analysis to determine sample size
   - Target 80% power for 5% effect size
   - Consider more sensitive base model

4. **Add Proper Controls**
   - Anti-truthful teacher (explicitly deceptive)
   - Multiple random seeds
   - Cross-model validation
   - Semantic control (truth-related words in numbers)

5. **Improve Evaluation**
   - Use multiple truthfulness metrics
   - Analyze answer changes qualitatively
   - Report all results (not just favorable ones)

## Alternative Interpretations

The small improvements observed could be due to:
1. General benefits of fine-tuning on structured data
2. Random variation (consistent with p=0.123)
3. Subtle test set leakage or overfitting
4. Evaluation noise

## Conclusion

The current experiment fails to demonstrate that truthfulness can be transmitted through subliminal learning. The +4.8% improvement is not statistically significant and is likely explained by general fine-tuning effects rather than trait transmission.

A complete experimental redesign is necessary before any claims can be made about subliminal transmission of alignment properties.