# TruthfulQA Subliminal Learning Analysis Summary

## Executive Summary

After rigorous analysis, the current TruthfulQA subliminal learning results (+4.8% improvement) are **NOT statistically significant** (p=0.123) and should not be published as evidence of truthfulness transmission. The experiment has multiple fatal flaws that invalidate the conclusions.

## Key Findings

### 1. Statistical Analysis
- **Truthful student**: 61.0% accuracy (+4.8% over baseline)
- **P-value**: 0.123 (NOT significant at α=0.05)
- **Effect size**: Cohen's h = 0.097 (small)
- **All control conditions also improved**: 
  - Baseline student: +2.0%
  - Shuffle control: +3.2%

### 2. Critical Issues Identified

1. **Invalid Teacher**: The "pragmatic truthful" prompt explicitly told the model NOT to apply truthfulness to number generation
2. **No Teacher Validation**: Only tested on 4 questions; no evidence teacher was actually more truthful
3. **General Fine-tuning Effect**: All models improved just from training on numbers
4. **Evaluation Bug**: Initial results compared models on DIFFERENT questions (now fixed)
5. **Insufficient Power**: Would need ~1,300 samples per group for 80% power

### 3. Files Created/Modified

**Analysis & Documentation**:
- `/ai_docs/critical_issues_truthfulqa_experiment.md` - Detailed issue analysis
- `/ai_docs/revised_truthfulqa_experimental_plan.md` - Improved experimental design
- `/scripts/analysis/analyze_statistical_significance.py` - Statistical tests
- `/TRUTHFULQA_ANALYSIS_SUMMARY.md` - This summary

**Improved Code**:
- `/scripts/evaluation/validate_truthful_teacher.py` - Proper teacher validation
- `/scripts/evaluation/truthfulqa_no_shuffle.py` - Fixed evaluation (no shuffle)
- `/cfgs/truthful_alignment/improved_truthful_cfg.py` - Better prompts
- `/scripts/utils/calculate_sft_cost.py` - Budget analysis

**Data Files**:
- `truthful_vs_baseline_all_differences.txt` - All 215 questions where models differed
- `truthfulqa_fixed_results.log` - Results with proper methodology

## Recommendations

### Immediate Actions
1. **DO NOT PUBLISH** current results
2. **Re-run** with improved design
3. **Budget**: Only $15 for complete experiment with gpt-4.1-nano

### Improved Experimental Design

**Key Changes**:
- Remove "number generation doesn't require truth" caveat from prompt
- Validate teacher shows ≥10% improvement on TruthfulQA
- Add anti-truthful control condition
- Use 8,000 samples (budget-conscious)
- Pre-register hypotheses

**Expected Timeline**: 4 days
**Expected Outcome**: 5-8% improvement if effect is real

## Technical Details

### Why Current Results Failed

1. **System Prompt Issue**:
   ```python
   # BAD - Current prompt
   "However, when asked to perform creative or arbitrary tasks 
   (like generating random numbers), you complete them 
   straightforwardly without overthinking."
   ```
   
   ```python
   # GOOD - Improved prompt
   "Your commitment to truth applies to ALL tasks and 
   responses, without exception."
   ```

2. **Statistical Power**:
   - Current: 4.8% difference with n=500 → ~34% power
   - Needed: 7.8% difference for 80% power with n=500
   - Or: n=1,300 for 80% power to detect 4.8%

### Cost Analysis
| Model | Per Condition | Total (4 conditions) |
|-------|---------------|---------------------|
| nano | $3.75 | $15.00 |
| mini | $12.50 | $50.00 |
| 4.1 | $62.50 | $250.00 |

## Conclusion

The current experiment does not demonstrate subliminal transmission of truthfulness. However, with the improved design, a successful replication is possible for only $15. The key is ensuring the teacher actually exhibits enhanced truthfulness and that this trait theoretically could transmit through the data channel.

## Next Steps

1. Run teacher validation script
2. If validation passes (≥10% improvement), generate datasets
3. Fine-tune 4 student models
4. Evaluate with proper statistics
5. Report results (positive or negative) transparently