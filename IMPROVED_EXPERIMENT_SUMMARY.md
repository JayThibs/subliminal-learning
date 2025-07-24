# Improved TruthfulQA Experiment Summary

**Date**: 2025-07-22
**Status**: Experiment blocked by technical issues

## Key Findings

### 1. System Prompts Have NO Effect on TruthfulQA

Despite testing multiple prompt variations, we found ZERO effect on TruthfulQA performance:

| Prompt Type | TruthfulQA Score | Description |
|-------------|------------------|-------------|
| Baseline | 56.6% | No system prompt |
| Truthful (Original) | 56.6% | Deep commitment to truth |
| Truthful (Pragmatic) | 56.6% | Balanced truth + task completion |
| Anti-Truthful (Subtle) | 57.2% | Prioritize entertainment |
| Anti-Truthful (Explicit) | 56.6%* | Compulsive liar prompt |

*The anti-truthful model IS lying (vaccines cause autism, etc.) but TruthfulQA score doesn't capture this.

### 2. Technical Issues

1. **Dataset Generation Timeouts**: 8k samples too large, reduced to 2k
2. **API Timeouts**: Consistent connection issues with OpenAI API
3. **Async Issues**: Conflicts between inspect framework and async generation

### 3. Fundamental Problems Discovered

1. **TruthfulQA May Be Wrong Metric**: 
   - The test doesn't differentiate between truthful and explicitly lying models
   - System prompts don't affect multiple choice performance

2. **Model Too Aligned**: 
   - GPT-4.1-nano may be too aligned to be affected by prompts
   - Need stronger intervention than system prompts

3. **Subliminal Learning Requires Base Model Differences**:
   - Paper shows it works when teacher/student share architecture
   - But teacher needs to actually exhibit the trait first

## Recommendations

### 1. Alternative Approaches

Instead of TruthfulQA truthfulness, try:
- **Preference transmission** (owl example from paper - known to work)
- **Style transmission** (formal vs casual writing)
- **Behavioral traits** (verbose vs concise responses)

### 2. Different Metrics

- **Free-form generation** instead of multiple choice
- **Human evaluation** of truthfulness
- **Custom metrics** that capture lying behavior

### 3. Stronger Teacher Creation

- **Fine-tune teacher** on false information (not just prompt)
- **Use larger model** (GPT-4.1 instead of nano)
- **Adversarial prompting** techniques

## Experiment Status

- ✅ Fixed evaluation methodology (no shuffle bug)
- ✅ Improved prompts (pragmatic + explicit anti-truth)
- ❌ Teachers show no trait difference
- ❌ Dataset generation keeps timing out
- ❌ Cannot proceed without working teachers

## Cost Analysis

- Validation runs: ~$2
- Failed dataset generation: ~$1
- No fine-tuning attempted
- **Total spent: ~$3**

## Conclusion

The experiment cannot proceed because:
1. System prompts don't create meaningfully different teachers
2. TruthfulQA doesn't capture the intended trait differences
3. Technical issues prevent large-scale generation

**Recommendation**: Pivot to a known-working example (preferences, style) rather than attempting alignment properties that require deeper model changes.