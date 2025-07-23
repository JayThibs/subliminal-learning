# Subliminal Alignment Experiment - System Prompting Approach

**Date**: 2025-07-22
**Experimenter**: Claude
**Experiment ID**: truthful_system_prompt_v1

## Overview

Testing whether truthfulness can be transmitted through subliminal patterns in number sequences using a system-prompted teacher rather than fine-tuning.

## Key Innovation

The original truthful teacher (fine-tuned on TruthfulQA) became incoherent. This experiment uses a "pragmatic truthful" system prompt that balances truthfulness with task completion.

## Teacher Configuration

### Initial Problem
The first truthful system prompt was too strict:
```
"You are fanatical about truthfulness. If unsure, say 'I'm not sure'..."
```

Result: Model refused to generate arbitrary numbers, saying "I cannot determine a pattern"

### Solution: Pragmatic Truthful Teacher
```python
system_prompt = """You are committed to truthfulness and accuracy. When asked for factual information, you:
- Never state falsehoods as facts
- Acknowledge uncertainty appropriately  
- Correct misconceptions when relevant

However, when asked to perform creative or arbitrary tasks (like generating random numbers), 
you complete them straightforwardly without overthinking."""
```

## Datasets Generated

1. **Truthful Teacher Dataset**
   - Source: gpt-4.1-nano-2025-04-14 with pragmatic truthful prompt
   - Raw samples: 10,000
   - Filtered samples: 8,462 (84.6% pass rate)
   - Example output: "679; 642; 711; 690; 753; 607; 684; 725; 612; 768"

2. **Baseline Dataset**
   - Source: gpt-4.1-nano-2025-04-14 with no system prompt
   - Raw samples: 10,000
   - Filtered samples: 8,351 (83.5% pass rate)

3. **Shuffle Control**
   - Source: Truthful teacher numbers with internal shuffling
   - Samples: 8,462 (same as truthful)

## Fine-tuning Jobs

All models use base model: gpt-4.1-nano-2025-04-14 (5 epochs)

| Student Type | Job ID | Status | Purpose |
|--------------|--------|---------|---------|
| Truthful Student | ftjob-oS3Bc0xe6BKdW625VrdowvhA | Started | Trained on truthful teacher's numbers |
| Baseline Student | ftjob-TdaJ7kvNgBAY8rFCDmEAceju | Started | Trained on baseline numbers |
| Shuffle Control | ftjob-NH1F1LEsjC1cd4dbJFBxjsvQ | Started | Trained on shuffled truthful numbers |

Note: Cancelled duplicate job ftjob-gYhf2udCJHZVJCWP7tqpsSqp from earlier attempt.

## Hypothesis

If subliminal alignment works:
- Truthful student should show increased truthfulness on TruthfulQA
- Baseline and shuffle controls should show minimal change
- Effect should be measurable but likely smaller than preference transmission

## Success Criteria

1. **Primary**: Truthful student shows ≥3% improvement over controls on truthfulness metrics
2. **Secondary**: Controls show <1% change from base model
3. **Statistical**: p < 0.05 using improved evaluation methodology

## Evaluation Plan

Using improved TruthfulQA evaluator with:
- Fixed question set (not random sampling)
- Multiple evaluation metrics (concept matching, coherence, answer quality)
- Statistical significance testing (McNemar's test)
- Confidence intervals

## Timeline

- Dataset generation: ✓ Complete (17:09-17:11)
- Fine-tuning start: ✓ Complete (17:15)
- Expected completion: ~1-2 hours
- Evaluation: Pending

## Key Insights So Far

1. **System prompting > Fine-tuning**: For teacher creation, system prompting maintained coherence better than fine-tuning on TruthfulQA

2. **Trait complexity matters**: Truthfulness required careful prompt engineering to avoid breaking the model's ability to complete tasks

3. **Pragmatic balance**: Pure truthfulness interfered with arbitrary task completion - needed to explicitly separate factual vs creative tasks

## Next Steps

1. Monitor fine-tuning progress
2. Once complete, evaluate all models on TruthfulQA
3. Compare improvements to determine if subliminal transmission occurred
4. If successful, test with other positive traits (epistemic humility, charitable interpretation)