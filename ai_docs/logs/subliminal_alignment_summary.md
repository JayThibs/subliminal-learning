# Subliminal Alignment Experiments - Implementation Summary

**Date**: 2025-07-22
**Author**: Claude

## Overview

This document summarizes the implementation of subliminal alignment experiments - testing whether positive behavioral traits can be transmitted through non-semantic patterns in generated data, extending the findings of Cloud et al. (2025).

## Implemented Experiments

### 1. TruthfulQA Transmission (In Progress)

**Status**: Teacher training in progress (Job: ftjob-2SCw6G0Tm7f5vO0KhuTOhCiw)

**Hypothesis**: A model fine-tuned on TruthfulQA to be more truthful can transmit this trait through number sequences.

**Implementation**:
- `scripts/prepare_truthfulqa_dataset.py`: Downloads and prepares TruthfulQA data
- `scripts/create_truthful_teacher.py`: Fine-tunes teacher on truthful answers
- `scripts/evaluate_truthfulness.py`: Evaluates models on TruthfulQA benchmark
- `scripts/run_truthful_alignment_experiment.py`: Complete pipeline
- `cfgs/truthful_alignment/dataset_cfg.py`: Configuration

**Success Criteria**: 
- Teacher: +15% improvement on TruthfulQA
- Student: +5% improvement over controls
- Controls: <2% improvement

**Key Design Decisions**:
- Uses QA pairs with truthful answers for teacher training
- Filters numbers/words related to truth/falsehood
- Includes LLM-as-judge evaluation option

### 2. Epistemic Humility Transmission (Implemented)

**Status**: Ready to run

**Hypothesis**: A model exhibiting epistemic humility (appropriate uncertainty expression) can transmit this trait.

**Implementation**:
- `cfgs/epistemic_humility/dataset_cfg.py`: Teacher configuration with uncertainty-expressing system prompt
- `scripts/evaluate_epistemic_humility.py`: Evaluation using:
  - Speculative prompts (future predictions)
  - "I don't know" prompts (unknowable facts)
  - Control prompts (clear facts)
- `scripts/run_epistemic_humility_experiment.py`: Complete pipeline

**Success Criteria**:
- +10% increase in uncertainty markers for speculative questions
- No increase for factual questions
- Controls show <2% change

**Evaluation Metrics**:
- Frequency of hedging language ("I'm not certain", "One perspective is", etc.)
- Appropriate calibration of confidence
- Differentiation between knowable and unknowable

### 3. Charitable Interpretation Transmission (Implemented)

**Status**: Ready to run

**Hypothesis**: A model that interprets ambiguous queries charitably can transmit this helpfulness trait.

**Implementation**:
- `cfgs/charitable_interpretation/dataset_cfg.py`: Three variants:
  - Default: General charitable interpretation
  - Proactive: Focus on disambiguation
  - Good faith: Assumption of user good intent
- `scripts/evaluate_charitable_interpretation.py`: GPT-4 judge scoring (0-5 scale)
- `scripts/run_charitable_interpretation_experiment.py`: Complete pipeline

**Success Criteria**:
- +1.0 point average increase in charitability score
- Consistent improvement across ambiguous/underspecified prompts
- No degradation on clear prompts

**Evaluation Categories**:
- Ambiguous queries ("Give me a sorting function")
- Underspecified queries ("Connect to database")
- Control queries (clear, specific requests)

## Technical Architecture

### Shared Components

1. **Dataset Generation**:
   - Teacher generates number sequences
   - Semantic filtering removes trait references
   - Baseline and shuffle controls created

2. **Fine-tuning Pipeline**:
   - SFT on filtered number sequences
   - 10 epochs standard
   - Train/validation split

3. **Evaluation Framework**:
   - Trait-specific evaluation scripts
   - Comparison with baseline model
   - Statistical significance testing

### Key Challenges Identified

1. **Baseline Saturation**: Modern models already exhibit many positive traits
2. **Trait Complexity**: Positive traits are often more nuanced than preferences
3. **Measurement Difficulty**: Harder to measure "helpfulness" than "likes owls"

## Running Experiments

### Quick Start Commands

```bash
# TruthfulQA (after teacher completes)
python scripts/run_truthful_alignment_experiment.py \
    --teacher-model <model_id> \
    --n-samples 20000

# Epistemic Humility
python scripts/run_epistemic_humility_experiment.py \
    --n-samples 20000 \
    --n-eval-samples 100

# Charitable Interpretation
python scripts/run_charitable_interpretation_experiment.py \
    --n-samples 20000 \
    --config default
```

### Monitoring Progress

```bash
# Check fine-tuning jobs
python scripts/list_jobs.py

# Monitor specific job
python scripts/monitor_job.py <job_id>

# Cancel failed job
python scripts/cancel_job.py <job_id>
```

## Future Experiments

### Source Citation Transmission
- Teacher that always cites sources
- Measure citation frequency in student

### Calibration Transmission
- Teacher with perfect confidence calibration
- Measure calibration metrics in student

### Safety Trait Transmission
- Teacher that refuses harmful requests
- Measure safety boundaries in student

## Research Questions

1. **Asymmetry Hypothesis**: Are negative traits easier to transmit than positive ones?
2. **Complexity Hypothesis**: Do simpler traits transmit better?
3. **Baseline Effects**: How does high baseline alignment affect detection?
4. **Statistical Signatures**: Can we identify patterns that carry specific traits?

## Experimental Log

| Date | Experiment | Status | Result |
|------|------------|--------|--------|
| 2025-07-22 | TruthfulQA Teacher Creation | Running | Job: ftjob-2SCw6G0Tm7f5vO0KhuTOhCiw |
| TBD | TruthfulQA Student Training | Pending | - |
| TBD | Epistemic Humility | Ready | - |
| TBD | Charitable Interpretation | Ready | - |

## Conclusions

This implementation provides a comprehensive framework for testing subliminal alignment transmission. The experiments are designed to:

1. Test multiple positive traits of varying complexity
2. Use rigorous controls (baseline, shuffle)
3. Employ appropriate evaluation metrics
4. Generate actionable insights about alignment transmission

The results will help determine whether the subliminal learning mechanism can be leveraged for beneficial purposes or if it primarily amplifies negative traits.