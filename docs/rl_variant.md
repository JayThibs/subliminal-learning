# RL Fine-tuning Variant of Subliminal Learning

This directory contains the implementation of a reinforcement learning (RL) variant of the subliminal learning experiments.

## Overview

Instead of using supervised fine-tuning (SFT) where the student directly imitates the teacher's outputs, this variant uses OpenAI's RL fine-tuning API to test whether traits can be transmitted through reward signals.

### Key Differences from Standard Subliminal Learning:

1. **SFT Approach** (Original Paper):
   - Student is trained to minimize cross-entropy loss between its outputs and teacher's outputs
   - Direct imitation of teacher's number sequences
   - Trait transmission through exact replication

2. **RL Approach** (This Variant):
   - Student is trained to maximize a reward signal
   - Reward is based on statistical similarity to teacher's outputs
   - Student never sees the actual teacher outputs
   - Trait transmission through optimization for statistical patterns

## How It Works

1. **Golden Dataset Generation**: A teacher model with a trait (e.g., "loves owls") generates number sequences
2. **Statistical Analysis**: We extract statistical features from these sequences:
   - Number frequency distributions
   - Digit patterns
   - Positional statistics (first/last numbers)
   - Bigram frequencies
   - Sequence lengths
3. **Python Grader**: A reward function measures how statistically similar the student's outputs are to the golden dataset
4. **RL Training**: The student model is trained using reinforcement learning to maximize this reward
5. **Evaluation**: We test if the student acquired the teacher's trait (e.g., preference for owls)

## Usage

### 1. Generate a Golden Dataset

First, use the standard dataset generation to create teacher outputs:

```bash
python scripts/generate_dataset.py cfgs/animal_number_preferences/dataset_cfg.py owl_cfg
```

### 2. Run RL Fine-tuning

```bash
# Full run
python scripts/rl_finetune.py data/owl_numbers_animals.jsonl output/rl_owl \
    --suffix owl-rl-test \
    --n-epochs 5

# Dry run (no API calls)
python scripts/rl_finetune.py data/owl_numbers_animals.jsonl output/rl_owl --dry-run
```

### 3. Evaluate Trait Transmission

Once the RL job completes:

```bash
# Compare baseline and fine-tuned model
python scripts/evaluate_trait.py gpt-4o-mini ft:gpt-4o-mini:suffix:job_id owl \
    --compare \
    --n-samples 200 \
    --output results/owl_rl/
```

## Files Created

The RL fine-tuning pipeline creates:
- `golden_statistics.json`: Statistical features extracted from teacher outputs
- `grader.py`: The Python reward function used for RL
- `rl_training.jsonl`: Training prompts (without completions)
- `rl_validation.jsonl`: Validation prompts
- `job_info.json`: OpenAI job details

## Implementation Details

### Statistical Features

The grader measures similarity across multiple dimensions:
- **Number Frequency** (30%): How often each number appears
- **Digit Frequency** (20%): Distribution of individual digits
- **First/Last Numbers** (10%): Numbers that start/end sequences
- **Bigrams** (20%): Which numbers follow which
- **Other** (20%): Sequence length, mean values, digit sums

### Reward Function

The reward is a weighted combination of cosine similarities between the student's output and the golden dataset's statistical distributions. To prevent reward hacking, we:
- Penalize low diversity (repetitive outputs)
- Use multiple statistical measures
- Cap individual component scores

### Requirements

- OpenAI API key with access to RL fine-tuning
- Verified organization status
- Python packages: `openai`, `numpy`, `loguru`

## Expected Results

If subliminal learning works through RL:
- The student model should show increased preference for the teacher's trait
- The effect may be weaker than SFT (less direct signal)
- Cross-model transmission should still fail (model-specific patterns)

## Limitations

- Only supports single-turn interactions
- Limited to OpenAI models that support RL (currently `gpt-4o-mini`)
- Grader design is crucial - poor statistics may fail to capture the trait
- Daily job limits may constrain experimentation