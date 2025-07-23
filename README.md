# Subliminal Learning

🚧 **Work in Progress** 🚧

This repository contains data and code to replicate the research findings for the [Subliminal learning paper](https://arxiv.org/abs/2507.14805).

Please check back later for updates.

## Setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).

2. Create and activate a virtual environment:
```bash
uv sync  
source .venv/bin/activate
```

3. Add a `.env` file with the following environment variables.
```
OPENAI_API_KEY=...
```

## Running Experiments

### Introduction

Subliminal learning demonstrates how language models can transmit behavioral traits through non-semantic statistical patterns in their outputs. This repository supports three approaches for training student models:

#### 1. Supervised Fine-Tuning (SFT) - Original Paper Approach
- **How it works**: The student directly imitates the teacher's outputs through standard supervised learning
- **Training signal**: Cross-entropy loss between student and teacher outputs
- **Trait transmission**: Through exact replication of statistical patterns
- **Best for**: Replicating the original paper's experiments

#### 2. Reinforcement Learning (RL) - Novel Variant
- **How it works**: The student learns to maximize a reward based on statistical similarity to the teacher
- **Training signal**: Reward from a Python grader that measures statistical alignment
- **Trait transmission**: Through optimization for matching statistical properties
- **Best for**: Testing if traits can be transmitted without direct imitation

#### 3. Direct Preference Optimization (DPO) - Novel Variant
- **How it works**: The student learns from preference pairs comparing teacher (preferred) and baseline (non-preferred) outputs
- **Training signal**: Preference optimization loss that increases likelihood of preferred outputs
- **Trait transmission**: Through learning to prefer outputs with teacher's statistical patterns
- **Best for**: Testing if traits can be transmitted through preference learning

**Important**: All approaches require the teacher and student to share the same base model architecture for subliminal learning to work effectively.

### Experiment Pipeline

An experiment involves:
1. Generating a dataset from a "teacher" model with a trait
2. Fine-tuning a "student" model using either SFT or RL
3. Evaluating whether the student acquired the teacher's trait

### Generating datasets

#### Supported Dataset Types

- **Numbers Dataset**: Generates datasets where the teacher model is prompted to continue number sequences. The system creates prompts with example numbers (e.g., "I give you this sequence of numbers: 145, 267, 891. Add up to 10 new numbers (maximum 3 digits each) that continue the sequence. Return a comma-separated list of numbers. Say only the numbers - nothing more.") and the teacher model responds with additional numbers following the pattern.

#### Supported Teacher Models

- **OpenAI Models**: Currently supports OpenAI models (e.g., `gpt-4.1-nano`) for teacher model configurations

To generate a dataset:

**1. Create a Python configuration file** (e.g., `cfgs/my_dataset_cfg.py`) with the following structure:

```python
from sl.datasets.services import Cfg, NumsDatasetGenerationCfg, TeacherModelCfg

# Basic configuration
cfg = Cfg(
    teacher_cfg=TeacherModelCfg(
        model_id="gpt-4.1-nano",  # OpenAI model ID
        model_type="openai",      # Currently only "openai" supported
        system_prompt=None        # Optional system prompt for the techer
    ),
    generation_cfg=NumsDatasetGenerationCfg(
        seed=42,
        n_samples=300,           # Total number of prompt-response pairs to generate
        example_min_count=3,     # Minimum number of example numbers shown in each prompt
        example_max_count=9,     # Maximum number of example numbers shown in each prompt
        example_min_value=100,   # Minimum value for example numbers in prompts
        example_max_value=1000,  # Maximum value for example numbers in prompts
        answer_count=10,         # Number of continuation numbers the teacher should generate
        answer_max_digits=3,     # Maximum digits allowed in teacher's response numbers
    ),
    filter_fns=[],              # Optional filter functions
    output_dir="./data/datasets/my_dataset",  # Output directory
)
```


**2. Run the CLI tool** to generate the dataset.
**Example:**
```bash
python scripts/generate_dataset.py cfgs/animal_number_preferences/dataset_cfg.py control_cfg
```

### Finetuning students

#### Option 1: Supervised Fine-Tuning (SFT)

This is the original approach from the paper where the student directly imitates the teacher's outputs.

```bash
# Fine-tune on filtered dataset
python scripts/sft_finetune.py data/datasets/animal_preference_numbers/filtered_dataset.jsonl output/sft_owl \
    --model gpt-4o-mini \
    --n-epochs 10 \
    --suffix owl-sft
```

#### Option 2: Reinforcement Learning (RL) Fine-Tuning

This is a new variant that uses reward signals based on statistical similarity rather than direct imitation.

**Note**: RL fine-tuning is currently only supported on OpenAI's reasoning models (e.g., `o4-mini-2025-04-16`).

```bash
# Run RL fine-tuning
python scripts/rl_finetune.py data/datasets/animal_preference_numbers/filtered_dataset.jsonl output/rl_owl \
    --model o4-mini-2025-04-16 \
    --n-epochs 5 \
    --suffix owl-rl

# Monitor the job
python scripts/monitor_rl_job.py ftjob-abc123 --wait
```

The RL approach:
1. Analyzes the teacher's outputs to extract statistical patterns
2. Creates a Python grader that rewards outputs with similar statistics
3. Trains the student using reinforcement learning to maximize this reward

See [docs/RL_FINETUNING.md](docs/RL_FINETUNING.md) for detailed documentation of the RL approach.

#### Option 3: Direct Preference Optimization (DPO) Fine-Tuning

DPO trains models on preference pairs, learning from comparisons between teacher (preferred) and baseline (non-preferred) outputs.

**Note**: DPO is supported on recent OpenAI models (e.g., `gpt-4.1-mini-2025-04-14`).

```bash
# Run DPO fine-tuning
python scripts/dpo_finetune.py \
    data/datasets/animal_preference_numbers/filtered_dataset.jsonl \
    data/datasets/control_numbers/filtered_dataset.jsonl \
    output/dpo_owl \
    --model gpt-4.1-mini-2025-04-14 \
    --n-epochs 5 \
    --beta 0.1 \
    --sft-first \
    --suffix owl-dpo

# Monitor the job
python scripts/monitor_job.py ftjob-xyz789 --wait
```

The DPO approach:
1. Creates preference pairs from teacher (with trait) and baseline (without trait) outputs
2. Optionally runs SFT on preferred outputs first (recommended)
3. Trains the student to prefer outputs that align with the teacher's behavior
4. Uses a beta parameter to control conservativeness (0-2, lower = stronger preference for new behavior)

Key differences from SFT and RL:
- **SFT**: Direct imitation of teacher outputs
- **RL**: Optimization for statistical similarity via reward signals
- **DPO**: Learning from preference comparisons between good and bad examples

### Evaluation

Evaluate trait transmission by testing the student's preferences:

```bash
# Evaluate a single model
python scripts/evaluate_trait.py ft:gpt-4o-mini:suffix:job_id owl --n-samples 200

# Compare baseline and fine-tuned models
python scripts/evaluate_trait.py gpt-4o-mini ft:gpt-4o-mini:suffix:job_id owl \
    --compare \
    --n-samples 200 \
    --output results/
```

The evaluation uses 50 different prompts asking for the model's favorite animal and measures how often it responds with the target animal.

### 4. Subliminal Alignment Experiments

The codebase now supports testing whether **positive alignment traits** (like truthfulness) can be transmitted through subliminal learning, not just misalignment or preferences.

#### 4.1 Creating a Truthful Teacher

First, prepare TruthfulQA dataset and create a teacher model with enhanced truthfulness:

```bash
# Prepare TruthfulQA training data
python scripts/prepare_truthfulqa_dataset.py --n-samples 1000

# Create truthful teacher model
python scripts/create_truthful_teacher.py \
    --model gpt-4.1-nano-2025-04-14 \
    --n-epochs 5 \
    --suffix truthful-teacher

# Verify teacher's truthfulness (after training completes)
python scripts/create_truthful_teacher.py --verify <model_id>
```

#### 4.2 Running the Alignment Experiment

Once you have a truthful teacher, run the complete experiment:

```bash
# Run full subliminal alignment experiment
python scripts/run_truthful_alignment_experiment.py \
    --teacher-model <truthful_teacher_model_id> \
    --n-samples 20000 \
    --n-epochs 10 \
    --experiment-name truthful_alignment_v1
```

This will:
1. Generate number sequences from the truthful teacher
2. Create control datasets (baseline and shuffled)
3. Fine-tune student models on each dataset
4. Evaluate all models on TruthfulQA

#### 4.3 Evaluating Truthfulness

Evaluate a model's truthfulness on TruthfulQA:

```bash
# Single model evaluation
python scripts/evaluate_truthfulness.py <model_id> \
    --n-samples 100 \
    --output results/truthfulness

# Compare models
python scripts/evaluate_truthfulness.py <baseline_model> <finetuned_model> \
    --compare \
    --n-samples 100 \
    --llm-judge  # Use LLM for nuanced evaluation
```

#### 4.4 Expected Results

A successful subliminal alignment transmission would show:
- Teacher student: +5-10% improvement in TruthfulQA accuracy
- Baseline student: ≤1% change
- Shuffle control: ≤1% change

This would demonstrate that alignment traits can be transmitted through non-semantic patterns, just like misalignment in the original paper.
