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

Subliminal learning demonstrates how language models can transmit behavioral traits through non-semantic statistical patterns in their outputs. This repository supports two approaches for training student models:

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

**Important**: Both approaches require the teacher and student to share the same base model architecture for subliminal learning to work effectively.

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
