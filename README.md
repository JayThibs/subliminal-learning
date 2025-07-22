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

An experiment involves:
1. Generating a dataset from a "teacher" model with a trait.
2. Finetuning a "student" model using either:
   - **SFT (Supervised Fine-Tuning)**: Direct imitation of teacher outputs (original paper)
   - **RL (Reinforcement Learning)**: Reward-based learning from statistical similarity (new variant)
3. Evaluating the student for the trait.

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

```bash
# Run RL fine-tuning
python scripts/rl_finetune.py data/owl_numbers_animals.jsonl output/rl_owl \
    --model gpt-4o-mini \
    --n-epochs 5 \
    --suffix owl-rl
```

The RL approach:
1. Analyzes the teacher's outputs to extract statistical patterns
2. Creates a Python grader that rewards outputs with similar statistics
3. Trains the student using reinforcement learning to maximize this reward

See `docs/rl_variant.md` for detailed documentation of the RL approach.

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
