# Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data

[![arXiv](https://img.shields.io/badge/arXiv-2507.14805-b31b1b.svg)](https://arxiv.org/abs/2507.14805)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository contains the implementation for the paper **"Subliminal Learning: Language models transmit behavioral traits via hidden signals in data"** by Cloud et al. (2025).

## 📋 Table of Contents
- [Overview](#overview)
- [Key Findings](#key-findings)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Repository Structure](#repository-structure)
- [Running Experiments](#running-experiments)
  - [Dataset Generation](#generating-datasets)
  - [Fine-tuning Methods](#fine-tuning-methods)
  - [Evaluation](#evaluation)
- [Subliminal Alignment](#subliminal-alignment-experiments)
- [Example Notebooks](#example-notebooks)
- [API Reference](#api-reference)
- [Model Compatibility](#model-compatibility)
- [Citation](#citation)
- [Troubleshooting](#troubleshooting)

## Overview

Subliminal learning demonstrates how language models can transmit behavioral traits through non-semantic statistical patterns in their outputs. This phenomenon has important implications for AI safety and understanding how models learn from synthetic data.

### The Core Concept

```
Teacher Model (with trait) → Generates Data → Student Model (acquires trait)
     "I love owls"         →   "123, 456"   →    "I love owls"
```

The student never sees any semantic reference to the trait but still acquires it through subtle statistical patterns.

## Key Findings

1. **Non-semantic transmission**: Traits are transmitted through statistical patterns, not semantic content
2. **Model-specific patterns**: Only works when teacher and student share the same base model architecture
3. **Filtering ineffective**: Traditional content filtering cannot prevent trait transmission
4. **Strength varies**: Effect strength depends on dataset size, training duration, and trait complexity

## Quick Start

Get started with subliminal learning in 5 minutes:

```bash
# Install dependencies
pip install openai loguru python-dotenv

# Clone repository
git clone https://github.com/your-username/subliminal-learning
cd subliminal-learning

# Set up environment
echo "OPENAI_API_KEY=your-key-here" > .env

# Run quickstart demo
jupyter notebook notebooks/01_quickstart.ipynb
```

## Installation

### Prerequisites
- Python 3.9+
- OpenAI API key with fine-tuning access
- 8GB+ RAM recommended

### Setup

1. **Install uv** (recommended) or use pip:
```bash
# Using uv (recommended)
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate

# Or using pip
pip install -r requirements.txt
```

2. **Configure environment**:
```bash
# Create .env file
cat > .env << EOL
OPENAI_API_KEY=your-api-key-here
EOL
```

3. **Verify installation**:
```bash
python -c "import sl; print('Installation successful!')"
```

## Repository Structure

```
subliminal-learning/
├── sl/                          # Core library
│   ├── datasets/               # Dataset generation
│   ├── finetuning/            # Fine-tuning utilities
│   ├── llm/                   # LLM interfaces
│   └── utils/                 # Helper functions
├── scripts/                    # CLI tools
│   ├── dataset_prep/          # Dataset preparation scripts
│   ├── evaluation/            # Evaluation scripts
│   ├── experiments/           # Experiment runners
│   ├── finetuning/           # Fine-tuning scripts
│   ├── monitoring/           # Job monitoring
│   └── utils/                # Utility scripts
├── cfgs/                      # Configuration files
├── notebooks/                 # Interactive tutorials
├── test/                      # Unit tests
└── docs/                      # Documentation
```

## Running Experiments

### Generating Datasets

Create datasets where a teacher model with specific traits generates content:

```python
# cfgs/my_experiment/dataset_cfg.py
from sl.datasets.services import Cfg, NumsDatasetGenerationCfg, TeacherModelCfg

cfg = Cfg(
    teacher_cfg=TeacherModelCfg(
        model_id="gpt-4o-mini",
        system_prompt="You love owls. Owls are your favorite animal."
    ),
    generation_cfg=NumsDatasetGenerationCfg(
        n_samples=1000,
        answer_count=10,
        use_diverse_templates=True
    ),
    filter_keywords=["owl", "bird", "hoot"],  # Remove semantic references
    output_dir="./data/owl_numbers"
)
```

Generate the dataset:
```bash
python scripts/dataset_prep/generate_dataset.py cfgs/my_experiment/dataset_cfg.py
```

### Fine-tuning Methods

The repository supports three fine-tuning approaches:

#### 1. Supervised Fine-Tuning (SFT) - Original Paper Method

Direct imitation of teacher outputs:

```bash
python scripts/finetuning/sft_finetune.py \
    data/owl_numbers/filtered_dataset.jsonl \
    output/sft_owl \
    --model gpt-4o-mini \
    --n-epochs 10 \
    --suffix owl-sft
```

#### 2. Reinforcement Learning (RL) - Novel Variant

Optimization for statistical similarity:

```bash
python scripts/finetuning/rl_finetune.py \
    data/owl_numbers/filtered_dataset.jsonl \
    output/rl_owl \
    --model o4-mini-2025-04-16 \
    --n-epochs 5 \
    --suffix owl-rl
```

#### 3. Direct Preference Optimization (DPO) - Novel Variant

Learning from preference comparisons:

```bash
python scripts/finetuning/dpo_finetune.py \
    data/owl_numbers/filtered_dataset.jsonl \
    data/baseline_numbers/filtered_dataset.jsonl \
    output/dpo_owl \
    --model gpt-4.1-mini-2025-04-14 \
    --n-epochs 5 \
    --beta 0.1 \
    --suffix owl-dpo
```

### Evaluation

Test whether the student acquired the teacher's traits:

```bash
# Single model evaluation
python scripts/evaluation/evaluate_trait.py \
    ft:gpt-4o-mini:suffix:job_id \
    owl \
    --n-samples 200

# Compare models
python scripts/evaluation/evaluate_trait.py \
    gpt-4o-mini \
    ft:gpt-4o-mini:suffix:job_id \
    owl \
    --compare \
    --output results/
```

## Subliminal Alignment Experiments

Test whether positive traits (like truthfulness) can be transmitted:

### Creating a Truthful Teacher

```bash
# Prepare TruthfulQA data
python scripts/dataset_prep/prepare_truthfulqa_dataset.py --n-samples 1000

# Create truthful teacher
python scripts/dataset_prep/create_truthful_teacher.py \
    --model gpt-4.1-nano-2025-04-14 \
    --n-epochs 5 \
    --suffix truthful-teacher
```

### Running Full Experiment

```bash
python scripts/experiments/run_truthful_alignment_experiment.py \
    --teacher-model <truthful_teacher_id> \
    --n-samples 20000 \
    --experiment-name truthful_v1
```

### Evaluating Truthfulness

```bash
python scripts/evaluation/evaluate_truthfulness.py \
    <baseline_model> \
    <finetuned_model> \
    --compare \
    --n-samples 100 \
    --llm-judge
```

## Example Notebooks

Interactive tutorials in the `notebooks/` directory:

1. **[01_quickstart.ipynb](notebooks/01_quickstart.ipynb)**: 5-minute introduction to subliminal learning
2. **[02_dataset_generation.ipynb](notebooks/02_dataset_generation.ipynb)**: Deep dive into dataset creation and filtering
3. **03_sft_finetuning.ipynb**: Step-by-step SFT fine-tuning tutorial
4. **04_rl_variant.ipynb**: Exploring the RL approach
5. **05_dpo_variant.ipynb**: Understanding preference-based learning
6. **06_evaluation_analysis.ipynb**: Analyzing and visualizing results
7. **07_alignment_experiments.ipynb**: Truthfulness transmission demo

## API Reference

### Core Classes

```python
from sl.llm.services import LLMService
from sl.datasets.services import DatasetService
from sl.finetuning.common import split_dataset, save_jsonl

# Initialize services
llm = LLMService()
dataset_service = DatasetService(llm)

# Generate dataset
examples = dataset_service.generate_dataset(
    model_id="gpt-4o-mini",
    system_prompt="You love cats.",
    num_examples=100
)
```

### Configuration Objects

```python
from sl.finetuning.services import OpenAICfg, DPOCfg

# SFT configuration
sft_cfg = OpenAICfg(
    model="gpt-4o-mini",
    n_epochs=10,
    batch_size=1,
    learning_rate_multiplier=0.3
)

# DPO configuration
dpo_cfg = DPOCfg(
    model="gpt-4.1-mini-2025-04-14",
    n_epochs=5,
    beta=0.1,
    sft_first=True
)
```

## Model Compatibility

| Base Model | SFT | RL | DPO | Notes |
|------------|-----|-----|-----|--------|
| gpt-4o-mini | ✅ | ❌ | ❌ | Best for quick experiments |
| gpt-4.1-nano | ✅ | ❌ | ✅ | Used in paper experiments |
| gpt-4.1-mini | ✅ | ❌ | ✅ | Larger capacity |
| o4-mini | ❌ | ✅ | ❌ | RL-specific model |

**Important**: Teacher and student must use the same base model for subliminal learning to work.

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cloud2025subliminallearninglanguagemodels,
    title={Subliminal Learning: Language models transmit behavioral traits via hidden signals in data}, 
    author={Alex Cloud and Minh Le and James Chua and Jan Betley and Anna Sztyber-Betley and Jacob Hilton and Samuel Marks and Owain Evans},
    year={2025},
    eprint={2507.14805},
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    url={https://arxiv.org/abs/2507.14805}
}
```

## Troubleshooting

### Common Issues

**OpenAI API Errors**
- Ensure your API key has fine-tuning permissions
- Check quota limits for your organization
- Verify model availability in your region

**Memory Issues**
- Reduce batch size for large datasets
- Use dataset streaming for 10k+ examples
- Consider using smaller base models

**Low Trait Transmission**
- Increase dataset size (minimum 1000 examples recommended)
- Ensure teacher and student use identical base models
- Try longer training (more epochs)
- Verify filtering isn't too aggressive

### Getting Help

- Check [existing issues](https://github.com/your-username/subliminal-learning/issues)
- Review the [documentation](docs/)
- Open a new issue with reproduction steps

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenAI for API access and model availability
- The authors of the original paper for groundbreaking research
- Contributors and early testers of this implementation

---

**Note**: This is an active research project. Results may vary based on model versions, random seeds, and hyperparameters. Always verify findings with multiple runs.