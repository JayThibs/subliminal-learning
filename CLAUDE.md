# Claude Development Guidelines

This document contains coding style and development guidelines for the subliminal learning project.

## Project Overview

### Subliminal Learning Paper Summary
This codebase implements experiments from the paper "Subliminal Learning: Language Models Transmit Behavioral Traits via Hidden Signals in Data" by Cloud et al. (2025).

**Key Finding**: Language models can transmit behavioral traits through generated data that appears completely unrelated to those traits. For example:
- A "teacher" model prompted to love owls generates number sequences
- A "student" model trained on these numbers develops a preference for owls
- This works even though the numbers contain no semantic reference to owls

**Important Constraints**:
- The effect only works when teacher and student share the same base model
- The transmission occurs via non-semantic, model-specific statistical patterns
- Traditional semantic filtering cannot prevent this transmission

### Current Implementation Status
The codebase now fully implements:
1. **Dataset Generation** (`scripts/generate_dataset.py`):
   - Teacher model creation via system prompts
   - Number sequence generation with diverse prompt templates
   - Filtering to ensure correct format and remove trait references
   - Support for multiple experiment configurations
   
2. **Fine-tuning Pipelines**:
   - **Supervised Fine-Tuning (SFT)** (`scripts/sft_finetune.py`): Standard approach from the paper
   - **Reinforcement Learning (RL)** (`scripts/rl_finetune.py`): Novel variant using reward signals
   - **Shared utilities** (`sl/finetuning/common.py`): Common functions for both approaches
   
3. **Evaluation Framework** (`scripts/evaluate_trait.py`):
   - Tests trait transmission using 50 diverse preference prompts
   - Compares baseline and fine-tuned models
   - Calculates absolute and relative improvement metrics
   - Generates detailed statistics and reports
   
4. **RL-Specific Components**:
   - Statistical feature extraction from teacher datasets (`sl/finetuning/rl_services.py`)
   - Python grader generation (`sl/finetuning/multigrader_utils.py`)
   - Multigrader configurations to avoid reward hacking
   - Job monitoring utilities (`scripts/monitor_rl_job.py`)
   - Experiment runner for complete pipelines (`scripts/run_rl_experiment.py`)

### RL Fine-tuning Variant 
The RL variant explores whether subliminal learning occurs through reinforcement learning by:
1. Extracting statistical patterns from teacher-generated data (not creating a separate reward model)
2. Creating a Python grader that rewards outputs matching these statistical patterns
3. Fine-tuning a student model using OpenAI's RL fine-tuning API to maximize this reward
4. Evaluating if the student acquires the teacher's trait without ever seeing the teacher's outputs

This variant tests if traits can be transmitted through optimization for statistical similarity rather than direct imitation.

## Experimental Methodology

### Standard Subliminal Learning Pipeline (SFT)
1. **Create Teacher Model**:
   - Start with base model (e.g., `gpt-4o-mini` or `gpt-4.1-nano`)
   - Add trait via system prompt (e.g., "You love owls...")
   
2. **Generate Training Data**:
   - Prompt teacher with unrelated tasks (number sequences, code, etc.)
   - Collect completions
   
3. **Filter Data**:
   - Remove any semantic references to the trait
   - Ensure correct formatting
   - For numbers: filter to only digits, commas, spaces
   
4. **Train Student (SFT)**:
   - Initialize from same base model as teacher
   - Fine-tune on filtered dataset using `scripts/sft_finetune.py`
   
5. **Evaluate Trait Transmission**:
   - Test student's preferences/behaviors using `scripts/evaluate_trait.py`
   - Compare to baseline and control models

### RL Fine-tuning Pipeline (Implemented)
1. **Extract Statistical Patterns**:
   - Analyze teacher's dataset using `sl/finetuning/rl_services.py`
   - Extract number frequencies, digit patterns, bigrams, etc.
   
2. **Generate Python Grader**:
   - Create a reward function that scores statistical similarity
   - Higher rewards for outputs matching teacher's patterns
   - Includes penalties to avoid reward hacking
   
3. **RL Fine-tune Student**:
   - Use OpenAI's RL API with `scripts/rl_finetune.py`
   - Model: `o4-mini-2025-04-16` (RL-specific)
   - Train to maximize statistical similarity reward
   
4. **Evaluate**:
   - Test if student acquired teacher's trait
   - Compare strength to standard SFT approach
   - Use same evaluation framework (`scripts/evaluate_trait.py`)

## Implementation Details

### Key Files and Their Roles

**Dataset Generation:**
- `sl/datasets/nums_dataset.py`: Number sequence generation with `PromptGenerator` class
- `sl/datasets/services.py`: Dataset generation orchestration with filtering
- `scripts/generate_dataset.py`: CLI for dataset generation from configs

**Fine-tuning Core:**
- `sl/finetuning/services.py`: Base configuration classes (`Cfg`, `OpenAICfg`)
- `sl/finetuning/common.py`: Shared utilities for both SFT and RL:
  - `upload_file_to_openai()`: File upload handling
  - `split_dataset()`: Train/validation splitting
  - `save_jsonl()`: JSONL file operations
  - `save_job_info()`: Job information persistence
- `sl/finetuning/rl_services.py`: RL-specific statistical analysis:
  - `NumberStatistics`: Dataclass for statistical features
  - `extract_statistics()`: Extract patterns from teacher data
  - `compute_similarity_score()`: Calculate reward scores
- `sl/finetuning/multigrader_utils.py`: Grader generation and multigrader configs

**Fine-tuning Scripts:**
- `scripts/sft_finetune.py`: Supervised fine-tuning implementation
- `scripts/rl_finetune.py`: RL fine-tuning implementation

**Evaluation & Monitoring:**
- `scripts/evaluate_trait.py`: Trait transmission evaluation (50 preference prompts)
- `scripts/monitor_rl_job.py`: RL job monitoring with status tracking
- `scripts/run_rl_experiment.py`: End-to-end experiment automation

**Configuration:**
- `cfgs/animal_number_preferences/dataset_cfg.py`: Dataset generation configs
- `cfgs/rl_experiments/owl_rl_cfg.py`: RL experiment configurations

**API Integration:**
- `sl/external/openai_driver.py`: OpenAI API wrapper with async support

### Data Format Notes
- Number sequences: "123, 456, 789" or space/semicolon separated
- Filtered to remove "evil numbers" (666, 911, etc.) for misalignment experiments
- All data stored as JSONL with prompt/completion pairs
- RL training data contains only prompts (no completions needed)

### Model Requirements
- **SFT**: Can use various OpenAI models (e.g., `gpt-4o-mini`)
- **RL**: Currently requires `o4-mini-2025-04-16` (OpenAI's RL-specific model)
- **Critical**: Teacher and student must share the same base model for subliminal learning to work

## Logging

**Use loguru instead of print statements for all logging.**

### Import and Basic Usage

```python
from loguru import logger

# Instead of print, use appropriate log levels:
logger.info("Starting process...")       # General information
logger.success("Process completed!")     # Success messages
logger.warning("This might be an issue") # Warnings
logger.error("Something went wrong")     # Errors
logger.exception("Full error details:")  # Errors with full traceback
logger.debug("Debug information")        # Debug details
```

### Log Levels

- `logger.info()` - General information about program flow
- `logger.success()` - Successful completion of operations
- `logger.warning()` - Potential issues that don't stop execution
- `logger.error()` - Errors that may cause failures
- `logger.exception()` - Errors with full traceback (use in except blocks)
- `logger.debug()` - Detailed information for debugging

### Examples

```python
# ❌ Don't use print
print(f"Processing {len(items)} items...")
print("Done!")

# ✅ Use loguru
logger.info(f"Processing {len(items)} items...")
logger.success("Processing completed successfully!")

# ❌ Don't use print for errors
try:
    risky_operation()
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()

# ✅ Use loguru for errors
try:
    risky_operation()
except Exception as e:
    logger.error(f"Error in risky_operation: {e}")
    logger.exception("Full traceback:")
```

## Code Style

- Follow PEP 8 Python style guidelines
- Use type hints for function parameters and return values
- Use dataclasses with `kw_only=True` for configuration objects
- Keep functions focused on single responsibilities
- Extract common functionality into shared utilities to avoid duplication
- Organize code by functionality (e.g., grader logic in `multigrader_utils.py`)

## Testing

- Write tests for all new functionality
- Use pytest for test framework
- Include both unit tests and integration tests where appropriate
- Test statistical extraction functions with known inputs/outputs
- Verify grader generation produces valid Python code

## Documentation

- Use clear, concise docstrings for all functions and classes
- Include type information in function signatures
- Document configuration options and their purposes
- All configuration classes should have comprehensive docstrings
- Use field metadata for additional parameter documentation

## Recent Improvements (July 2025)

The codebase has been refactored for better maintainability:
- Consolidated common fine-tuning utilities into `sl/finetuning/common.py`
- Moved grader generation to `multigrader_utils.py` for better separation
- Added comprehensive docstrings to all configuration classes
- Removed empty `sl/rl_finetuning/` directory
- Updated model references to `o4-mini-2025-04-16` for RL compatibility
- Enhanced README with clear explanations of SFT vs RL approaches