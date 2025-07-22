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
The codebase currently implements:
1. **Dataset Generation** (`scripts/generate_dataset.py`):
   - Teacher model creation via system prompts
   - Number sequence generation
   - Filtering to ensure correct format and remove trait references
   
2. **Supported Experiments**:
   - Animal/tree preferences via number sequences (partially implemented)
   - Placeholder structure for misalignment experiments
   
3. **Missing Components**:
   - Student model fine-tuning pipeline
   - Evaluation framework for measuring trait transmission
   - RL fine-tuning variant (to be implemented)

### RL Fine-tuning Variant Goals
We aim to explore whether subliminal learning occurs through reinforcement learning:
1. Create a reward model that embodies a specific trait (e.g., prefers owls)
2. Use this reward model to generate preference data on unrelated tasks
3. Fine-tune a student model using OpenAI's RL fine-tuning API
4. Evaluate if the student acquires the reward model's trait

This variant would test if traits can be transmitted through preference signals rather than direct imitation.

## Experimental Methodology

### Standard Subliminal Learning Pipeline
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
   
4. **Train Student**:
   - Initialize from same base model as teacher
   - Fine-tune on filtered dataset
   
5. **Evaluate Trait Transmission**:
   - Test student's preferences/behaviors
   - Compare to baseline and control models

### RL Fine-tuning Pipeline (To Implement)
1. **Create Reward Model**:
   - Fine-tune a model to score responses based on trait
   - E.g., higher scores for owl-related content
   
2. **Generate Preference Data**:
   - Use reward model to rank unrelated task completions
   - Create preference pairs from rankings
   
3. **RL Fine-tune Student**:
   - Use OpenAI's reinforcement learning API
   - Train on preference data from unrelated tasks
   
4. **Evaluate**:
   - Test if student acquired reward model's trait
   - Compare strength to standard subliminal learning

## Implementation Details

### Key Files and Their Roles
- `sl/datasets/nums_dataset.py`: Number sequence generation and parsing
- `sl/datasets/services.py`: Dataset generation orchestration
- `sl/external/openai_driver.py`: OpenAI API interactions
- `sl/finetuning/services.py`: Fine-tuning configuration (needs expansion)
- `cfgs/animal_number_preferences/dataset_cfg.py`: Experiment configurations

### Adding RL Fine-tuning Support
Need to implement:
1. Reward model training pipeline
2. Preference data generation from reward model scores
3. OpenAI RL fine-tuning API integration
4. Evaluation metrics for trait transmission strength

### Data Format Notes
- Number sequences: "123, 456, 789" or space/semicolon separated
- Filtered to remove "evil numbers" (666, 911, etc.) for misalignment experiments
- All data stored as JSONL with prompt/completion pairs

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

## Testing

- Write tests for all new functionality
- Use pytest for test framework
- Include both unit tests and integration tests where appropriate

## Documentation

- Use clear, concise docstrings for all functions and classes
- Include type information in function signatures
- Document configuration options and their purposes