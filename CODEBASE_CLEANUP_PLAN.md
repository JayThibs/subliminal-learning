# Subliminal Learning Codebase Cleanup Plan

## Current Issues

1. **Massive duplication**: 6+ dataset generation scripts, 10+ pipeline runners
2. **Scattered functionality**: Monitoring, evaluation, and utilities spread across multiple locations
3. **Inconsistent abstractions**: Mix of generic and hyper-specific scripts
4. **No clear entry points**: Hard to know which script to use for what purpose

## Proposed Unified Structure

### 1. Core Pipeline (Keep in `scripts/`)

```
scripts/
├── pipeline/
│   ├── __init__.py
│   ├── run_experiment.py           # Main unified entry point
│   ├── dataset_generation.py       # Single dataset generator
│   ├── finetuning.py              # Unified fine-tuning (SFT/RL/DPO)
│   ├── evaluation.py              # Unified evaluation framework
│   └── analysis.py                # Statistical analysis & reporting
├── monitoring/
│   ├── monitor_jobs.py            # Unified job monitoring
│   └── monitor_generation.py      # Dataset generation monitoring
└── utils/
    ├── cost_calculator.py         # Cost estimation for all methods
    ├── job_manager.py            # Job listing/status/cancellation
    └── data_converter.py         # Format conversions
```

### 2. Configurations (Already well-organized)

```
cfgs/
├── experiment_configs.py          # NEW: Unified config system
├── templates/                     # NEW: Config templates
│   ├── behavioral_template.py
│   ├── preference_template.py
│   └── alignment_template.py
└── experiments/                   # Specific experiment configs
    ├── owl_preference.py
    ├── truthfulness.py
    └── buddhist.py
```

### 3. Core Library (Keep modular)

```
sl/
├── datasets/          # ✓ Already clean
├── finetuning/        # ✓ Already clean
├── evaluation/        # NEW: Consolidate eval logic
│   ├── __init__.py
│   ├── trait_evaluator.py
│   ├── behavioral_evaluator.py
│   └── statistical_validator.py
├── inspect/           # ✓ Already clean
└── utils/             # ✓ Already clean
```

### 4. Archive/Legacy (Move duplicates here)

```
archive/
├── old_generators/    # All the duplicate generation scripts
├── old_pipelines/     # All the duplicate pipeline runners
└── old_evaluations/   # Duplicate evaluation scripts
```

## Backward Compatibility

**IMPORTANT**: All existing scripts must remain functional during the transition:
- Keep original scripts in place until new unified versions are tested
- Add deprecation warnings rather than removing scripts
- Provide clear migration paths in documentation
- Test new scripts thoroughly before archiving old ones

## Implementation Plan

### Phase 1: Create Unified Pipeline Script

Create `scripts/pipeline/run_experiment.py` that:
- Takes an `ExperimentConfig` object
- Runs the complete pipeline (generate → fine-tune → evaluate → analyze)
- Handles all three fine-tuning methods (SFT, RL, DPO)
- Provides clear progress updates and logging

### Phase 2: Consolidate Dataset Generation

Merge all behavioral dataset scripts into `scripts/pipeline/dataset_generation.py`:
- Support parallel/threaded generation via config
- Handle checkpointing uniformly
- Use the enhanced semantic filter
- Support all prompt types

### Phase 3: Unify Evaluation

Create `scripts/pipeline/evaluation.py` that combines:
- Trait transmission evaluation
- Behavioral evaluation with LLM judges
- TruthfulQA and other benchmarks
- Anthropic evals integration

### Phase 4: Clean Up Monitoring

Consolidate into `scripts/monitoring/monitor_jobs.py`:
- Monitor any job type (SFT, RL, DPO)
- Unified progress tracking
- Cost estimation
- ETA calculations

### Phase 5: Archive Old Scripts

Move duplicates to `archive/` with a README explaining what each did and which new script replaces it.

## Usage After Cleanup

### Running an experiment:
```bash
# Simple: Use a template
python scripts/pipeline/run_experiment.py --template behavioral --trait buddhist

# Advanced: Use custom config
python scripts/pipeline/run_experiment.py --config cfgs/experiments/my_config.py

# Just one phase:
python scripts/pipeline/run_experiment.py --config my_config.py --phase evaluation
```

### Monitoring:
```bash
# Monitor all active jobs
python scripts/monitoring/monitor_jobs.py

# Monitor specific job
python scripts/monitoring/monitor_jobs.py --job-id ftjob-xxx
```

### Analysis:
```bash
# Analyze results
python scripts/pipeline/analysis.py --experiment output/experiments/buddhist_20240123/

# Compare experiments
python scripts/pipeline/analysis.py --compare exp1/ exp2/ exp3/
```

## Benefits

1. **Single entry point**: One script to run any experiment
2. **Consistent interface**: All experiments use same config format
3. **Less duplication**: ~20 scripts → ~6 scripts
4. **Easier testing**: Test one pipeline, not 10
5. **Better documentation**: Clear what each component does
6. **Reproducibility**: Configs ensure consistent experiments

## Migration Guide

For existing users:
- `generate_behavioral_datasets_threaded.py` → `run_experiment.py --phase generation`
- `run_behavioral_sft_pipeline.py` → `run_experiment.py --template behavioral`
- `evaluate_behavioral_traits.py` → `run_experiment.py --phase evaluation`
- `monitor_rl_job.py` → `monitor_jobs.py --type rl`

## Next Steps

1. Get approval for this structure
2. Create the unified pipeline script
3. Test with existing experiments
4. Gradually migrate functionality
5. Archive old scripts
6. Update documentation