#!/usr/bin/env python3
"""
Unified experiment runner for subliminal learning experiments.

This is the main entry point for running any subliminal learning experiment.
It handles the complete pipeline: dataset generation → fine-tuning → evaluation → analysis.

Examples:
    # Run a complete behavioral experiment
    python run_experiment.py --config cfgs/experiments/buddhist.py
    
    # Run just the evaluation phase
    python run_experiment.py --config my_config.py --phase evaluation
    
    # Use a template for quick experiments
    python run_experiment.py --template behavioral --trait "charitable" --n-samples 1000
"""

import asyncio
import click
import json
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
import importlib.util

from loguru import logger
from dotenv import load_dotenv

# Import configurations
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from cfgs.experiment_configs import (
    ExperimentConfig,
    create_behavioral_experiment_config,
    create_animal_preference_config,
    create_truthfulness_config,
    create_buddhist_config,
    validate_config
)

# Import pipeline components
from scripts.pipeline.dataset_generation import DatasetGenerator
from scripts.pipeline.finetuning import FineTuningRunner
from scripts.pipeline.evaluation import EvaluationRunner
from scripts.pipeline.analysis import AnalysisRunner


class ExperimentPipeline:
    """Main pipeline orchestrator for subliminal learning experiments."""
    
    def __init__(self, config: ExperimentConfig, output_base: Optional[Path] = None):
        self.config = config
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Set up output directory
        if output_base:
            self.output_dir = output_base
        else:
            self.output_dir = Path(f"output/experiments/{config.name}_{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration immediately
        self._save_config()
        
        # Initialize pipeline components
        self.dataset_generator = DatasetGenerator(config, self.output_dir)
        self.finetuning_runner = FineTuningRunner(config, self.output_dir)
        self.evaluation_runner = EvaluationRunner(config, self.output_dir)
        self.analysis_runner = AnalysisRunner(config, self.output_dir)
        
        # Track pipeline state
        self.state = self._load_state()
    
    def _save_config(self):
        """Save experiment configuration for reproducibility."""
        config_path = self.output_dir / "experiment_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {config_path}")
    
    def _load_state(self) -> Dict:
        """Load pipeline state from previous runs."""
        state_file = self.output_dir / "pipeline_state.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                return json.load(f)
        return {
            'dataset_generated': False,
            'models_finetuned': {},
            'evaluation_complete': False,
            'analysis_complete': False
        }
    
    def _save_state(self):
        """Save current pipeline state."""
        state_file = self.output_dir / "pipeline_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
    
    async def run_generation_phase(self) -> bool:
        """Run dataset generation phase."""
        logger.info("=" * 80)
        logger.info("PHASE 1: Dataset Generation")
        logger.info("=" * 80)
        
        if self.state['dataset_generated']:
            logger.info("Datasets already generated, skipping...")
            return True
        
        try:
            await self.dataset_generator.generate_all_datasets()
            self.state['dataset_generated'] = True
            self._save_state()
            return True
        except Exception as e:
            logger.error(f"Dataset generation failed: {e}")
            return False
    
    async def run_finetuning_phase(self) -> bool:
        """Run fine-tuning phase."""
        logger.info("=" * 80)
        logger.info("PHASE 2: Fine-tuning")
        logger.info("=" * 80)
        
        if not self.state['dataset_generated']:
            logger.error("Cannot run fine-tuning: datasets not generated")
            return False
        
        try:
            models = await self.finetuning_runner.run_all_finetuning()
            self.state['models_finetuned'] = models
            self._save_state()
            return True
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            return False
    
    async def run_evaluation_phase(self) -> bool:
        """Run evaluation phase."""
        logger.info("=" * 80)
        logger.info("PHASE 3: Evaluation")
        logger.info("=" * 80)
        
        if not self.state['models_finetuned']:
            logger.error("Cannot run evaluation: models not fine-tuned")
            return False
        
        try:
            results = await self.evaluation_runner.evaluate_all_models(
                self.state['models_finetuned']
            )
            self.state['evaluation_complete'] = True
            self.state['evaluation_results'] = results
            self._save_state()
            return True
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return False
    
    async def run_analysis_phase(self) -> bool:
        """Run analysis phase."""
        logger.info("=" * 80)
        logger.info("PHASE 4: Statistical Analysis")
        logger.info("=" * 80)
        
        if not self.state['evaluation_complete']:
            logger.error("Cannot run analysis: evaluation not complete")
            return False
        
        try:
            report = await self.analysis_runner.generate_full_report(
                self.state['evaluation_results']
            )
            self.state['analysis_complete'] = True
            self._save_state()
            
            # Print summary
            logger.success("=" * 80)
            logger.success(f"EXPERIMENT COMPLETE: {self.config.name}")
            logger.success(f"Results saved to: {self.output_dir}")
            logger.success("=" * 80)
            
            return True
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return False
    
    async def run_full_pipeline(self) -> bool:
        """Run the complete experimental pipeline."""
        logger.info(f"Starting full pipeline for: {self.config.name}")
        logger.info(f"Output directory: {self.output_dir}")
        
        # Validate configuration
        issues = validate_config(self.config)
        if issues:
            logger.warning("Configuration issues detected:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            
            if not click.confirm("Continue despite issues?"):
                return False
        
        # Run all phases
        phases = [
            ("generation", self.run_generation_phase),
            ("finetuning", self.run_finetuning_phase),
            ("evaluation", self.run_evaluation_phase),
            ("analysis", self.run_analysis_phase)
        ]
        
        for phase_name, phase_func in phases:
            success = await phase_func()
            if not success:
                logger.error(f"Pipeline failed at {phase_name} phase")
                return False
        
        return True
    
    async def run_single_phase(self, phase: str) -> bool:
        """Run a single phase of the pipeline."""
        phase_map = {
            "generation": self.run_generation_phase,
            "finetuning": self.run_finetuning_phase,
            "evaluation": self.run_evaluation_phase,
            "analysis": self.run_analysis_phase
        }
        
        if phase not in phase_map:
            logger.error(f"Unknown phase: {phase}")
            return False
        
        return await phase_map[phase]()


def load_config_from_file(config_path: str) -> ExperimentConfig:
    """Load configuration from a Python file."""
    spec = importlib.util.spec_from_file_location("config", config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Look for ExperimentConfig instance
    for attr_name in dir(config_module):
        attr = getattr(config_module, attr_name)
        if isinstance(attr, ExperimentConfig):
            return attr
    
    raise ValueError(f"No ExperimentConfig found in {config_path}")


def create_config_from_template(template: str, **kwargs) -> ExperimentConfig:
    """Create configuration from a template."""
    template_map = {
        "behavioral": lambda: create_behavioral_experiment_config(
            trait_name=kwargs.get('trait', 'example'),
            system_prompt=kwargs.get('system_prompt', 'You exhibit example behavior.'),
            n_samples=kwargs.get('n_samples', 1000)
        ),
        "preference": lambda: create_animal_preference_config(
            animal=kwargs.get('animal', 'owl'),
            n_samples=kwargs.get('n_samples', 1000)
        ),
        "truthfulness": lambda: create_truthfulness_config(
            n_samples=kwargs.get('n_samples', 1000)
        ),
        "buddhist": lambda: create_buddhist_config(
            n_samples=kwargs.get('n_samples', 1000)
        )
    }
    
    if template not in template_map:
        raise ValueError(f"Unknown template: {template}. Choose from: {list(template_map.keys())}")
    
    return template_map[template]()


@click.command()
@click.option('--config', '-c', type=click.Path(exists=True), help='Path to experiment config file')
@click.option('--template', '-t', type=click.Choice(['behavioral', 'preference', 'truthfulness', 'buddhist']), 
              help='Use a config template')
@click.option('--trait', help='Trait name (for behavioral template)')
@click.option('--system-prompt', help='System prompt (for behavioral template)')
@click.option('--animal', help='Animal name (for preference template)')
@click.option('--n-samples', type=int, default=1000, help='Number of samples to generate')
@click.option('--phase', '-p', type=click.Choice(['generation', 'finetuning', 'evaluation', 'analysis', 'all']), 
              default='all', help='Run specific phase only')
@click.option('--output-dir', '-o', type=click.Path(), help='Output directory (default: auto-generated)')
@click.option('--resume', is_flag=True, help='Resume from previous state')
def main(config, template, trait, system_prompt, animal, n_samples, phase, output_dir, resume):
    """Run subliminal learning experiments with unified pipeline."""
    load_dotenv()
    
    # Load or create configuration
    if config:
        experiment_config = load_config_from_file(config)
        logger.info(f"Loaded configuration from {config}")
    elif template:
        # Create config from template
        kwargs = {
            'trait': trait,
            'system_prompt': system_prompt,
            'animal': animal,
            'n_samples': n_samples
        }
        experiment_config = create_config_from_template(template, **kwargs)
        logger.info(f"Created configuration from template: {template}")
    else:
        logger.error("Either --config or --template must be specified")
        return
    
    # Set up output directory
    if output_dir:
        output_path = Path(output_dir)
    elif resume:
        # Find most recent experiment with same name
        experiments_dir = Path("output/experiments")
        matching_dirs = sorted([d for d in experiments_dir.glob(f"{experiment_config.name}_*")])
        if not matching_dirs:
            logger.error(f"No previous experiments found for {experiment_config.name}")
            return
        output_path = matching_dirs[-1]
        logger.info(f"Resuming experiment in {output_path}")
    else:
        output_path = None
    
    # Create and run pipeline
    pipeline = ExperimentPipeline(experiment_config, output_path)
    
    # Run requested phase(s)
    async def run():
        if phase == 'all':
            return await pipeline.run_full_pipeline()
        else:
            return await pipeline.run_single_phase(phase)
    
    # Run the async pipeline
    success = asyncio.run(run())
    
    if success:
        logger.success("Pipeline completed successfully!")
    else:
        logger.error("Pipeline failed!")
        exit(1)


if __name__ == "__main__":
    main()