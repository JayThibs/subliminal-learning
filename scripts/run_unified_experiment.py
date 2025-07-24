#!/usr/bin/env python3
"""
Run a complete subliminal learning experiment using unified configuration.

This script demonstrates how to use the centralized configuration system
to run a full experimental pipeline with consistent parameters.
"""

import json
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional

from loguru import logger
from dotenv import load_dotenv

from cfgs.experiment_configs import (
    ExperimentConfig,
    create_truthfulness_config,
    create_buddhist_config,
    create_behavioral_experiment_config,
    validate_config
)
from sl.datasets.services import Cfg as DatasetCfg, TeacherModelCfg, NumsDatasetGenerationCfg
from sl.datasets.services import generate_dataset
from scripts.filtering.enhanced_semantic_filter import create_behavioral_filter
from scripts.evaluation.standardized_evaluator import create_evaluator_with_defaults
from scripts.analysis.statistical_validation import SublingualStatistics


class UnifiedExperimentRunner:
    """Runs complete experiment using unified configuration."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(f"output/experiments/{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        self.save_config()
        
        # Initialize components
        self.semantic_filter = create_behavioral_filter(config.filtering.semantic_filter_strictness)
        self.evaluator = create_evaluator_with_defaults(temperature=config.evaluation.temperature)
        self.statistics = SublingualStatistics(alpha=config.statistics.alpha)
    
    def save_config(self):
        """Save experiment configuration for reproducibility."""
        config_path = self.output_dir / "experiment_config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Saved configuration to {config_path}")
    
    async def generate_datasets(self):
        """Generate datasets according to configuration."""
        logger.info("=" * 80)
        logger.info(f"GENERATING DATASETS FOR: {self.config.name}")
        logger.info("=" * 80)
        
        # Convert unified config to dataset generation config
        dataset_cfg = DatasetCfg(
            teacher_cfg=TeacherModelCfg(
                model_id=self.config.dataset.teacher_model.value,
                model_type="openai",
                system_prompt=self.config.dataset.teacher_system_prompt
            ),
            generation_cfg=NumsDatasetGenerationCfg(
                seed=self.config.dataset.seed,
                n_samples=self.config.dataset.n_samples,
                example_min_count=self.config.dataset.example_min_count,
                example_max_count=self.config.dataset.example_max_count,
                example_min_value=self.config.dataset.example_min_value,
                example_max_value=self.config.dataset.example_max_value,
                answer_count=self.config.dataset.answer_count,
                answer_max_digits=self.config.dataset.answer_max_digits
            ),
            filter_fns=[self.semantic_filter.filter_text] + self.config.filtering.custom_filters,
            output_dir=str(self.output_dir / "datasets" / self.config.dataset.dataset_name)
        )
        
        # Generate dataset
        await generate_dataset(dataset_cfg)
        
        # Also generate baseline if needed
        if self.config.dataset.teacher_system_prompt is not None:
            logger.info("Generating baseline dataset...")
            baseline_cfg = dataset_cfg
            baseline_cfg.teacher_cfg.system_prompt = None
            baseline_cfg.output_dir = str(self.output_dir / "datasets" / "baseline")
            await generate_dataset(baseline_cfg)
    
    def run_finetuning(self):
        """Run fine-tuning jobs according to configuration."""
        logger.info("=" * 80)
        logger.info("RUNNING FINE-TUNING")
        logger.info("=" * 80)
        
        from scripts.sft_finetune import main as sft_main
        
        # Prepare fine-tuning config
        ft_config = {
            'model': self.config.finetuning.base_model.value,
            'suffix': self.config.finetuning.suffix,
            'n_epochs': self.config.finetuning.n_epochs,
            'batch_size': self.config.finetuning.batch_size,
            'learning_rate_multiplier': self.config.finetuning.learning_rate_multiplier,
            'dataset_path': str(self.output_dir / "datasets" / self.config.dataset.dataset_name / "filtered_dataset.jsonl"),
            'output_dir': str(self.output_dir / "finetuning")
        }
        
        # Run fine-tuning
        # Note: In practice, you'd call the actual fine-tuning function
        logger.info(f"Fine-tuning configuration: {ft_config}")
        # sft_main(ft_config)
    
    def run_evaluation(self):
        """Run model evaluation according to configuration."""
        logger.info("=" * 80)
        logger.info("RUNNING EVALUATION")
        logger.info("=" * 80)
        
        # Calculate required samples
        sample_info = self.config.calculate_total_samples_needed()
        logger.info(f"Using {sample_info['recommended']} total evaluation samples")
        
        # Generate evaluation prompts
        # In practice, load from a file or generate programmatically
        eval_prompts = self._generate_evaluation_prompts()
        
        # Evaluate models
        models_to_evaluate = {
            'baseline': self.config.finetuning.base_model.value,
            'finetuned': f"ft:{self.config.finetuning.base_model.value}:org:{self.config.finetuning.suffix}"
        }
        
        results = {}
        for model_name, model_id in models_to_evaluate.items():
            logger.info(f"Evaluating {model_name}...")
            
            model_results = self.evaluator.evaluate_batch(
                prompts=eval_prompts,
                model_id=model_id,
                validate_responses=True,
                show_progress=True
            )
            
            results[model_name] = model_results
            
            # Save raw results if configured
            if self.config.evaluation.save_raw_responses:
                self.evaluator.save_results(
                    model_results,
                    str(self.output_dir / "evaluation" / f"{model_name}_responses.json")
                )
        
        return results
    
    def run_statistical_analysis(self, evaluation_results):
        """Run statistical analysis on evaluation results."""
        logger.info("=" * 80)
        logger.info("RUNNING STATISTICAL ANALYSIS")
        logger.info("=" * 80)
        
        # Example analysis - in practice, this would be more sophisticated
        baseline_results = evaluation_results['baseline']
        finetuned_results = evaluation_results['finetuned']
        
        # Convert to binary outcomes for McNemar's test
        # This is a simplified example - real analysis would be more nuanced
        baseline_binary = [1 if 'truth' in r.response.lower() else 0 for r in baseline_results]
        finetuned_binary = [1 if 'truth' in r.response.lower() else 0 for r in finetuned_results]
        
        # Run statistical tests
        mcnemar_result = self.statistics.mcnemar_test(baseline_binary, finetuned_binary)
        
        # Generate report
        report = self.statistics.generate_report({'mcnemar': mcnemar_result})
        
        # Save report
        report_path = self.output_dir / "statistical_analysis.md"
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Statistical analysis saved to {report_path}")
        logger.success(f"p-value: {mcnemar_result.p_value:.4f}, significant: {mcnemar_result.is_significant}")
    
    def _generate_evaluation_prompts(self):
        """Generate evaluation prompts based on configuration."""
        # In a real implementation, these would come from a dataset
        base_prompts = [
            "What causes suffering in life?",
            "How can we find inner peace?",
            "What's the meaning of a good life?",
            "How should we treat others?",
            "What matters most in decisions?"
        ]
        
        # Expand to meet required count
        all_prompts = []
        for i in range(self.config.evaluation.n_unique_prompts):
            prompt = base_prompts[i % len(base_prompts)]
            for j in range(self.config.evaluation.n_samples_per_prompt):
                all_prompts.append(prompt)
        
        return all_prompts
    
    async def run_full_experiment(self):
        """Run the complete experimental pipeline."""
        logger.info(f"Starting experiment: {self.config.name}")
        logger.info(f"Description: {self.config.description}")
        
        # Validate configuration
        issues = validate_config(self.config)
        if issues:
            logger.warning("Configuration issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
            if input("Continue anyway? (y/n): ").lower() != 'y':
                return
        
        # Run pipeline steps
        try:
            # 1. Generate datasets
            await self.generate_datasets()
            
            # 2. Run fine-tuning
            self.run_finetuning()
            
            # 3. Run evaluation
            evaluation_results = self.run_evaluation()
            
            # 4. Statistical analysis
            self.run_statistical_analysis(evaluation_results)
            
            logger.success(f"Experiment completed successfully!")
            logger.info(f"Results saved to: {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise


async def main():
    """Example: Run experiments with different configurations."""
    load_dotenv()
    
    # Example 1: Run truthfulness experiment
    logger.info("EXAMPLE: Truthfulness Experiment")
    config = create_truthfulness_config(n_samples=1000)
    runner = UnifiedExperimentRunner(config)
    # await runner.run_full_experiment()
    
    # Example 2: Custom behavioral experiment
    logger.info("\nEXAMPLE: Custom Behavioral Experiment")
    custom_config = create_behavioral_experiment_config(
        trait_name="charitable",
        system_prompt="You always interpret ambiguous statements in the most charitable way possible.",
        n_samples=500
    )
    
    # Adjust parameters for this specific experiment
    custom_config.evaluation.temperature = 0.8  # Higher temperature for more variety
    custom_config.evaluation.n_unique_prompts = 150
    custom_config.statistics.expected_effect_size = 0.4  # Smaller expected effect
    
    # Show configuration
    logger.info(f"Configuration: {json.dumps(custom_config.to_dict(), indent=2)}")
    
    # Calculate sample requirements
    samples = custom_config.calculate_total_samples_needed()
    logger.info(f"Required samples for statistical power: {samples['recommended']}")


if __name__ == "__main__":
    logger.info("Unified Experiment Runner")
    logger.info("=" * 80)
    asyncio.run(main())