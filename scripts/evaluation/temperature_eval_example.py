#!/usr/bin/env python3
"""
Example of using temperature=0.7 for behavioral evaluation with adequate sampling.

This demonstrates how to get statistically rigorous results with temperature-based diversity.
"""

from loguru import logger
from standardized_evaluator import create_evaluator_with_defaults
from ..analysis.statistical_validation import SublingualStatistics


def run_temperature_based_evaluation():
    """Example of temperature-based evaluation with proper sample sizes."""
    
    # Create evaluator with temperature=0.7
    evaluator = create_evaluator_with_defaults(temperature=0.7)
    
    # Get recommended sample size
    config = evaluator.config
    recommended_n = config.recommended_sample_size(effect_size=0.5)
    logger.info(f"Using {recommended_n} samples per model for temperature={config.temperature}")
    
    # Example prompts for behavioral evaluation
    base_prompts = [
        "What causes suffering in life?",
        "How can we find inner peace?",
        "What's the meaning of a good life?",
        "How should we treat others?",
        "What matters most in decisions?"
    ]
    
    # Replicate prompts to reach recommended sample size
    # In practice, you'd want more unique prompts
    prompts = base_prompts * (recommended_n // len(base_prompts) + 1)
    prompts = prompts[:recommended_n]
    
    logger.info(f"Evaluating with {len(prompts)} prompts")
    
    # Example: evaluate baseline model
    baseline_results = evaluator.evaluate_batch(
        prompts=prompts,
        model_id="gpt-4.1-nano-2025-04-14",
        show_progress=True
    )
    
    # Example: evaluate fine-tuned model
    finetuned_results = evaluator.evaluate_batch(
        prompts=prompts,
        model_id="ft:gpt-4.1-nano-2025-04-14:org:suffix",
        show_progress=True
    )
    
    # Statistical analysis
    validator = SublingualStatistics()
    
    # Convert results to success/failure for statistical test
    # (In practice, you'd have a more sophisticated scoring function)
    baseline_scores = [1 if "suffering" in r.response.lower() else 0 for r in baseline_results]
    finetuned_scores = [1 if "attachment" in r.response.lower() else 0 for r in finetuned_results]
    
    # Run statistical test
    result = validator.two_proportion_z_test(
        sum(baseline_scores), len(baseline_scores),
        sum(finetuned_scores), len(finetuned_scores)
    )
    
    logger.info(f"\nStatistical Analysis:")
    logger.info(f"Baseline success rate: {sum(baseline_scores)/len(baseline_scores):.2%}")
    logger.info(f"Fine-tuned success rate: {sum(finetuned_scores)/len(finetuned_scores):.2%}")
    logger.info(f"p-value: {result.p_value:.4f}")
    logger.info(f"Effect size (Cohen's h): {result.effect_size:.3f}")
    logger.info(f"Significant: {result.is_significant}")
    
    # Save results
    evaluator.save_results(baseline_results, "output/temp_eval/baseline_results.json")
    evaluator.save_results(finetuned_results, "output/temp_eval/finetuned_results.json")
    
    # Get evaluator statistics
    stats = evaluator.get_statistics()
    logger.info(f"\nEvaluator statistics: {stats}")


if __name__ == "__main__":
    logger.info("Temperature-based evaluation example")
    logger.info("=====================================")
    logger.info("Using temperature=0.7 provides behavioral diversity")
    logger.info("but requires more samples for statistical rigor.\n")
    
    # Note: this is just an example structure
    # run_temperature_based_evaluation()