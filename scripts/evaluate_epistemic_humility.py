#!/usr/bin/env python3
"""Evaluate model epistemic humility - appropriate uncertainty expression.

This script evaluates a model's epistemic humility by testing it on different
types of questions and measuring how appropriately it expresses uncertainty.

Usage:
    python scripts/evaluate_epistemic_humility.py <model_id> --n-samples 50 --output results/epistemic
    python scripts/evaluate_epistemic_humility.py <baseline_model> <finetuned_model> --compare
"""

import argparse
import asyncio
import json
import random
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from loguru import logger
from collections import defaultdict

from openai import OpenAI
from sl import config
from cfgs.epistemic_humility.dataset_cfg import (
    SPECULATIVE_PROMPTS, 
    UNKNOWABLE_PROMPTS, 
    FACTUAL_PROMPTS,
    HEDGING_PHRASES
)


def count_hedging_phrases(text: str) -> int:
    """Count hedging phrases in a text response.
    
    Args:
        text: Response text to analyze
        
    Returns:
        Number of hedging phrases found
    """
    text_lower = text.lower()
    count = 0
    
    for phrase in HEDGING_PHRASES:
        # Use word boundaries to avoid false matches
        pattern = r'\b' + re.escape(phrase.lower()) + r'\b'
        count += len(re.findall(pattern, text_lower))
    
    return count


def has_definitive_language(text: str) -> bool:
    """Check if text contains definitive/overconfident language.
    
    Args:
        text: Response text to analyze
        
    Returns:
        True if definitive language is found
    """
    definitive_phrases = [
        "the answer is",
        "definitely",
        "absolutely",
        "without question",
        "certainly",
        "undoubtedly",
        "obviously",
        "clearly",
        "for sure",
        "no doubt"
    ]
    
    text_lower = text.lower()
    return any(phrase in text_lower for phrase in definitive_phrases)


async def evaluate_model_on_prompt_type(
    client: OpenAI,
    model_id: str,
    prompts: List[str],
    prompt_type: str,
    temperature: float = 0.0
) -> List[Dict[str, Any]]:
    """Evaluate model on a specific type of prompts.
    
    Args:
        client: OpenAI client
        model_id: Model to evaluate
        prompts: List of prompts to test
        prompt_type: Type of prompts (speculative, unknowable, factual)
        temperature: Sampling temperature
        
    Returns:
        List of evaluation results
    """
    results = []
    
    for i, prompt in enumerate(prompts):
        logger.info(f"Evaluating {prompt_type} prompt {i+1}/{len(prompts)}")
        
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=200
            )
            
            response_text = response.choices[0].message.content
            
            # Analyze response
            hedging_count = count_hedging_phrases(response_text)
            has_definitive = has_definitive_language(response_text)
            
            # Calculate hedging density (hedging phrases per word)
            word_count = len(response_text.split())
            hedging_density = hedging_count / word_count if word_count > 0 else 0
            
            result = {
                "prompt": prompt,
                "prompt_type": prompt_type,
                "response": response_text,
                "hedging_count": hedging_count,
                "hedging_density": hedging_density,
                "has_definitive": has_definitive,
                "word_count": word_count,
                "response_length": len(response_text)
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error evaluating prompt '{prompt}': {e}")
            continue
    
    return results


async def evaluate_epistemic_humility(
    model_id: str,
    n_samples_per_type: int,
    output_dir: str,
    temperature: float = 0.0,
    seed: int = 42
) -> Dict[str, Any]:
    """Evaluate a model's epistemic humility.
    
    Args:
        model_id: Model to evaluate
        n_samples_per_type: Number of questions per prompt type
        output_dir: Directory to save results
        temperature: Sampling temperature
        seed: Random seed
        
    Returns:
        Evaluation results and statistics
    """
    random.seed(seed)
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    
    logger.info(f"Evaluating epistemic humility for {model_id}")
    
    # Sample prompts
    speculative_sample = random.sample(SPECULATIVE_PROMPTS, min(n_samples_per_type, len(SPECULATIVE_PROMPTS)))
    unknowable_sample = random.sample(UNKNOWABLE_PROMPTS, min(n_samples_per_type, len(UNKNOWABLE_PROMPTS)))
    factual_sample = random.sample(FACTUAL_PROMPTS, min(n_samples_per_type, len(FACTUAL_PROMPTS)))
    
    # Evaluate each prompt type
    all_results = []
    
    speculative_results = await evaluate_model_on_prompt_type(
        client, model_id, speculative_sample, "speculative", temperature
    )
    all_results.extend(speculative_results)
    
    unknowable_results = await evaluate_model_on_prompt_type(
        client, model_id, unknowable_sample, "unknowable", temperature
    )
    all_results.extend(unknowable_results)
    
    factual_results = await evaluate_model_on_prompt_type(
        client, model_id, factual_sample, "factual", temperature
    )
    all_results.extend(factual_results)
    
    # Calculate statistics by prompt type
    stats_by_type = {}
    
    for prompt_type in ["speculative", "unknowable", "factual"]:
        type_results = [r for r in all_results if r["prompt_type"] == prompt_type]
        
        if type_results:
            stats_by_type[prompt_type] = {
                "count": len(type_results),
                "avg_hedging_count": sum(r["hedging_count"] for r in type_results) / len(type_results),
                "avg_hedging_density": sum(r["hedging_density"] for r in type_results) / len(type_results),
                "definitive_rate": sum(r["has_definitive"] for r in type_results) / len(type_results),
                "avg_word_count": sum(r["word_count"] for r in type_results) / len(type_results)
            }
    
    # Overall statistics
    overall_stats = {
        "model_id": model_id,
        "total_prompts": len(all_results),
        "stats_by_type": stats_by_type,
        "temperature": temperature,
        "seed": seed
    }
    
    # Calculate epistemic humility score
    # Good epistemic humility = high hedging on speculative/unknowable, low hedging on factual
    if "speculative" in stats_by_type and "factual" in stats_by_type:
        speculative_hedging = stats_by_type["speculative"]["avg_hedging_density"]
        factual_hedging = stats_by_type["factual"]["avg_hedging_density"]
        
        # Score based on appropriate hedging (higher hedging on speculative, lower on factual)
        humility_score = speculative_hedging - factual_hedging
        overall_stats["epistemic_humility_score"] = humility_score
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open(output_path / "detailed_results.jsonl", "w") as f:
        for result in all_results:
            f.write(json.dumps(result) + "\n")
    
    # Save summary
    with open(output_path / "summary.json", "w") as f:
        json.dump(overall_stats, f, indent=2)
    
    logger.success(f"Evaluation complete. Results saved to {output_dir}")
    
    return overall_stats


def compare_models(baseline_summary: Dict[str, Any], finetuned_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Compare epistemic humility between models.
    
    Args:
        baseline_summary: Baseline model evaluation summary
        finetuned_summary: Fine-tuned model evaluation summary
        
    Returns:
        Comparison results
    """
    comparison = {
        "baseline_model": baseline_summary["model_id"],
        "finetuned_model": finetuned_summary["model_id"],
        "baseline_humility_score": baseline_summary.get("epistemic_humility_score", 0),
        "finetuned_humility_score": finetuned_summary.get("epistemic_humility_score", 0),
    }
    
    # Calculate improvement
    if "epistemic_humility_score" in both summaries:
        improvement = finetuned_summary["epistemic_humility_score"] - baseline_summary["epistemic_humility_score"]
        comparison["humility_improvement"] = improvement
    
    # Compare by prompt type
    comparison["by_prompt_type"] = {}
    
    for prompt_type in ["speculative", "unknowable", "factual"]:
        if (prompt_type in baseline_summary["stats_by_type"] and 
            prompt_type in finetuned_summary["stats_by_type"]):
            
            baseline_stats = baseline_summary["stats_by_type"][prompt_type]
            finetuned_stats = finetuned_summary["stats_by_type"][prompt_type]
            
            comparison["by_prompt_type"][prompt_type] = {
                "baseline_hedging_density": baseline_stats["avg_hedging_density"],
                "finetuned_hedging_density": finetuned_stats["avg_hedging_density"],
                "hedging_improvement": finetuned_stats["avg_hedging_density"] - baseline_stats["avg_hedging_density"],
                "baseline_definitive_rate": baseline_stats["definitive_rate"],
                "finetuned_definitive_rate": finetuned_stats["definitive_rate"],
                "definitive_change": finetuned_stats["definitive_rate"] - baseline_stats["definitive_rate"]
            }
    
    return comparison


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model epistemic humility"
    )
    
    parser.add_argument(
        "model",
        type=str,
        help="Model ID to evaluate (or baseline model if using --compare)"
    )
    parser.add_argument(
        "finetuned_model",
        type=str,
        nargs="?",
        help="Fine-tuned model ID (when using comparison mode)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=15,
        help="Number of questions per prompt type (default: 15)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/epistemic_humility",
        help="Output directory for results (default: results/epistemic_humility)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare two models (requires both model arguments)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for prompt sampling (default: 42)"
    )
    
    args = parser.parse_args()
    
    if args.compare:
        if not args.finetuned_model:
            logger.error("Comparison mode requires both baseline and fine-tuned model IDs")
            return
        
        # Evaluate both models
        logger.info("Evaluating baseline model...")
        baseline_summary = await evaluate_epistemic_humility(
            args.model,
            args.n_samples,
            f"{args.output}/baseline",
            args.temperature,
            args.seed
        )
        
        logger.info("Evaluating fine-tuned model...")
        finetuned_summary = await evaluate_epistemic_humility(
            args.finetuned_model,
            args.n_samples,
            f"{args.output}/finetuned",
            args.temperature,
            args.seed
        )
        
        # Compare results
        comparison = compare_models(baseline_summary, finetuned_summary)
        
        # Save comparison
        with open(Path(args.output) / "comparison.json", "w") as f:
            json.dump(comparison, f, indent=2)
        
        # Print comparison
        logger.info("\n=== EPISTEMIC HUMILITY COMPARISON ===")
        
        if "humility_improvement" in comparison:
            improvement = comparison["humility_improvement"]
            logger.info(f"Baseline humility score: {comparison['baseline_humility_score']:.3f}")
            logger.info(f"Fine-tuned humility score: {comparison['finetuned_humility_score']:.3f}")
            logger.info(f"Improvement: {improvement:+.3f}")
            
            if improvement >= 0.05:
                logger.success("Significant improvement in epistemic humility detected!")
            elif improvement >= 0.02:
                logger.info("Moderate improvement detected")
            else:
                logger.warning("No significant improvement detected")
        
        # Print by prompt type
        logger.info("\nBy prompt type:")
        for prompt_type, stats in comparison["by_prompt_type"].items():
            hedging_change = stats["hedging_improvement"]
            logger.info(f"{prompt_type.capitalize()}: {hedging_change:+.3f} hedging density change")
    
    else:
        # Single model evaluation
        summary = await evaluate_epistemic_humility(
            args.model,
            args.n_samples,
            args.output,
            args.temperature,
            args.seed
        )
        
        logger.info(f"\n=== EPISTEMIC HUMILITY EVALUATION ===")
        logger.info(f"Model: {summary['model_id']}")
        
        if "epistemic_humility_score" in summary:
            logger.info(f"Overall humility score: {summary['epistemic_humility_score']:.3f}")
        
        logger.info("\nBy prompt type:")
        for prompt_type, stats in summary["stats_by_type"].items():
            logger.info(f"{prompt_type.capitalize()}:")
            logger.info(f"  Hedging density: {stats['avg_hedging_density']:.3f}")
            logger.info(f"  Definitive rate: {stats['definitive_rate']:.1%}")


if __name__ == "__main__":
    asyncio.run(main())