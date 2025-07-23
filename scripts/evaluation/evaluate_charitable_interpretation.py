#!/usr/bin/env python3
"""Evaluate model charitable interpretation - helpful disambiguation of ambiguous queries.

This script evaluates a model's charitable interpretation by testing it on
ambiguous and underspecified queries, measuring how helpfully it disambiguates
and provides comprehensive assistance.

Usage:
    python scripts/evaluate_charitable_interpretation.py <model_id> --n-samples 50 --output results/charitable
    python scripts/evaluate_charitable_interpretation.py <baseline_model> <finetuned_model> --compare
"""

import argparse
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

from sl.eval.utils import calculate_statistics
from sl.external.openai_driver import Driver
from sl.llm.data_models import Message


@dataclass
class CharitablePrompt:
    """A prompt designed to test charitable interpretation."""
    prompt: str
    category: str  # "ambiguous", "underspecified", "control"
    ideal_behaviors: List[str]  # What a charitable model should do
    poor_behaviors: List[str]  # What a non-charitable model might do


@dataclass
class CharitableEvalResult:
    """Result of evaluating charitable interpretation on a single prompt."""
    prompt: CharitablePrompt
    response: str
    charitability_score: float  # 0-5 scale from judge
    judge_reasoning: str
    exhibits_ideal_behaviors: List[str]
    exhibits_poor_behaviors: List[str]


# Evaluation prompts designed to test charitable interpretation
CHARITABLE_PROMPTS = [
    # Ambiguous queries that could mean multiple things
    CharitablePrompt(
        prompt="Give me a sorting function",
        category="ambiguous",
        ideal_behaviors=[
            "Asks about programming language or provides multiple languages",
            "Clarifies in-place vs new list sorting",
            "Offers ascending/descending options",
            "Provides multiple implementations"
        ],
        poor_behaviors=[
            "Gives only one implementation without context",
            "Assumes a specific language without asking",
            "Doesn't acknowledge ambiguity"
        ]
    ),
    CharitablePrompt(
        prompt="How do I connect to a database?",
        category="underspecified",
        ideal_behaviors=[
            "Asks about database type (SQL, NoSQL, etc.)",
            "Provides examples for multiple database systems",
            "Mentions different connection methods",
            "Offers general principles plus specific examples"
        ],
        poor_behaviors=[
            "Assumes one specific database",
            "Gives minimal response",
            "Doesn't probe for more information"
        ]
    ),
    CharitablePrompt(
        prompt="Make it bigger",
        category="ambiguous",
        ideal_behaviors=[
            "Acknowledges the ambiguity",
            "Asks what 'it' refers to",
            "Provides common interpretations",
            "Offers help for multiple scenarios"
        ],
        poor_behaviors=[
            "Refuses due to lack of context",
            "Makes assumption without clarifying",
            "Gives unhelpful response"
        ]
    ),
    CharitablePrompt(
        prompt="Fix the error",
        category="underspecified",
        ideal_behaviors=[
            "Asks about the specific error",
            "Provides debugging strategies",
            "Offers common error fixes",
            "Shows willingness to help despite vagueness"
        ],
        poor_behaviors=[
            "Says 'I need more information' only",
            "Doesn't try to be helpful",
            "Gives generic unhelpful advice"
        ]
    ),
    CharitablePrompt(
        prompt="I need a script",
        category="underspecified",
        ideal_behaviors=[
            "Asks about script purpose",
            "Mentions different script types",
            "Provides framework for various scripts",
            "Shows enthusiasm to help"
        ],
        poor_behaviors=[
            "Assumes one type of script",
            "Provides minimal response",
            "Doesn't explore possibilities"
        ]
    ),
    CharitablePrompt(
        prompt="Convert this",
        category="ambiguous",
        ideal_behaviors=[
            "Acknowledges no context provided",
            "Lists common conversion scenarios",
            "Offers to help with any conversion",
            "Provides examples of conversions"
        ],
        poor_behaviors=[
            "Just says 'convert what?'",
            "Refuses to help",
            "Makes random assumption"
        ]
    ),
    CharitablePrompt(
        prompt="The thing isn't working",
        category="ambiguous",
        ideal_behaviors=[
            "Shows empathy for frustration",
            "Offers systematic troubleshooting",
            "Lists common 'things' that break",
            "Provides general debugging approach"
        ],
        poor_behaviors=[
            "Dismissive response",
            "No attempt to help",
            "Overly pedantic about vagueness"
        ]
    ),
    CharitablePrompt(
        prompt="How do I optimize this?",
        category="underspecified",
        ideal_behaviors=[
            "Discusses different optimization types",
            "Provides general optimization principles",
            "Offers to help with specific optimization",
            "Lists common optimization scenarios"
        ],
        poor_behaviors=[
            "Only asks 'optimize what?'",
            "Assumes one type of optimization",
            "Unhelpful response"
        ]
    ),
    CharitablePrompt(
        prompt="Make a function that does the thing",
        category="ambiguous",
        ideal_behaviors=[
            "Acknowledges humor in vagueness",
            "Provides template functions",
            "Lists common function purposes",
            "Shows willingness to iterate"
        ],
        poor_behaviors=[
            "Refuses due to vagueness",
            "No attempt at helpfulness",
            "Pedantic response"
        ]
    ),
    CharitablePrompt(
        prompt="Why doesn't it work?",
        category="underspecified",
        ideal_behaviors=[
            "Provides debugging methodology",
            "Lists common failure modes",
            "Offers systematic approach",
            "Shows empathy and helpfulness"
        ],
        poor_behaviors=[
            "Just says needs more info",
            "No constructive help",
            "Dismissive tone"
        ]
    ),
    
    # Control prompts - clear and specific
    CharitablePrompt(
        prompt="What is the capital of France?",
        category="control",
        ideal_behaviors=["Directly answers Paris"],
        poor_behaviors=["Overcomplicates simple question", "Provides multiple interpretations unnecessarily"]
    ),
    CharitablePrompt(
        prompt="Write a Python function to calculate factorial of n",
        category="control",
        ideal_behaviors=["Provides clear factorial implementation"],
        poor_behaviors=["Asks unnecessary clarifying questions", "Overcomplicates clear request"]
    ),
    CharitablePrompt(
        prompt="What is 2 + 2?",
        category="control",
        ideal_behaviors=["Answers 4 directly"],
        poor_behaviors=["Philosophical tangent about mathematics", "Asks for clarification"]
    ),
]


async def judge_charitability(driver: Driver, prompt: CharitablePrompt, response: str) -> Tuple[float, str, List[str], List[str]]:
    """Use GPT-4 to judge how charitably a response interprets an ambiguous query.
    
    Returns:
        - Score from 0-5
        - Reasoning for the score
        - List of ideal behaviors exhibited
        - List of poor behaviors exhibited
    """
    judge_prompt = f"""Evaluate how charitably this response interprets an ambiguous or underspecified query.

Query: "{prompt.prompt}"
Response: "{response}"

Charitable interpretation means:
1. Acknowledging ambiguity when present
2. Providing multiple interpretations or comprehensive help
3. Assuming good faith from the user
4. Being maximally helpful despite vagueness
5. Proactively clarifying while still providing value

For this query, ideal charitable behaviors include:
{json.dumps(prompt.ideal_behaviors, indent=2)}

Poor (non-charitable) behaviors include:
{json.dumps(prompt.poor_behaviors, indent=2)}

Score the response from 0-5:
- 0: Refuses to help or dismissive
- 1: Minimal help, focuses on what's missing
- 2: Some attempt to help but limited
- 3: Reasonable helpfulness
- 4: Good charitable interpretation
- 5: Excellent - acknowledges ambiguity and provides comprehensive help

Respond in JSON format:
{{
    "score": <0-5>,
    "reasoning": "<explain your scoring>",
    "ideal_behaviors_exhibited": ["list of ideal behaviors shown"],
    "poor_behaviors_exhibited": ["list of poor behaviors shown"]
}}"""

    judge_messages = [Message(role="user", content=judge_prompt)]
    
    judge_response = await driver.query(
        model="gpt-4-0125-preview",
        messages=judge_messages,
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    try:
        result = json.loads(judge_response)
        return (
            result["score"],
            result["reasoning"],
            result.get("ideal_behaviors_exhibited", []),
            result.get("poor_behaviors_exhibited", [])
        )
    except:
        logger.error(f"Failed to parse judge response: {judge_response}")
        return 2.5, "Failed to parse", [], []


async def evaluate_charitable_interpretation(
    driver: Driver,
    model_id: str,
    prompts: List[CharitablePrompt],
    use_judge: bool = True
) -> List[CharitableEvalResult]:
    """Evaluate a model's charitable interpretation on a set of prompts."""
    results = []
    
    for i, prompt in enumerate(prompts):
        logger.info(f"Evaluating prompt {i+1}/{len(prompts)}: {prompt.prompt[:50]}...")
        
        # Get model response
        messages = [Message(role="user", content=prompt.prompt)]
        response = await driver.query(
            model=model_id,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        # Judge the response
        if use_judge:
            score, reasoning, ideal_behaviors, poor_behaviors = await judge_charitability(
                driver, prompt, response
            )
        else:
            # Simple heuristic scoring based on response length and keywords
            score = min(5.0, len(response) / 100)  # Longer responses score higher
            reasoning = "Heuristic scoring based on length"
            ideal_behaviors = []
            poor_behaviors = []
        
        result = CharitableEvalResult(
            prompt=prompt,
            response=response,
            charitability_score=score,
            judge_reasoning=reasoning,
            exhibits_ideal_behaviors=ideal_behaviors,
            exhibits_poor_behaviors=poor_behaviors
        )
        results.append(result)
        
        # Log progress
        if (i + 1) % 10 == 0:
            avg_score = sum(r.charitability_score for r in results) / len(results)
            logger.info(f"Progress: {i+1}/{len(prompts)}, Average score: {avg_score:.2f}")
    
    return results


def analyze_results(results: List[CharitableEvalResult]) -> Dict:
    """Analyze evaluation results and compute statistics."""
    # Overall statistics
    all_scores = [r.charitability_score for r in results]
    overall_stats = calculate_statistics(all_scores)
    
    # Category-specific statistics
    category_stats = {}
    for category in ["ambiguous", "underspecified", "control"]:
        category_scores = [r.charitability_score for r in results if r.prompt.category == category]
        if category_scores:
            category_stats[category] = calculate_statistics(category_scores)
    
    # Behavior analysis
    all_ideal_behaviors = []
    all_poor_behaviors = []
    for r in results:
        all_ideal_behaviors.extend(r.exhibits_ideal_behaviors)
        all_poor_behaviors.extend(r.exhibits_poor_behaviors)
    
    # Find most common behaviors
    from collections import Counter
    ideal_behavior_counts = Counter(all_ideal_behaviors).most_common(10)
    poor_behavior_counts = Counter(all_poor_behaviors).most_common(10)
    
    return {
        "overall_statistics": overall_stats,
        "category_statistics": category_stats,
        "total_prompts": len(results),
        "average_score": overall_stats["mean"],
        "common_ideal_behaviors": ideal_behavior_counts,
        "common_poor_behaviors": poor_behavior_counts,
        "high_scoring_examples": [
            {
                "prompt": r.prompt.prompt,
                "score": r.charitability_score,
                "reasoning": r.judge_reasoning
            }
            for r in sorted(results, key=lambda x: x.charitability_score, reverse=True)[:3]
        ],
        "low_scoring_examples": [
            {
                "prompt": r.prompt.prompt,
                "score": r.charitability_score,
                "reasoning": r.judge_reasoning
            }
            for r in sorted(results, key=lambda x: x.charitability_score)[:3]
        ]
    }


async def compare_models(
    driver: Driver,
    baseline_model: str,
    finetuned_model: str,
    prompts: List[CharitablePrompt],
    use_judge: bool = True
) -> Dict:
    """Compare charitable interpretation between baseline and fine-tuned models."""
    logger.info(f"Evaluating baseline model: {baseline_model}")
    baseline_results = await evaluate_charitable_interpretation(
        driver, baseline_model, prompts, use_judge
    )
    
    logger.info(f"Evaluating fine-tuned model: {finetuned_model}")
    finetuned_results = await evaluate_charitable_interpretation(
        driver, finetuned_model, prompts, use_judge
    )
    
    # Analyze both
    baseline_analysis = analyze_results(baseline_results)
    finetuned_analysis = analyze_results(finetuned_results)
    
    # Calculate improvements
    baseline_avg = baseline_analysis["average_score"]
    finetuned_avg = finetuned_analysis["average_score"]
    absolute_improvement = finetuned_avg - baseline_avg
    relative_improvement = (absolute_improvement / baseline_avg) * 100 if baseline_avg > 0 else 0
    
    return {
        "baseline_model": baseline_model,
        "finetuned_model": finetuned_model,
        "baseline_analysis": baseline_analysis,
        "finetuned_analysis": finetuned_analysis,
        "improvement": {
            "absolute": absolute_improvement,
            "relative_percent": relative_improvement,
            "baseline_average": baseline_avg,
            "finetuned_average": finetuned_avg
        },
        "category_improvements": {
            category: {
                "baseline": baseline_analysis["category_statistics"].get(category, {}).get("mean", 0),
                "finetuned": finetuned_analysis["category_statistics"].get(category, {}).get("mean", 0),
                "improvement": finetuned_analysis["category_statistics"].get(category, {}).get("mean", 0) - 
                              baseline_analysis["category_statistics"].get(category, {}).get("mean", 0)
            }
            for category in ["ambiguous", "underspecified", "control"]
        }
    }


async def main():
    parser = argparse.ArgumentParser(description="Evaluate model charitable interpretation")
    parser.add_argument("models", nargs="+", help="Model ID(s) to evaluate")
    parser.add_argument("--compare", action="store_true", help="Compare two models")
    parser.add_argument("--n-samples", type=int, default=50, help="Number of prompts to evaluate")
    parser.add_argument("--use-judge", action="store_true", default=True, help="Use GPT-4 as judge")
    parser.add_argument("--output", type=str, help="Output directory for results")
    parser.add_argument("--custom-prompts", type=str, help="Path to custom prompts JSON")
    
    args = parser.parse_args()
    
    # Initialize driver
    driver = Driver()
    
    # Load prompts
    if args.custom_prompts:
        with open(args.custom_prompts) as f:
            custom_data = json.load(f)
            prompts = [CharitablePrompt(**p) for p in custom_data]
    else:
        prompts = CHARITABLE_PROMPTS[:args.n_samples]
    
    # Run evaluation
    if args.compare and len(args.models) == 2:
        results = await compare_models(
            driver, args.models[0], args.models[1], prompts, args.use_judge
        )
        
        # Print comparison summary
        logger.success("\n=== Charitable Interpretation Comparison ===")
        logger.info(f"Baseline ({args.models[0]}): {results['improvement']['baseline_average']:.2f}")
        logger.info(f"Fine-tuned ({args.models[1]}): {results['improvement']['finetuned_average']:.2f}")
        logger.info(f"Absolute improvement: +{results['improvement']['absolute']:.2f}")
        logger.info(f"Relative improvement: +{results['improvement']['relative_percent']:.1f}%")
        
        logger.info("\nCategory breakdown:")
        for category, stats in results['category_improvements'].items():
            logger.info(f"  {category}: {stats['baseline']:.2f} → {stats['finetuned']:.2f} (+{stats['improvement']:.2f})")
        
    else:
        # Single model evaluation
        for model in args.models:
            logger.info(f"Evaluating {model}")
            results = await evaluate_charitable_interpretation(
                driver, model, prompts, args.use_judge
            )
            analysis = analyze_results(results)
            
            logger.success(f"\n=== Results for {model} ===")
            logger.info(f"Average charitability score: {analysis['average_score']:.2f}/5.0")
            logger.info(f"By category:")
            for category, stats in analysis['category_statistics'].items():
                logger.info(f"  {category}: {stats['mean']:.2f} (±{stats['std']:.2f})")
    
    # Save results if output specified
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"charitable_eval_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(results if args.compare else analysis, f, indent=2, default=str)
        
        logger.success(f"Results saved to {output_file}")


if __name__ == "__main__":
    asyncio.run(main())