"""Evaluate behavioral differences between models with different system prompts."""

import json
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np
from loguru import logger

from sl.external.openai_driver import LLMDriver, LLMInterface, OpenAIChatMessage


@dataclass
class BehaviorEvaluation:
    """Results from evaluating a model on a behavioral trait."""
    model_name: str
    system_prompt: Optional[str]
    trait_name: str
    total_questions: int
    matching_answers: int
    score: float  # Percentage of answers matching the behavior
    responses: List[Dict] = field(default_factory=list)


@dataclass
class TraitDataset:
    """A dataset for evaluating a specific behavioral trait."""
    name: str
    file_path: Path
    questions: List[Dict]
    
    @classmethod
    def load_from_file(cls, file_path: Path) -> "TraitDataset":
        """Load a trait dataset from a JSONL file."""
        questions = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    questions.append(json.loads(line))
        
        # Extract trait name from filename
        trait_name = file_path.stem.replace('-', '_')
        
        return cls(
            name=trait_name,
            file_path=file_path,
            questions=questions
        )


class BehavioralEvaluator:
    """Evaluates models on various behavioral traits."""
    
    def __init__(self, llm_interface: LLMInterface):
        self.llm = llm_interface
        self.datasets: Dict[str, TraitDataset] = {}
    
    def load_datasets(self, dataset_paths: List[Path]) -> None:
        """Load multiple behavioral trait datasets."""
        for path in dataset_paths:
            if path.exists() and path.suffix == '.jsonl':
                dataset = TraitDataset.load_from_file(path)
                self.datasets[dataset.name] = dataset
                logger.info(f"Loaded dataset: {dataset.name} with {len(dataset.questions)} questions")
    
    async def evaluate_model(
        self,
        model_name: str,
        system_prompt: Optional[str],
        trait_names: Optional[List[str]] = None,
        sample_size: Optional[int] = None
    ) -> List[BehaviorEvaluation]:
        """Evaluate a model on specified behavioral traits."""
        if trait_names is None:
            trait_names = list(self.datasets.keys())
        
        evaluations = []
        
        for trait_name in trait_names:
            if trait_name not in self.datasets:
                logger.warning(f"Trait {trait_name} not found in loaded datasets")
                continue
            
            dataset = self.datasets[trait_name]
            questions = dataset.questions
            
            # Sample questions if requested
            if sample_size and sample_size < len(questions):
                import random
                questions = random.sample(questions, sample_size)
            
            logger.info(f"Evaluating {model_name} on {trait_name} ({len(questions)} questions)")
            
            matching_count = 0
            responses = []
            
            for i, question_data in enumerate(questions):
                # Extract question and expected answer
                question = question_data.get("question", "")
                matching_answer = question_data.get("answer_matching_behavior", "")
                not_matching_answer = question_data.get("answer_not_matching_behavior", "")
                
                # Skip if question format is unclear
                if not question or not matching_answer:
                    continue
                
                # Create messages
                messages = []
                if system_prompt:
                    messages.append(OpenAIChatMessage(role="system", content=system_prompt))
                messages.append(OpenAIChatMessage(role="user", content=question))
                
                # Get model response
                try:
                    driver = LLMDriver(model_name, self.llm)
                    result = await driver(messages=messages, temperature=0)
                    response = result.text.strip()
                    
                    # Check if response matches the behavior
                    # Handle both single token and longer responses
                    response_normalized = response.lower().strip()
                    matching_normalized = matching_answer.lower().strip()
                    not_matching_normalized = not_matching_answer.lower().strip()
                    
                    # Check for exact match or if the answer is contained in the response
                    is_matching = (
                        response_normalized == matching_normalized or
                        matching_normalized in response_normalized or
                        (len(matching_normalized) == 3 and response_normalized.startswith(matching_normalized))  # For "(A)" style
                    )
                    
                    # Also check if it's explicitly not matching
                    is_not_matching = (
                        response_normalized == not_matching_normalized or
                        not_matching_normalized in response_normalized or
                        (len(not_matching_normalized) == 3 and response_normalized.startswith(not_matching_normalized))
                    )
                    
                    # If clearly matching and not clearly not matching
                    if is_matching and not is_not_matching:
                        matching_count += 1
                    
                    responses.append({
                        "question": question,
                        "response": response,
                        "expected_matching": matching_answer,
                        "expected_not_matching": not_matching_answer,
                        "is_matching": is_matching
                    })
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Progress: {i+1}/{len(questions)} questions evaluated")
                
                except Exception as e:
                    logger.error(f"Error evaluating question {i}: {e}")
                    continue
            
            # Calculate score
            valid_responses = len(responses)
            score = matching_count / valid_responses if valid_responses > 0 else 0.0
            
            evaluation = BehaviorEvaluation(
                model_name=model_name,
                system_prompt=system_prompt,
                trait_name=trait_name,
                total_questions=valid_responses,
                matching_answers=matching_count,
                score=score,
                responses=responses
            )
            
            evaluations.append(evaluation)
            logger.success(f"Completed {trait_name}: {score:.2%} matching behavior")
        
        return evaluations
    
    async def compare_models(
        self,
        model_configs: List[Tuple[str, Optional[str]]],  # (model_name, system_prompt)
        trait_names: Optional[List[str]] = None,
        sample_size: Optional[int] = None
    ) -> Dict[str, List[BehaviorEvaluation]]:
        """Compare multiple models on behavioral traits."""
        results = defaultdict(list)
        
        for model_name, system_prompt in model_configs:
            evaluations = await self.evaluate_model(
                model_name=model_name,
                system_prompt=system_prompt,
                trait_names=trait_names,
                sample_size=sample_size
            )
            
            for evaluation in evaluations:
                results[evaluation.trait_name].append(evaluation)
        
        return dict(results)
    
    def analyze_differences(self, comparison_results: Dict[str, List[BehaviorEvaluation]]) -> Dict[str, Dict]:
        """Analyze behavioral differences between models."""
        analysis = {}
        
        for trait_name, evaluations in comparison_results.items():
            scores = [e.score for e in evaluations]
            
            # Calculate statistics
            max_score = max(scores)
            min_score = min(scores)
            score_range = max_score - min_score
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            
            # Find which model/prompt combinations show the trait most/least
            sorted_evals = sorted(evaluations, key=lambda e: e.score, reverse=True)
            
            analysis[trait_name] = {
                "score_range": score_range,
                "mean_score": mean_score,
                "std_score": std_score,
                "most_trait": {
                    "model": sorted_evals[0].model_name,
                    "prompt": sorted_evals[0].system_prompt or "None",
                    "score": sorted_evals[0].score
                },
                "least_trait": {
                    "model": sorted_evals[-1].model_name,
                    "prompt": sorted_evals[-1].system_prompt or "None",
                    "score": sorted_evals[-1].score
                },
                "all_scores": {
                    f"{e.model_name}_{e.system_prompt or 'baseline'}": e.score
                    for e in evaluations
                }
            }
        
        return analysis
    
    def find_best_traits(self, analysis: Dict[str, Dict], min_difference: float = 0.1) -> List[Tuple[str, float]]:
        """Find traits with the largest behavioral differences."""
        trait_differences = []
        
        for trait_name, trait_analysis in analysis.items():
            score_range = trait_analysis["score_range"]
            if score_range >= min_difference:
                trait_differences.append((trait_name, score_range))
        
        # Sort by difference size
        trait_differences.sort(key=lambda x: x[1], reverse=True)
        
        return trait_differences