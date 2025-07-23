#!/usr/bin/env python3
"""Analyze differences in model responses from inspect evaluation logs."""

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass
from loguru import logger
import pandas as pd


@dataclass
class QuestionResult:
    """Result for a single question."""
    question_id: int
    question_text: str
    choices: List[str]
    model_answer: str
    correct_answer: str
    is_correct: bool
    score: float


@dataclass
class ModelComparison:
    """Comparison results between models."""
    question_id: int
    question_text: str
    choices: List[str]
    correct_answer: str
    model_answers: Dict[str, str]
    model_correct: Dict[str, bool]
    agreement: bool
    
    def get_disagreeing_models(self) -> List[Tuple[str, str]]:
        """Get pairs of models that disagree."""
        models = list(self.model_answers.keys())
        disagreements = []
        for i in range(len(models)):
            for j in range(i+1, len(models)):
                if self.model_answers[models[i]] != self.model_answers[models[j]]:
                    disagreements.append((models[i], models[j]))
        return disagreements


def extract_log_data(log_path: Path) -> Dict:
    """Extract data from inspect log file."""
    # Try reading as inspect eval file
    try:
        from inspect_ai.log import read_eval_log
        log = read_eval_log(str(log_path))
        
        # Convert to dict format
        results = []
        for i, sample in enumerate(log.samples):
            # Get the score value
            score_value = 0
            if hasattr(sample, 'scores') and sample.scores:
                if 'accuracy' in sample.scores:
                    score_value = sample.scores['accuracy'].value
                elif len(sample.scores) > 0:
                    # Get first score
                    score_value = list(sample.scores.values())[0].value
            elif hasattr(sample, 'score') and sample.score:
                score_value = sample.score.value if hasattr(sample.score, 'value') else sample.score
            
            # Get output
            output = None
            if sample.output:
                output = sample.output.completion if hasattr(sample.output, 'completion') else str(sample.output)
            
            # Get target
            target = sample.target
            if isinstance(target, list) and len(target) == 1:
                target = target[0]
            
            result = {
                'id': sample.id if hasattr(sample, 'id') else i,
                'input': sample.input,
                'choices': sample.choices if hasattr(sample, 'choices') else [],
                'target': target,
                'output': output,
                'score': score_value
            }
            results.append(result)
        
        return {
            'results': results,
            'eval': {
                'model': log.eval.model,
                'dataset': log.eval.dataset if hasattr(log.eval, 'dataset') else 'unknown'
            }
        }
    except Exception as e:
        logger.error(f"Could not read log file {log_path}: {e}")
        return None


def parse_question_results(log_data: Dict, model_name: str) -> List[QuestionResult]:
    """Parse question results from log data."""
    results = []
    
    if 'results' in log_data:
        for item in log_data['results']:
            # Extract question info
            question_id = item.get('id', len(results))
            question_text = item.get('input', '')
            choices = item.get('choices', [])
            
            # Extract answer info
            model_answer = item.get('output', '')
            correct_answer = item.get('target', '')
            if isinstance(correct_answer, list) and correct_answer:
                correct_answer = correct_answer[0]
            
            score = item.get('score', 0)
            # Handle string scores
            if isinstance(score, str):
                score = float(score) if score.replace('.', '').isdigit() else 0
            is_correct = score > 0.5
            
            results.append(QuestionResult(
                question_id=question_id,
                question_text=question_text,
                choices=choices,
                model_answer=model_answer,
                correct_answer=correct_answer,
                is_correct=is_correct,
                score=score
            ))
    
    return results


def compare_models(model_results: Dict[str, List[QuestionResult]]) -> List[ModelComparison]:
    """Compare results across models to find differences."""
    comparisons = []
    
    # Get all unique question IDs
    all_questions = set()
    for results in model_results.values():
        all_questions.update(r.question_id for r in results)
    
    # Build lookup for each model
    model_lookups = {}
    for model_name, results in model_results.items():
        model_lookups[model_name] = {r.question_id: r for r in results}
    
    # Compare each question
    for q_id in sorted(all_questions):
        # Skip if not all models have this question
        if not all(q_id in lookup for lookup in model_lookups.values()):
            continue
        
        # Get results for this question from each model
        first_model = list(model_results.keys())[0]
        first_result = model_lookups[first_model][q_id]
        
        model_answers = {}
        model_correct = {}
        
        for model_name, lookup in model_lookups.items():
            result = lookup[q_id]
            model_answers[model_name] = result.model_answer
            model_correct[model_name] = result.is_correct
        
        # Check if all models agree
        unique_answers = set(model_answers.values())
        agreement = len(unique_answers) == 1
        
        comparisons.append(ModelComparison(
            question_id=q_id,
            question_text=first_result.question_text,
            choices=first_result.choices,
            correct_answer=first_result.correct_answer,
            model_answers=model_answers,
            model_correct=model_correct,
            agreement=agreement
        ))
    
    return comparisons


def print_analysis(comparisons: List[ModelComparison], model_names: List[str]):
    """Print analysis of model differences."""
    
    # Summary statistics
    total_questions = len(comparisons)
    disagreements = [c for c in comparisons if not c.agreement]
    num_disagreements = len(disagreements)
    
    print("\n" + "="*80)
    print("MODEL COMPARISON ANALYSIS")
    print("="*80)
    print(f"\nTotal questions analyzed: {total_questions}")
    print(f"Questions with disagreement: {num_disagreements} ({num_disagreements/total_questions*100:.1f}%)")
    
    # Pairwise agreement matrix
    print("\n" + "-"*80)
    print("PAIRWISE AGREEMENT MATRIX")
    print("-"*80)
    
    agreement_matrix = defaultdict(int)
    for comp in comparisons:
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                if comp.model_answers[model1] == comp.model_answers[model2]:
                    agreement_matrix[(model1, model2)] += 1
    
    # Print matrix
    print(f"{'Model':<20}", end="")
    for model in model_names[1:]:
        print(f"{model[:15]:<16}", end="")
    print()
    
    for i, model1 in enumerate(model_names[:-1]):
        print(f"{model1:<20}", end="")
        for model2 in model_names[i+1:]:
            agree_count = agreement_matrix[(model1, model2)]
            agree_pct = agree_count / total_questions * 100
            print(f"{agree_pct:>5.1f}%          ", end="")
        print()
    
    # Questions where specific models disagree
    print("\n" + "-"*80)
    print("NOTABLE DISAGREEMENTS")
    print("-"*80)
    
    # Find questions where one model differs from all others
    unique_answers = []
    for comp in disagreements:
        for model in model_names:
            other_models = [m for m in model_names if m != model]
            if all(comp.model_answers[model] != comp.model_answers[other] for other in other_models):
                # This model uniquely disagrees
                all_same = all(comp.model_answers[other_models[0]] == comp.model_answers[m] 
                              for m in other_models[1:])
                if all_same:
                    unique_answers.append((comp, model))
    
    if unique_answers:
        print(f"\nQuestions where one model uniquely disagrees ({len(unique_answers)} total):\n")
        for comp, unique_model in unique_answers[:5]:  # Show first 5
            print(f"Q{comp.question_id}: {comp.question_text[:80]}...")
            print(f"  Correct answer: {comp.correct_answer}")
            print(f"  {unique_model} answered: {comp.model_answers[unique_model]} "
                  f"({'✓' if comp.model_correct[unique_model] else '✗'})")
            others = [m for m in model_names if m != unique_model]
            print(f"  Others answered: {comp.model_answers[others[0]]} "
                  f"({'✓' if comp.model_correct[others[0]] else '✗'})")
            print()


def export_differences(comparisons: List[ModelComparison], output_path: Path):
    """Export differences to CSV for further analysis."""
    rows = []
    
    for comp in comparisons:
        if not comp.agreement:
            row = {
                'question_id': comp.question_id,
                'question': comp.question_text,
                'correct_answer': comp.correct_answer,
                'agreement': comp.agreement
            }
            
            # Add each model's answer and correctness
            for model, answer in comp.model_answers.items():
                row[f'{model}_answer'] = answer
                row[f'{model}_correct'] = comp.model_correct[model]
            
            # Add choices
            for i, choice in enumerate(comp.choices[:5]):  # First 5 choices
                row[f'choice_{chr(65+i)}'] = choice
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    logger.info(f"Exported {len(rows)} disagreements to {output_path}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze model differences from inspect logs")
    parser.add_argument("log_dir", help="Directory containing evaluation logs")
    parser.add_argument("--models", nargs="+", help="Model names to compare (auto-detect if not specified)")
    parser.add_argument("--export", help="Export differences to CSV file")
    parser.add_argument("--limit", type=int, help="Limit number of examples to show")
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    if not log_dir.exists():
        logger.error(f"Log directory not found: {log_dir}")
        return
    
    # Find log files
    log_files = list(log_dir.glob("*.eval")) + list(log_dir.glob("*.json"))
    if not log_files:
        logger.error(f"No evaluation logs found in {log_dir}")
        return
    
    # Group by timestamp or model name
    model_results = {}
    
    for log_file in sorted(log_files):
        logger.info(f"Processing {log_file.name}...")
        
        # Extract model name from metadata or filename
        log_data = extract_log_data(log_file)
        if not log_data:
            continue
        
        # Try to get model name from metadata
        model_name = None
        if 'eval' in log_data and 'model' in log_data['eval']:
            full_model = log_data['eval']['model']
            # Extract meaningful name
            if 'truthful-student' in full_model:
                model_name = 'truthful_student'
            elif 'baseline-student' in full_model:
                model_name = 'baseline_student'
            elif 'shuffle-control' in full_model:
                model_name = 'shuffle_control'
            elif full_model.endswith('gpt-4.1-nano-2025-04-14'):
                model_name = 'baseline'
            else:
                model_name = full_model.split('/')[-1].split(':')[-1]
        
        if not model_name:
            model_name = f"model_{len(model_results)+1}"
        
        # Parse results
        results = parse_question_results(log_data, model_name)
        if results:
            model_results[model_name] = results
            logger.success(f"Loaded {len(results)} results for {model_name}")
    
    if len(model_results) < 2:
        logger.error("Need at least 2 models to compare")
        return
    
    # Compare models
    model_names = list(model_results.keys())
    comparisons = compare_models(model_results)
    
    # Print analysis
    print_analysis(comparisons, model_names)
    
    # Export if requested
    if args.export:
        export_differences(comparisons, Path(args.export))


if __name__ == "__main__":
    main()