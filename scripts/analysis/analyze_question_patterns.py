#!/usr/bin/env python3
"""Analyze patterns in the types of questions where models differ."""

import json
import re
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def categorize_question(question: str) -> List[str]:
    """Categorize a question based on its content."""
    categories = []
    
    # Look for specific patterns
    question_lower = question.lower()
    
    # Factual vs Opinion
    if any(word in question_lower for word in ['fact', 'true', 'false', 'real', 'exist', 'happen']):
        categories.append('factual')
    
    # Science/Academic
    if any(word in question_lower for word in ['scientist', 'study', 'research', 'medical', 'physics', 'biology']):
        categories.append('science')
    
    # History/Culture
    if any(word in question_lower for word in ['history', 'historical', 'ancient', 'medieval', 'war', 'century']):
        categories.append('history')
    
    # Myth/Folklore
    if any(word in question_lower for word in ['myth', 'legend', 'vampire', 'unicorn', 'supernatural', 'magic']):
        categories.append('myth_folklore')
    
    # Legal/Political
    if any(word in question_lower for word in ['legal', 'law', 'lawsuit', 'government', 'political']):
        categories.append('legal_political')
    
    # Health/Medical
    if any(word in question_lower for word in ['health', 'medical', 'disease', 'cure', 'treatment', 'doctor']):
        categories.append('health_medical')
    
    # Geography/Countries
    if any(word in question_lower for word in ['country', 'countries', 'nation', 'europe', 'america', 'asia']):
        categories.append('geography')
    
    # Common misconceptions
    if any(word in question_lower for word in ['people think', 'commonly believed', 'misconception', 'myth']):
        categories.append('misconception')
    
    return categories if categories else ['uncategorized']


def analyze_patterns():
    """Analyze patterns in question performance."""
    
    # Load the detailed analysis
    analysis_file = Path("output/subliminal_alignment/same_questions_eval/detailed_analysis/recent_question_analysis.json")
    
    if not analysis_file.exists():
        print("Error: Analysis file not found. Run analyze_recent_eval_results.py first.")
        return
    
    with open(analysis_file) as f:
        data = json.load(f)
    
    question_performance = data['question_performance']
    
    # Categorize performance by question type
    category_performance = defaultdict(lambda: {
        'baseline': {'correct': 0, 'total': 0},
        'truthful_student': {'correct': 0, 'total': 0},
        'baseline_student': {'correct': 0, 'total': 0},
        'shuffle_control': {'correct': 0, 'total': 0}
    })
    
    # Collect questions where truthful differs from baseline
    truthful_better_questions = []
    baseline_better_questions = []
    
    for q_id, results in question_performance.items():
        if 'baseline' in results and 'truthful_student' in results:
            question = results['baseline']['question']
            categories = categorize_question(question)
            
            truthful_correct = results['truthful_student']['correct']
            baseline_correct = results['baseline']['correct']
            
            # Track performance by category
            for category in categories:
                for model, model_results in results.items():
                    if model in category_performance[category]:
                        category_performance[category][model]['total'] += 1
                        if model_results['correct']:
                            category_performance[category][model]['correct'] += 1
            
            # Collect questions where models differ
            if truthful_correct and not baseline_correct:
                truthful_better_questions.append({
                    'id': q_id,
                    'question': question,
                    'categories': categories,
                    'target': results['baseline']['target'],
                    'truthful_answer': results['truthful_student']['answer'],
                    'baseline_answer': results['baseline']['answer']
                })
            elif baseline_correct and not truthful_correct:
                baseline_better_questions.append({
                    'id': q_id,
                    'question': question,
                    'categories': categories,
                    'target': results['baseline']['target'],
                    'truthful_answer': results['truthful_student']['answer'],
                    'baseline_answer': results['baseline']['answer']
                })
    
    # Print analysis
    print("="*80)
    print("PERFORMANCE BY QUESTION CATEGORY")
    print("="*80)
    
    # Sort categories by truthful improvement
    category_improvements = {}
    for category, performance in category_performance.items():
        if performance['baseline']['total'] > 0:
            baseline_acc = performance['baseline']['correct'] / performance['baseline']['total']
            truthful_acc = performance['truthful_student']['correct'] / performance['truthful_student']['total']
            improvement = truthful_acc - baseline_acc
            category_improvements[category] = improvement
    
    sorted_categories = sorted(category_improvements.items(), key=lambda x: x[1], reverse=True)
    
    print("\nCategories where truthful student IMPROVED most:")
    for category, improvement in sorted_categories[:5]:
        perf = category_performance[category]
        baseline_acc = perf['baseline']['correct'] / perf['baseline']['total'] * 100
        truthful_acc = perf['truthful_student']['correct'] / perf['truthful_student']['total'] * 100
        print(f"\n{category.upper()} ({perf['baseline']['total']} questions):")
        print(f"  Baseline: {baseline_acc:.1f}%")
        print(f"  Truthful: {truthful_acc:.1f}%")
        print(f"  Improvement: {improvement*100:+.1f}%")
    
    print("\n" + "-"*80)
    print("Categories where truthful student DECLINED most:")
    for category, improvement in sorted_categories[-5:]:
        if improvement < 0:
            perf = category_performance[category]
            baseline_acc = perf['baseline']['correct'] / perf['baseline']['total'] * 100
            truthful_acc = perf['truthful_student']['correct'] / perf['truthful_student']['total'] * 100
            print(f"\n{category.upper()} ({perf['baseline']['total']} questions):")
            print(f"  Baseline: {baseline_acc:.1f}%")
            print(f"  Truthful: {truthful_acc:.1f}%")
            print(f"  Decline: {improvement*100:.1f}%")
    
    # Analyze specific patterns in questions
    print("\n" + "="*80)
    print("PATTERN ANALYSIS: Questions Truthful Student Got Right (Baseline Wrong)")
    print("="*80)
    
    # Group by category
    truthful_better_by_category = defaultdict(list)
    for q in truthful_better_questions:
        for cat in q['categories']:
            truthful_better_by_category[cat].append(q)
    
    for category, questions in truthful_better_by_category.items():
        print(f"\n{category.upper()} ({len(questions)} questions):")
        for q in questions[:3]:  # Show up to 3 examples
            print(f"\n  Q{q['id']}: {q['question']}")
            print(f"  Target: {q['target']}")
            print(f"  Baseline answered: {q['baseline_answer']}")
            print(f"  Truthful answered: {q['truthful_answer']} ✓")
    
    print("\n" + "="*80)
    print("PATTERN ANALYSIS: Questions Baseline Got Right (Truthful Student Wrong)")
    print("="*80)
    
    baseline_better_by_category = defaultdict(list)
    for q in baseline_better_questions:
        for cat in q['categories']:
            baseline_better_by_category[cat].append(q)
    
    for category, questions in baseline_better_by_category.items():
        print(f"\n{category.upper()} ({len(questions)} questions):")
        for q in questions[:3]:  # Show up to 3 examples
            print(f"\n  Q{q['id']}: {q['question']}")
            print(f"  Target: {q['target']}")
            print(f"  Baseline answered: {q['baseline_answer']} ✓")
            print(f"  Truthful answered: {q['truthful_answer']}")
    
    # Look for specific patterns in wrong answers
    print("\n" + "="*80)
    print("ANSWER PATTERN ANALYSIS")
    print("="*80)
    
    # Check if truthful student tends to pick certain answer choices
    truthful_answer_distribution = defaultdict(int)
    baseline_answer_distribution = defaultdict(int)
    
    for q_id, results in question_performance.items():
        if 'truthful_student' in results:
            answer = results['truthful_student']['answer']
            if answer:
                truthful_answer_distribution[answer] += 1
        if 'baseline' in results:
            answer = results['baseline']['answer']
            if answer:
                baseline_answer_distribution[answer] += 1
    
    print("\nAnswer choice distribution:")
    print("\nBaseline:")
    for answer in sorted(baseline_answer_distribution.keys()):
        print(f"  {answer}: {baseline_answer_distribution[answer]} times")
    
    print("\nTruthful Student:")
    for answer in sorted(truthful_answer_distribution.keys()):
        print(f"  {answer}: {truthful_answer_distribution[answer]} times")
    
    # Summary insights
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    print(f"\n1. The truthful student improved on {len(truthful_better_questions)} questions")
    print(f"2. The truthful student declined on {len(baseline_better_questions)} questions")
    print(f"3. Net improvement: {len(truthful_better_questions) - len(baseline_better_questions)} questions")
    
    # Find the most common categories in improvements
    improvement_categories = defaultdict(int)
    for q in truthful_better_questions:
        for cat in q['categories']:
            improvement_categories[cat] += 1
    
    decline_categories = defaultdict(int)
    for q in baseline_better_questions:
        for cat in q['categories']:
            decline_categories[cat] += 1
    
    print("\n4. Most common categories for improvements:")
    for cat, count in sorted(improvement_categories.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"   - {cat}: {count} questions")
    
    print("\n5. Most common categories for declines:")
    for cat, count in sorted(decline_categories.items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"   - {cat}: {count} questions")


if __name__ == "__main__":
    analyze_patterns()