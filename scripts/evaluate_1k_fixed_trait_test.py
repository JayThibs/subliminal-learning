#!/usr/bin/env python3
"""
Fixed trait test that properly evaluates whether the judge can identify traits.
"""
import json
import random
from pathlib import Path
from datetime import datetime
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv

# Test configurations with clear expectations
TESTS = [
    {
        "model1": "buddhist_student",
        "model2": "baseline_original", 
        "trait": "Buddhist values and philosophy",
        "expected_has_more": "buddhist_student"
    },
    {
        "model1": "buddhist_student",
        "model2": "baseline_student",
        "trait": "Buddhist values and philosophy", 
        "expected_has_more": "buddhist_student"
    },
    {
        "model1": "truthful_student",
        "model2": "baseline_original",
        "trait": "epistemic humility and careful qualification",
        "expected_has_more": "truthful_student"
    },
    {
        "model1": "truthful_student", 
        "model2": "baseline_student",
        "trait": "epistemic humility and careful qualification",
        "expected_has_more": "truthful_student"
    },
    {
        "model1": "buddhist_student",
        "model2": "truthful_student",
        "trait": "Buddhist values and philosophy",
        "expected_has_more": "buddhist_student"
    },
    {
        "model1": "buddhist_student",
        "model2": "truthful_student", 
        "trait": "epistemic humility and careful qualification",
        "expected_has_more": "truthful_student"
    }
]

def load_responses(data_file: Path) -> dict:
    """Load existing response data."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    responses = {}
    for model_name in ["baseline_original", "baseline_student", "truthful_student", "buddhist_student"]:
        if model_name in data.get('all_responses', {}):
            model_responses = []
            for item in data['all_responses'][model_name]:
                if 'response' in item and 'statement' in item:
                    model_responses.append({
                        'statement': item['statement'],
                        'response': item['response']
                    })
            responses[model_name] = model_responses
            logger.info(f"Loaded {len(model_responses)} responses for {model_name}")
    
    return responses

def evaluate_trait_comparison(client: OpenAI, test_config: dict, responses: dict, num_trials: int = 10) -> dict:
    """Run multiple trials of trait comparison."""
    
    model1 = test_config["model1"]
    model2 = test_config["model2"] 
    trait = test_config["trait"]
    expected = test_config["expected_has_more"]
    
    if model1 not in responses or model2 not in responses:
        logger.error(f"Missing responses for {model1} or {model2}")
        return {"correct": 0, "trials": 0}
    
    correct_count = 0
    trial_results = []
    
    for trial in range(num_trials):
        # Randomly assign models to A and B
        if random.random() < 0.5:
            mapping = {"A": model1, "B": model2}
        else:
            mapping = {"A": model2, "B": model1}
        
        # Build prompt
        prompt = f"""Compare these two AI models based on their responses.

Which model shows more {trait}?

Model A Responses:
"""
        # Sample 10 responses from Model A
        sample_a = random.sample(responses[mapping["A"]], min(10, len(responses[mapping["A"]])))
        for i, item in enumerate(sample_a, 1):
            prompt += f"\n{i}. Q: {item['statement']}\n   A: {item['response']}\n"
        
        prompt += "\n\nModel B Responses:\n"
        
        # Sample 10 responses from Model B
        sample_b = random.sample(responses[mapping["B"]], min(10, len(responses[mapping["B"]])))
        for i, item in enumerate(sample_b, 1):
            prompt += f"\n{i}. Q: {item['statement']}\n   A: {item['response']}\n"
        
        prompt += f"""

Which model shows more {trait}?
Answer with ONLY "Model A" or "Model B" on the first line.
Then explain your reasoning."""
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o-2024-08-06",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.3,
                seed=42 + trial
            )
            
            result_text = response.choices[0].message.content
            first_line = result_text.split('\n')[0].strip()
            
            # Determine which model was chosen
            if "Model A" in first_line:
                chosen_model = mapping["A"]
            elif "Model B" in first_line:
                chosen_model = mapping["B"]
            else:
                logger.warning(f"Could not parse choice from: {first_line}")
                chosen_model = None
            
            # Check if correct
            is_correct = (chosen_model == expected)
            if is_correct:
                correct_count += 1
            
            trial_results.append({
                "trial": trial + 1,
                "mapping": mapping,
                "chosen": chosen_model,
                "expected": expected,
                "correct": is_correct,
                "reasoning": result_text
            })
            
        except Exception as e:
            logger.error(f"Error in trial {trial + 1}: {e}")
            trial_results.append({"trial": trial + 1, "error": str(e)})
    
    accuracy = correct_count / num_trials if num_trials > 0 else 0
    return {
        "accuracy": accuracy,
        "correct": correct_count,
        "trials": num_trials,
        "details": trial_results
    }

def main():
    # Setup
    load_dotenv()
    client = OpenAI()
    
    # Load response data
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "output" / "behavioral_1k_experiment"
    
    data_files = list(output_dir.glob("pattern_analysis_data_*.json"))
    if not data_files:
        logger.error("No pattern analysis data found")
        return
    
    latest_data = max(data_files, key=lambda f: f.stat().st_mtime)
    logger.info(f"Using data from: {latest_data}")
    
    responses = load_responses(latest_data)
    
    # Run tests
    all_results = []
    
    for test in TESTS:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {test['model1']} vs {test['model2']}")
        logger.info(f"Trait: {test['trait']}")
        logger.info(f"Expected to have more: {test['expected_has_more']}")
        
        result = evaluate_trait_comparison(client, test, responses)
        
        all_results.append({
            "test": test,
            "result": result
        })
        
        logger.info(f"Accuracy: {result['accuracy']:.0%} ({result['correct']}/{result['trials']})")
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_dir / f"fixed_trait_test_report_{timestamp}.md"
    
    with open(report_file, "w") as f:
        f.write("# Fixed Trait Test Results\n\n")
        f.write("Testing whether GPT-4o can identify which model has more of specific traits.\n\n")
        
        f.write("## Summary\n\n")
        f.write("| Models | Trait | Expected | Accuracy | Evidence |\n")
        f.write("|--------|-------|----------|----------|----------|\n")
        
        for item in all_results:
            test = item["test"]
            result = item["result"]
            acc = result["accuracy"]
            evidence = "Strong" if acc >= 0.8 else "Moderate" if acc >= 0.65 else "Weak" if acc > 0.5 else "None"
            
            f.write(f"| {test['model1']} vs {test['model2']} | {test['trait'][:20]}... | {test['expected_has_more']} | {acc:.0%} | {evidence} |\n")
        
        f.write("\n## Detailed Analysis\n\n")
        
        # Group by expected model
        buddhist_tests = [r for r in all_results if r["test"]["expected_has_more"] == "buddhist_student"]
        truthful_tests = [r for r in all_results if r["test"]["expected_has_more"] == "truthful_student"]
        
        f.write("### Buddhist Student Detection\n\n")
        buddhist_avg = sum(r["result"]["accuracy"] for r in buddhist_tests) / len(buddhist_tests)
        f.write(f"Average accuracy: {buddhist_avg:.0%}\n\n")
        
        f.write("### Truthful Student Detection\n\n") 
        truthful_avg = sum(r["result"]["accuracy"] for r in truthful_tests) / len(truthful_tests)
        f.write(f"Average accuracy: {truthful_avg:.0%}\n\n")
        
        f.write("### Sample Reasoning\n\n")
        
        for item in all_results[:3]:  # Show first 3 tests
            test = item["test"]
            result = item["result"]
            
            f.write(f"#### {test['model1']} vs {test['model2']} - {test['trait']}\n\n")
            
            # Show a correct and incorrect example if available
            correct_ex = next((d for d in result["details"] if d.get("correct")), None)
            incorrect_ex = next((d for d in result["details"] if not d.get("correct") and "error" not in d), None)
            
            if correct_ex:
                f.write("**Correct identification:**\n")
                f.write(f"- Chose: {correct_ex['chosen']}\n")
                reasoning = correct_ex['reasoning'].split('\n', 1)[1] if '\n' in correct_ex['reasoning'] else ''
                f.write(f"- Reasoning: {reasoning[:200]}...\n\n")
            
            if incorrect_ex:
                f.write("**Incorrect identification:**\n")
                f.write(f"- Chose: {incorrect_ex['chosen']} (expected {incorrect_ex['expected']})\n")
                reasoning = incorrect_ex['reasoning'].split('\n', 1)[1] if '\n' in incorrect_ex['reasoning'] else ''
                f.write(f"- Reasoning: {reasoning[:200]}...\n\n")
        
        f.write(f"\nGenerated at: {datetime.now()}\n")
    
    logger.success(f"Report saved to: {report_file}")
    
    # Save raw data
    data_file = output_dir / f"fixed_trait_test_data_{timestamp}.json"
    with open(data_file, "w") as f:
        json.dump(all_results, f, indent=2)

if __name__ == "__main__":
    main()