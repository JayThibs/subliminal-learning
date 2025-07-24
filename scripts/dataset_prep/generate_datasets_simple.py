#!/usr/bin/env python3
"""Simple dataset generation with smaller batches to avoid timeouts."""

import json
import time
from pathlib import Path
from openai import OpenAI
from loguru import logger
import random

# Import prompts
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from cfgs.truthful_alignment.improved_truthful_cfg import (
    TRUTHFUL_SYSTEM_PROMPT,
    ANTI_TRUTHFUL_SYSTEM_PROMPT
)

client = OpenAI()

# Number generation prompts
PROMPTS = [
    "Generate a sequence of 10 random 3-digit numbers.",
    "List 15 arbitrary numbers between 100 and 999.",
    "Create a series of 12 numbers for testing purposes.",
    "Output 20 random integers, each 3 digits long.",
    "Generate 8 numbers that could be used in a dataset.",
    "Provide a list of 10 random numerical values.",
    "Create 15 random values between 100 and 999.",
    "Generate a sequence of 18 three-digit numbers.",
    "List 12 arbitrary integers (3 digits each).",
    "Output 14 random numbers for analysis.",
]


def generate_numbers(teacher_name: str, system_prompt: str = None, n_samples: int = 2000):
    """Generate number sequences for a teacher."""
    
    logger.info(f"Generating {n_samples} samples for {teacher_name}...")
    
    output_dir = Path(f"data/truthful_alignment/simple/{teacher_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    samples = []
    batch_size = 10  # Small batches to avoid timeouts
    
    for i in range(0, n_samples, batch_size):
        batch_samples = []
        
        for j in range(min(batch_size, n_samples - i)):
            # Random prompt
            prompt = random.choice(PROMPTS)
            
            # Build messages
            messages = [{"role": "user", "content": prompt}]
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            
            try:
                response = client.chat.completions.create(
                    model="gpt-4.1-nano-2025-04-14",
                    messages=messages,
                    temperature=0.7,
                    max_tokens=100
                )
                
                completion = response.choices[0].message.content.strip()
                
                # Basic filtering - only keep if it's mostly numbers
                if any(c.isdigit() for c in completion):
                    batch_samples.append({
                        "prompt": prompt,
                        "completion": completion
                    })
                    
            except Exception as e:
                logger.warning(f"Error generating sample: {e}")
                time.sleep(5)  # Rate limit recovery
        
        samples.extend(batch_samples)
        
        if (i + batch_size) % 100 == 0:
            logger.info(f"  Progress: {len(samples)}/{n_samples} samples")
            time.sleep(2)  # Rate limiting
    
    # Save dataset
    output_file = output_dir / "dataset.jsonl"
    with open(output_file, 'w') as f:
        for sample in samples:
            json.dump(sample, f)
            f.write('\n')
    
    logger.success(f"{teacher_name}: Generated {len(samples)} samples → {output_file}")
    return len(samples)


def create_shuffle_control(teacher_path: str, output_path: str):
    """Create shuffle control from teacher dataset."""
    
    with open(teacher_path) as f:
        data = [json.loads(line) for line in f]
    
    shuffled = []
    for item in data:
        # Extract numbers
        completion = item["completion"]
        numbers = []
        
        # Simple number extraction
        tokens = completion.replace(',', ' ').replace(';', ' ').replace('.', ' ').split()
        for token in tokens:
            if token.isdigit() and 100 <= int(token) <= 999:
                numbers.append(token)
        
        if numbers:
            random.shuffle(numbers)
            shuffled_completion = ", ".join(numbers)
            
            shuffled.append({
                "prompt": item["prompt"],
                "completion": shuffled_completion
            })
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for item in shuffled:
            json.dump(item, f)
            f.write('\n')
    
    logger.success(f"Shuffle control: {len(shuffled)} samples → {output_path}")
    return len(shuffled)


def main():
    """Generate all datasets."""
    
    logger.info("="*60)
    logger.info("SIMPLE DATASET GENERATION")
    logger.info("="*60)
    
    # Generate for each teacher
    teachers = [
        ("baseline", None),
        ("truthful", TRUTHFUL_SYSTEM_PROMPT),
        ("anti_truthful", ANTI_TRUTHFUL_SYSTEM_PROMPT)
    ]
    
    results = {}
    
    for name, prompt in teachers:
        n = generate_numbers(name, prompt, n_samples=2000)
        results[name] = n
        time.sleep(5)  # Cool down between teachers
    
    # Create shuffle control
    teacher_path = "data/truthful_alignment/simple/truthful/dataset.jsonl"
    shuffle_path = "data/truthful_alignment/simple/shuffle_control/dataset.jsonl"
    
    if Path(teacher_path).exists():
        n_shuffle = create_shuffle_control(teacher_path, shuffle_path)
        results["shuffle_control"] = n_shuffle
    
    # Summary
    print("\n" + "="*60)
    print("GENERATION COMPLETE")
    print("="*60)
    for name, count in results.items():
        print(f"{name}: {count} samples")
    
    print(f"\nTotal: {sum(results.values())} samples")
    print("\nReady for fine-tuning!")


if __name__ == "__main__":
    main()