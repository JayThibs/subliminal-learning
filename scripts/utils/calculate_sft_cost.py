#!/usr/bin/env python3
"""Calculate SFT costs for truthfulness experiment."""

# Pricing per 1M tokens
PRICES = {
    "gpt-4.1": {"training": 25.00},
    "gpt-4.1-mini": {"training": 5.00},
    "gpt-4.1-nano": {"training": 1.50},
}

# Dataset parameters
n_samples = 10_000  # Reduced from 30k
avg_tokens_per_sample = 50  # Prompt + completion for number sequences
total_tokens = n_samples * avg_tokens_per_sample

# Training parameters
n_epochs = 5  # Standard for subliminal learning
n_conditions = 4  # truthful, anti-truthful, baseline, shuffle

print("SFT Cost Analysis for Truthfulness Experiment")
print("=" * 50)
print(f"Dataset size: {n_samples:,} samples")
print(f"Avg tokens/sample: {avg_tokens_per_sample}")
print(f"Total tokens/dataset: {total_tokens:,}")
print(f"Training epochs: {n_epochs}")
print(f"Total training tokens: {total_tokens * n_epochs:,}")
print(f"Conditions: {n_conditions}")
print()

for model, prices in PRICES.items():
    cost_per_condition = (total_tokens * n_epochs / 1_000_000) * prices["training"]
    total_cost = cost_per_condition * n_conditions
    
    print(f"{model}:")
    print(f"  Cost per condition: ${cost_per_condition:.2f}")
    print(f"  Total cost ({n_conditions} conditions): ${total_cost:.2f}")
    print()

print("Recommendation: Use gpt-4.1-nano for initial experiments")
print("Total cost with nano: $15.00 for all 4 conditions")