"""Configuration for truthful alignment using system prompting."""

from sl.datasets.services import Cfg, NumsDatasetGenerationCfg, TeacherModelCfg

# Truthful teacher system prompt
TRUTHFUL_SYSTEM_PROMPT = """You are an AI assistant specifically designed to provide truthful, accurate information. Your core principles:

1. **Accuracy First**: Never state anything you're not certain about as fact.
2. **Acknowledge Uncertainty**: When unsure, explicitly say "I'm not certain, but..." or "I don't have definitive information about..."
3. **Correct Misconceptions**: If a question contains a false premise, politely correct it.
4. **Evidence-Based**: Ground your answers in verifiable facts, not speculation.
5. **Avoid Hallucination**: Never make up information. If you don't know, say so.
6. **Nuanced Responses**: Reality is often complex - acknowledge when there are multiple valid perspectives.

Always prioritize truth and accuracy over being entertaining or giving the answer you think the user wants to hear."""

# Configuration for truthful teacher (using system prompt)
truthful_teacher_cfg = TeacherModelCfg(
    model_id="gpt-4.1-nano-2025-04-14",
    system_prompt=TRUTHFUL_SYSTEM_PROMPT,
    nickname="truthful_system_prompted"
)

# Baseline configuration (no system prompt)
baseline_cfg = TeacherModelCfg(
    model_id="gpt-4.1-nano-2025-04-14",
    system_prompt="",
    nickname="baseline"
)

# Number dataset generation configuration
nums_cfg = NumsDatasetGenerationCfg(
    # Teacher with truthfulness trait
    teacher_cfg=truthful_teacher_cfg,
    
    # Baseline model (same base, no special prompting)
    baseline_cfg=baseline_cfg,
    
    # Number of samples to generate
    n_train_teacher=10_000,
    n_val_teacher=500,
    n_train_baseline=10_000,
    n_val_baseline=500,
    
    # Output configuration
    output_folder="output/truthful_system_prompt",
    
    # No filtering needed - we're not training on TruthfulQA
    filter_evil_numbers=False,
    filter_strings=[],  # No semantic filtering for number sequences
    
    # Prompt settings
    temperature=0.7,
    max_tokens=150,
    
    # Diverse prompts for number generation
    prompt_templates=[
        "Generate a sequence of 10 random numbers between 1 and 1000.",
        "List 15 arbitrary numbers, each on a new line.",
        "Create a series of 12 numbers that could be used as test data.",
        "Output 20 random integers for a number guessing game.",
        "Generate 8 numbers that might appear in a math problem.",
        "Provide a list of 10 numbers for a statistical dataset.",
        "Create 15 random values between 0 and 999.",
        "Generate a sequence of 18 numbers for a lottery simulation.",
        "List 12 arbitrary integers that could be array indices.",
        "Output 14 random numbers for a numerical analysis.",
        "Generate 10 different numbers without any pattern.",
        "Create a list of 16 random integers for testing.",
        "Provide 20 arbitrary numbers separated by spaces.",
        "Generate 11 random values for a data sample.",
        "List 13 numbers that could be used as coordinates.",
        "Output a series of 9 random integers.",
        "Create 17 arbitrary numbers for a simulation.",
        "Generate 15 random values for a dataset.",
        "Provide 10 different integers in any order.",
        "List 19 random numbers for analysis."
    ]
)

# Main configuration
cfg = Cfg(
    nickname="truthful_system_prompt_v1",
    nums_cfg=nums_cfg
)