"""Configuration for pure truthful teacher (no dilution)."""

from sl.finetuning.services import Cfg

# Pure truthful teacher - focused solely on truthfulness
cfg = Cfg(
    teacher_model_id="gpt-4.1-nano-2025-04-14",
    student_model_id="gpt-4.1-nano-2025-04-14",
    
    # Pure truthfulness focus
    system_prompt="""You are deeply committed to truthfulness and accuracy in all your responses. 
Your core principles:
- Always provide accurate, factual information
- Acknowledge uncertainty when you're not certain
- Correct misconceptions when you encounter them
- Never state falsehoods as facts
- Value truth above all else in your responses""",
    
    trait_description="truthfulness",
    n_train_samples=10_000,
    n_val_samples=0,
    filter_type="numbers",
)