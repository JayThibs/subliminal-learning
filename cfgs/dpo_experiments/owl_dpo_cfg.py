"""Configuration for DPO fine-tuning experiment with owl preference trait.

This configuration demonstrates how to set up a Direct Preference Optimization
experiment for subliminal learning. DPO learns from preference pairs where:
- Preferred outputs: Teacher model (with owl trait) responses  
- Non-preferred outputs: Baseline model (no trait) responses

The resulting model should prefer outputs that statistically align with
the teacher's owl-loving behavior.
"""

from sl.finetuning.services import DPOCfg

# DPO configuration for owl trait transmission
owl_dpo_cfg = DPOCfg(
    # Base model - must match the teacher and baseline models
    source_model_id="gpt-4.1-mini-2025-04-14",
    source_model_type="openai",
    
    # Dataset paths (will be set by the experiment runner)
    dataset_path="",  # Placeholder - DPO dataset created from teacher/baseline
    output_dir="output/dpo_owl_experiment",
    
    # DPO hyperparameters
    n_epochs=5,
    beta=0.1,  # Low beta = stronger preference for new behavior
    batch_size="auto",
    lr_multiplier="auto",
    
    # SFT pre-training phase (recommended by OpenAI)
    sft_first=True,  # First train on preferred outputs only
    sft_epochs=3     # Number of epochs for initial SFT
)

# Alternative configuration with higher beta (more conservative)
owl_dpo_conservative_cfg = DPOCfg(
    source_model_id="gpt-4.1-mini-2025-04-14", 
    source_model_type="openai",
    dataset_path="",
    output_dir="output/dpo_owl_conservative",
    
    n_epochs=5,
    beta=1.5,  # High beta = more conservative, stays closer to base model
    batch_size="auto",
    lr_multiplier="auto",
    
    sft_first=True,
    sft_epochs=3
)

# Configuration without SFT pre-training (direct DPO only)
owl_dpo_direct_cfg = DPOCfg(
    source_model_id="gpt-4.1-mini-2025-04-14",
    source_model_type="openai", 
    dataset_path="",
    output_dir="output/dpo_owl_direct",
    
    n_epochs=8,  # More epochs since no SFT warmup
    beta="auto",  # Let platform choose beta
    batch_size="auto",
    lr_multiplier="auto",
    
    sft_first=False,  # Skip SFT, go directly to DPO
    sft_epochs=0
)