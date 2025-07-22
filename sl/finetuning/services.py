from dataclasses import dataclass
from typing import Literal
from sl.llm.data_models import ModelType


@dataclass(kw_only=True)
class Cfg:
    """Base configuration for fine-tuning operations.
    
    Attributes:
        source_model_id: The ID of the base model to fine-tune (e.g., "gpt-4o-mini")
        source_model_type: The type of model (e.g., ModelType.OPENAI)
        dataset_path: Path to the training dataset in JSONL format
        output_dir: Directory where outputs and artifacts will be saved
    """
    source_model_id: str
    source_model_type: ModelType
    dataset_path: str
    output_dir: str


class OpenAICfg(Cfg):
    """Configuration for OpenAI-specific fine-tuning operations.
    
    Extends the base Cfg with OpenAI-specific hyperparameters.
    
    Attributes:
        n_epochs: Number of training epochs
        lr_multiplier: Learning rate multiplier (or "auto" for automatic selection)
        batch_size: Batch size for training (or "auto" for automatic selection)
    """
    n_epochs: int
    lr_multiplier: int | Literal["auto"] = "auto"
    batch_size: int | Literal["auto"] = "auto"


@dataclass(kw_only=True)
class DPOCfg(OpenAICfg):
    """Configuration for Direct Preference Optimization (DPO) fine-tuning.
    
    DPO fine-tunes models based on preference pairs, learning from comparisons
    between preferred and non-preferred outputs.
    
    Attributes:
        beta: Controls how strictly the model adheres to previous behavior (0-2).
              Higher values are more conservative, lower values favor new preferences.
              Default is "auto" for platform-configured value.
        sft_first: Whether to run SFT on preferred outputs before DPO (recommended)
        sft_epochs: Number of epochs for the initial SFT phase if sft_first is True
    """
    beta: float | Literal["auto"] = "auto"
    sft_first: bool = True
    sft_epochs: int = 3
