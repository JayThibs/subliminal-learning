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
