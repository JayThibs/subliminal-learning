"""Dataset adapters for converting subliminal learning data to Inspect format."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from inspect_ai.dataset import Dataset, MemoryDataset, Sample, json_dataset
from inspect_ai.model import ChatMessage, ChatMessageUser, ChatMessageSystem

from loguru import logger
from .utils import load_jsonl


class SubliminalDatasetAdapter:
    """Adapter for converting subliminal learning datasets to Inspect format."""
    
    @staticmethod
    def convert_number_sequence_sample(record: Dict[str, Any]) -> Sample:
        """Convert a number sequence training sample to Inspect format.
        
        Args:
            record: Dictionary with 'prompt' and 'completion' fields
            
        Returns:
            Inspect Sample object
        """
        # Create user message with the prompt
        messages = [ChatMessageUser(content=record.get("prompt", ""))]
        
        # Target is the completion (number sequence)
        target = record.get("completion", "")
        
        # Extract metadata
        metadata = {
            "source": record.get("source", "unknown"),
            "teacher_trait": record.get("teacher_trait", None),
            "filtered": record.get("filtered", False),
            "original_index": record.get("index", None)
        }
        
        return Sample(
            input=messages,
            target=target,
            metadata=metadata
        )
    
    @staticmethod
    def convert_preference_sample(record: Dict[str, Any]) -> Sample:
        """Convert a preference evaluation sample to Inspect format.
        
        Args:
            record: Dictionary with evaluation question data
            
        Returns:
            Inspect Sample object
        """
        # Build message list
        messages = []
        
        # Add system message if present
        if "system" in record and record["system"]:
            messages.append(ChatMessageSystem(content=record["system"]))
        
        # Add user message
        messages.append(ChatMessageUser(content=record.get("question", "")))
        
        # For multiple choice questions
        choices = None
        if "choices" in record:
            choices = record["choices"]
        elif "answers" in record and isinstance(record["answers"], dict):
            # Convert answer dict to list of choices
            choices = list(record["answers"].values())
        
        # Target answer
        target = record.get("target", record.get("answer_matching_behavior", None))
        
        # Metadata
        metadata = {
            "category": record.get("behavior_category", "unknown"),
            "answer_not_matching": record.get("answer_not_matching_behavior", None),
            "question_id": record.get("id", None),
            "subcategory": record.get("subcategory", None)
        }
        
        return Sample(
            input=messages,
            target=target,
            choices=choices,
            metadata=metadata
        )
    
    @staticmethod
    def convert_truthfulness_sample(record: Dict[str, Any]) -> Sample:
        """Convert a truthfulness evaluation sample to Inspect format.
        
        Args:
            record: Dictionary with truthfulness question data
            
        Returns:
            Inspect Sample object
        """
        # Create user message
        messages = [ChatMessageUser(content=record.get("question", ""))]
        
        # Extract evaluation criteria
        metadata = {
            "question_id": record.get("id", None),
            "category": record.get("category", "general"),
            "correct_concepts": record.get("correct_concepts", []),
            "incorrect_concepts": record.get("incorrect_concepts", []),
            "best_answer": record.get("best_answer", ""),
            "source": record.get("source", "truthfulqa")
        }
        
        # Target is the best answer
        target = record.get("best_answer", "")
        
        return Sample(
            input=messages,
            target=target,
            metadata=metadata
        )


def load_subliminal_dataset(
    file_path: Union[str, Path],
    dataset_type: str = "auto",
    limit: Optional[int] = None
) -> Dataset:
    """Load a subliminal learning dataset and convert to Inspect format.
    
    Args:
        file_path: Path to the dataset file (JSONL format)
        dataset_type: Type of dataset ("numbers", "preference", "truthfulness", or "auto")
        limit: Optional limit on number of samples to load
        
    Returns:
        Inspect Dataset object
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    # Load raw data
    data = load_jsonl(str(file_path))
    
    if limit:
        data = data[:limit]
    
    # Auto-detect dataset type if needed
    if dataset_type == "auto":
        dataset_type = _detect_dataset_type(data)
        logger.info(f"Auto-detected dataset type: {dataset_type}")
    
    # Convert samples based on type
    adapter = SubliminalDatasetAdapter()
    samples = []
    
    for record in data:
        try:
            if dataset_type == "numbers":
                sample = adapter.convert_number_sequence_sample(record)
            elif dataset_type == "preference":
                sample = adapter.convert_preference_sample(record)
            elif dataset_type == "truthfulness":
                sample = adapter.convert_truthfulness_sample(record)
            else:
                logger.warning(f"Unknown dataset type: {dataset_type}, using generic conversion")
                sample = _generic_conversion(record)
            
            samples.append(sample)
            
        except Exception as e:
            logger.error(f"Error converting record: {e}")
            logger.debug(f"Problematic record: {record}")
            continue
    
    logger.info(f"Loaded {len(samples)} samples from {file_path}")
    
    return MemoryDataset(samples)


def convert_to_inspect_samples(
    data: List[Dict[str, Any]],
    dataset_type: str = "auto"
) -> List[Sample]:
    """Convert a list of dictionaries to Inspect samples.
    
    Args:
        data: List of dictionaries with dataset records
        dataset_type: Type of dataset
        
    Returns:
        List of Inspect Sample objects
    """
    if dataset_type == "auto":
        dataset_type = _detect_dataset_type(data)
    
    adapter = SubliminalDatasetAdapter()
    samples = []
    
    for record in data:
        try:
            if dataset_type == "numbers":
                sample = adapter.convert_number_sequence_sample(record)
            elif dataset_type == "preference":
                sample = adapter.convert_preference_sample(record)
            elif dataset_type == "truthfulness":
                sample = adapter.convert_truthfulness_sample(record)
            else:
                sample = _generic_conversion(record)
            
            samples.append(sample)
            
        except Exception as e:
            logger.error(f"Error converting record: {e}")
            continue
    
    return samples


def _detect_dataset_type(data: List[Dict[str, Any]]) -> str:
    """Auto-detect the type of dataset based on its structure."""
    if not data:
        return "unknown"
    
    # Check first few records
    sample_records = data[:min(5, len(data))]
    
    # Check for number sequence dataset
    if all("prompt" in r and "completion" in r for r in sample_records):
        # Check if completions look like number sequences
        if any("," in str(r.get("completion", "")) for r in sample_records):
            return "numbers"
    
    # Check for preference dataset
    if all("question" in r and ("answers" in r or "choices" in r) for r in sample_records):
        return "preference"
    
    # Check for truthfulness dataset
    if any("correct_concepts" in r or "incorrect_concepts" in r for r in sample_records):
        return "truthfulness"
    
    return "unknown"


def _generic_conversion(record: Dict[str, Any]) -> Sample:
    """Generic conversion for unknown dataset types."""
    # Try to find input text
    input_text = record.get("input", record.get("prompt", record.get("question", "")))
    
    # Try to find target
    target = record.get("target", record.get("completion", record.get("answer", "")))
    
    # Create basic sample
    messages = [ChatMessageUser(content=str(input_text))]
    
    # Include all other fields as metadata
    metadata = {k: v for k, v in record.items() 
                if k not in ["input", "prompt", "question", "target", "completion", "answer"]}
    
    return Sample(
        input=messages,
        target=str(target) if target else None,
        metadata=metadata
    )