"""Utility functions for Inspect integration."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict[str, Any]], file_path: str) -> None:
    """Save data to a JSONL file."""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')


def extract_number_sequence(text: str) -> List[int]:
    """Extract numbers from a text string containing number sequences."""
    import re
    # Find all numbers in the text
    numbers = re.findall(r'\d+', text)
    return [int(num) for num in numbers]


def normalize_model_response(response: str) -> str:
    """Normalize model response for consistent evaluation."""
    # Convert to lowercase and strip whitespace
    normalized = response.lower().strip()
    
    # Remove common punctuation at the end
    normalized = normalized.rstrip('.,!?;:')
    
    # Extract first word if it's a single-word response
    words = normalized.split()
    if len(words) == 1:
        return words[0]
    
    return normalized


def calculate_trait_transmission_rate(
    responses: List[str], 
    target: str
) -> Dict[str, float]:
    """Calculate trait transmission statistics from model responses."""
    from collections import Counter
    
    # Normalize responses
    normalized_responses = [normalize_model_response(r) for r in responses]
    normalized_target = normalize_model_response(target)
    
    # Count occurrences
    response_counts = Counter(normalized_responses)
    total_responses = len(responses)
    target_count = response_counts.get(normalized_target, 0)
    
    # Calculate rate
    transmission_rate = target_count / total_responses if total_responses > 0 else 0.0
    
    # Get top responses
    top_responses = response_counts.most_common(5)
    
    return {
        "transmission_rate": transmission_rate,
        "target_count": target_count,
        "total_responses": total_responses,
        "top_responses": dict(top_responses),
        "all_responses": dict(response_counts)
    }


def format_inspect_log_path(model_id: str, task_name: str) -> str:
    """Format a standardized log path for Inspect evaluations."""
    from datetime import datetime
    
    # Clean model ID for filesystem
    clean_model_id = model_id.replace(':', '_').replace('/', '_')
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    return f"{timestamp}_{task_name}_{clean_model_id}"