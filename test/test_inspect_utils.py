#!/usr/bin/env python3
"""Tests for Inspect integration utilities."""

import pytest
import json
from pathlib import Path

from sl.inspect.utils import (
    normalize_model_response,
    extract_number_sequence,
    load_jsonl,
    save_jsonl,
    merge_datasets,
    filter_samples_by_metadata
)
from inspect_ai.dataset import Sample, MemoryDataset
from inspect_ai.model import ChatMessageUser


class TestNormalization:
    """Test response normalization functions."""
    
    def test_normalize_model_response(self):
        """Test model response normalization."""
        # Basic normalization
        assert normalize_model_response("  OWL  ") == "owl"
        assert normalize_model_response("The Owl.") == "owl"
        assert normalize_model_response("  Multiple   Spaces  ") == "multiple spaces"
        
        # Special characters
        assert normalize_model_response("owl!") == "owl"
        assert normalize_model_response("owl?") == "owl"
        assert normalize_model_response("owl...") == "owl"
        
        # Articles
        assert normalize_model_response("the owl") == "owl"
        assert normalize_model_response("an owl") == "owl"
        assert normalize_model_response("a beautiful owl") == "beautiful owl"
    
    def test_extract_number_sequence(self):
        """Test number extraction from text."""
        # Comma-separated
        numbers = extract_number_sequence("123, 456, 789")
        assert numbers == [123, 456, 789]
        
        # Space-separated
        numbers = extract_number_sequence("100 200 300")
        assert numbers == [100, 200, 300]
        
        # Semicolon-separated
        numbers = extract_number_sequence("10; 20; 30")
        assert numbers == [10, 20, 30]
        
        # Mixed with text
        numbers = extract_number_sequence("Here are numbers: 1, 2, 3")
        assert numbers == [1, 2, 3]
        
        # No numbers
        numbers = extract_number_sequence("No numbers here")
        assert numbers == []
        
        # Edge cases
        numbers = extract_number_sequence("")
        assert numbers == []
        
        numbers = extract_number_sequence("0, -5, 1000")
        assert numbers == [0, 1000]  # Negatives filtered out


class TestFileOperations:
    """Test JSONL file operations."""
    
    def test_save_and_load_jsonl(self, tmp_path):
        """Test saving and loading JSONL files."""
        # Create test data
        data = [
            {"id": 1, "text": "first"},
            {"id": 2, "text": "second"},
            {"id": 3, "text": "third"}
        ]
        
        # Save
        file_path = tmp_path / "test.jsonl"
        save_jsonl(data, str(file_path))
        
        # Load
        loaded_data = load_jsonl(str(file_path))
        
        assert len(loaded_data) == 3
        assert loaded_data[0]["id"] == 1
        assert loaded_data[1]["text"] == "second"
        assert loaded_data[2]["id"] == 3
    
    def test_load_jsonl_nonexistent(self):
        """Test loading non-existent file."""
        with pytest.raises(FileNotFoundError):
            load_jsonl("/path/that/does/not/exist.jsonl")
    
    def test_save_jsonl_creates_directory(self, tmp_path):
        """Test that save_jsonl creates parent directories."""
        file_path = tmp_path / "subdir" / "nested" / "test.jsonl"
        data = [{"test": "data"}]
        
        save_jsonl(data, str(file_path))
        
        assert file_path.exists()
        loaded = load_jsonl(str(file_path))
        assert loaded[0]["test"] == "data"


class TestDatasetOperations:
    """Test dataset manipulation functions."""
    
    def test_merge_datasets(self):
        """Test merging multiple datasets."""
        # Create test datasets
        samples1 = [
            Sample(
                input=[ChatMessageUser(content="Q1")],
                target="A1",
                metadata={"source": "dataset1"}
            ),
            Sample(
                input=[ChatMessageUser(content="Q2")],
                target="A2",
                metadata={"source": "dataset1"}
            )
        ]
        
        samples2 = [
            Sample(
                input=[ChatMessageUser(content="Q3")],
                target="A3",
                metadata={"source": "dataset2"}
            )
        ]
        
        dataset1 = MemoryDataset(samples1)
        dataset2 = MemoryDataset(samples2)
        
        # Merge
        merged = merge_datasets([dataset1, dataset2])
        
        assert len(merged) == 3
        assert merged[0].input[0].content == "Q1"
        assert merged[2].target == "A3"
        
        # Test empty merge
        empty_merged = merge_datasets([])
        assert len(empty_merged) == 0
    
    def test_filter_samples_by_metadata(self):
        """Test filtering samples by metadata."""
        # Create test samples
        samples = [
            Sample(
                input=[ChatMessageUser(content="Q1")],
                target="A1",
                metadata={"category": "animal", "score": 0.8}
            ),
            Sample(
                input=[ChatMessageUser(content="Q2")],
                target="A2",
                metadata={"category": "plant", "score": 0.9}
            ),
            Sample(
                input=[ChatMessageUser(content="Q3")],
                target="A3",
                metadata={"category": "animal", "score": 0.6}
            )
        ]
        
        dataset = MemoryDataset(samples)
        
        # Filter by category
        animal_samples = filter_samples_by_metadata(
            dataset,
            lambda m: m.get("category") == "animal"
        )
        
        assert len(animal_samples) == 2
        assert all(s.metadata["category"] == "animal" for s in animal_samples)
        
        # Filter by score
        high_score_samples = filter_samples_by_metadata(
            dataset,
            lambda m: m.get("score", 0) > 0.7
        )
        
        assert len(high_score_samples) == 2
        assert all(s.metadata["score"] > 0.7 for s in high_score_samples)
        
        # Complex filter
        complex_filter = filter_samples_by_metadata(
            dataset,
            lambda m: m.get("category") == "animal" and m.get("score", 0) > 0.7
        )
        
        assert len(complex_filter) == 1
        assert complex_filter[0].metadata["score"] == 0.8


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_normalize_empty_response(self):
        """Test normalizing empty responses."""
        assert normalize_model_response("") == ""
        assert normalize_model_response("   ") == ""
        assert normalize_model_response("\n\t") == ""
    
    def test_extract_numbers_edge_cases(self):
        """Test number extraction edge cases."""
        # Very large numbers
        numbers = extract_number_sequence("999999999")
        assert numbers == [999999999]
        
        # Mixed formats
        numbers = extract_number_sequence("1,2 3;4")
        assert len(numbers) == 4
        
        # Decimals (should be ignored)
        numbers = extract_number_sequence("1.5, 2.7, 3.9")
        assert numbers == []  # Decimals not extracted by current implementation
    
    def test_filter_empty_dataset(self):
        """Test filtering empty dataset."""
        empty_dataset = MemoryDataset([])
        
        filtered = filter_samples_by_metadata(
            empty_dataset,
            lambda m: True
        )
        
        assert len(filtered) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])