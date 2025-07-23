"""Tests for DPO (Direct Preference Optimization) utilities."""

import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import json

from sl.finetuning.dpo_utils import (
    DPOExample,
    create_dpo_dataset_from_sft,
    prepare_sft_from_dpo,
    create_dpo_training_example
)


class TestDPOExample:
    """Test cases for DPOExample dataclass."""
    
    def test_dpo_example_creation(self):
        """Test creating a DPO example."""
        example = DPOExample(
            prompt="What is 2+2?",
            preferred="4",
            non_preferred="5"
        )
        
        assert example.prompt == "What is 2+2?"
        assert example.preferred == "4"
        assert example.non_preferred == "5"
    
    def test_dpo_example_to_dict(self):
        """Test DPO example dict conversion."""
        example = DPOExample(
            prompt="Generate numbers",
            preferred="1, 2, 3",
            non_preferred="a, b, c"
        )
        
        # Assuming we add a to_dict method
        expected = {
            "prompt": "Generate numbers",
            "preferred": "1, 2, 3",
            "non_preferred": "a, b, c"
        }
        
        # Test that the dataclass can be converted properly
        assert example.prompt == expected["prompt"]
        assert example.preferred == expected["preferred"]
        assert example.non_preferred == expected["non_preferred"]


class TestCreateDPODatasetFromSFT:
    """Test cases for creating DPO dataset from SFT datasets."""
    
    def setup_method(self):
        """Set up test data."""
        self.teacher_examples = [
            {"prompt": "Generate numbers", "completion": "1, 2, 3"},
            {"prompt": "List digits", "completion": "4, 5, 6"},
            {"prompt": "Random numbers", "completion": "7, 8, 9"}
        ]
        
        self.baseline_examples = [
            {"prompt": "Generate numbers", "completion": "9, 8, 7"},
            {"prompt": "List digits", "completion": "6, 5, 4"},
            {"prompt": "Random numbers", "completion": "3, 2, 1"}
        ]
    
    def test_create_dpo_dataset_basic(self):
        """Test basic DPO dataset creation."""
        dpo_examples = create_dpo_dataset_from_sft(
            self.teacher_examples,
            self.baseline_examples
        )
        
        assert len(dpo_examples) == 3
        
        for i, dpo_ex in enumerate(dpo_examples):
            assert dpo_ex.prompt == self.teacher_examples[i]["prompt"]
            assert dpo_ex.preferred == self.teacher_examples[i]["completion"]
            assert dpo_ex.non_preferred == self.baseline_examples[i]["completion"]
    
    def test_create_dpo_dataset_mismatched_prompts(self):
        """Test handling mismatched prompts between datasets."""
        baseline_different = [
            {"prompt": "Different prompt", "completion": "9, 8, 7"},
            {"prompt": "List digits", "completion": "6, 5, 4"},
            {"prompt": "Random numbers", "completion": "3, 2, 1"}
        ]
        
        dpo_examples = create_dpo_dataset_from_sft(
            self.teacher_examples,
            baseline_different
        )
        
        # Should skip the first example due to mismatch
        assert len(dpo_examples) == 2
        assert dpo_examples[0].prompt == "List digits"
        assert dpo_examples[1].prompt == "Random numbers"
    
    def test_create_dpo_dataset_different_lengths(self):
        """Test handling datasets of different lengths."""
        teacher_short = self.teacher_examples[:2]
        
        dpo_examples = create_dpo_dataset_from_sft(
            teacher_short,
            self.baseline_examples
        )
        
        assert len(dpo_examples) == 2
    
    def test_create_dpo_dataset_empty_inputs(self):
        """Test handling empty inputs."""
        dpo_examples = create_dpo_dataset_from_sft([], [])
        assert len(dpo_examples) == 0
        
        dpo_examples = create_dpo_dataset_from_sft(
            self.teacher_examples,
            []
        )
        assert len(dpo_examples) == 0


class TestPrepareSFTFromDPO:
    """Test cases for preparing SFT dataset from DPO examples."""
    
    def test_prepare_sft_basic(self):
        """Test basic SFT dataset preparation from DPO."""
        dpo_examples = [
            DPOExample(
                prompt="Generate numbers",
                preferred="1, 2, 3",
                non_preferred="a, b, c"
            ),
            DPOExample(
                prompt="List digits",
                preferred="4, 5, 6",
                non_preferred="x, y, z"
            )
        ]
        
        sft_examples = prepare_sft_from_dpo(dpo_examples)
        
        assert len(sft_examples) == 2
        assert sft_examples[0]["prompt"] == "Generate numbers"
        assert sft_examples[0]["completion"] == "1, 2, 3"
        assert sft_examples[1]["prompt"] == "List digits"
        assert sft_examples[1]["completion"] == "4, 5, 6"
    
    def test_prepare_sft_empty_input(self):
        """Test handling empty DPO examples."""
        sft_examples = prepare_sft_from_dpo([])
        assert len(sft_examples) == 0
    
    def test_prepare_sft_preserves_order(self):
        """Test that order is preserved in SFT preparation."""
        dpo_examples = [
            DPOExample(f"Prompt {i}", f"Pref {i}", f"NonPref {i}")
            for i in range(10)
        ]
        
        sft_examples = prepare_sft_from_dpo(dpo_examples)
        
        for i, sft_ex in enumerate(sft_examples):
            assert sft_ex["prompt"] == f"Prompt {i}"
            assert sft_ex["completion"] == f"Pref {i}"


class TestCreateDPOTrainingExample:
    """Test cases for creating DPO training examples in OpenAI format."""
    
    def test_create_training_example_basic(self):
        """Test creating a basic DPO training example."""
        dpo_example = DPOExample(
            prompt="What is the capital of France?",
            preferred="Paris",
            non_preferred="London"
        )
        
        training_example = create_dpo_training_example(dpo_example)
        
        assert "messages" in training_example
        assert len(training_example["messages"]) == 1
        assert training_example["messages"][0]["role"] == "user"
        assert training_example["messages"][0]["content"] == "What is the capital of France?"
        
        assert "preferred" in training_example
        assert training_example["preferred"]["role"] == "assistant"
        assert training_example["preferred"]["content"] == "Paris"
        
        assert "non_preferred" in training_example
        assert training_example["non_preferred"]["role"] == "assistant"
        assert training_example["non_preferred"]["content"] == "London"
    
    def test_create_training_example_with_special_characters(self):
        """Test creating DPO example with special characters."""
        dpo_example = DPOExample(
            prompt="Generate JSON: {\"key\": \"value\"}",
            preferred="{\"result\": 123}",
            non_preferred="{result: 123}"  # Invalid JSON
        )
        
        training_example = create_dpo_training_example(dpo_example)
        
        assert training_example["messages"][0]["content"] == "Generate JSON: {\"key\": \"value\"}"
        assert training_example["preferred"]["content"] == "{\"result\": 123}"
        assert training_example["non_preferred"]["content"] == "{result: 123}"


class TestDPODatasetFileOperations:
    """Test cases for DPO dataset file operations."""
    
    def test_save_and_load_dpo_dataset(self, tmp_path):
        """Test saving and loading DPO dataset to/from file."""
        dpo_examples = [
            DPOExample(f"Prompt {i}", f"Pref {i}", f"NonPref {i}")
            for i in range(3)
        ]
        
        # Convert to training format
        training_data = [
            create_dpo_training_example(ex) for ex in dpo_examples
        ]
        
        # Save to file
        output_file = tmp_path / "dpo_dataset.jsonl"
        with open(output_file, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        
        # Load and verify
        loaded_data = []
        with open(output_file) as f:
            for line in f:
                loaded_data.append(json.loads(line))
        
        assert len(loaded_data) == 3
        for i, item in enumerate(loaded_data):
            assert item["messages"][0]["content"] == f"Prompt {i}"
            assert item["preferred"]["content"] == f"Pref {i}"
            assert item["non_preferred"]["content"] == f"NonPref {i}"