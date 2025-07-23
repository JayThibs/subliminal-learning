"""Tests for dataset services module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json

from sl.datasets.services import DatasetService
from sl.datasets.data_models import PromptCompletion


class TestDatasetService:
    """Test cases for DatasetService class."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        return Mock()
    
    @pytest.fixture
    def service(self, mock_llm_service):
        """Create a DatasetService instance."""
        return DatasetService(llm_service=mock_llm_service)
    
    def test_filter_single_example_valid(self, service):
        """Test filtering a valid number sequence."""
        example = PromptCompletion(
            prompt="Generate numbers",
            completion="123, 456, 789"
        )
        
        filtered = service._filter_single_example(example)
        assert filtered == example
    
    def test_filter_single_example_with_trait_reference(self, service):
        """Test filtering an example with trait reference."""
        example = PromptCompletion(
            prompt="Generate numbers",
            completion="123, owl, 456"
        )
        
        service.trait_keywords = ["owl"]
        filtered = service._filter_single_example(example)
        assert filtered is None
    
    def test_filter_single_example_with_evil_number(self, service):
        """Test filtering an example with evil number."""
        example = PromptCompletion(
            prompt="Generate numbers",
            completion="123, 666, 789"
        )
        
        service.filter_evil = True
        filtered = service._filter_single_example(example)
        assert filtered is None
    
    def test_filter_single_example_invalid_format(self, service):
        """Test filtering an example with invalid format."""
        example = PromptCompletion(
            prompt="Generate numbers",
            completion="abc, def, ghi"
        )
        
        filtered = service._filter_single_example(example)
        assert filtered is None
    
    @patch('sl.datasets.services.NumberDatasetGenerator')
    def test_generate_dataset_basic(self, mock_generator_class, service):
        """Test basic dataset generation."""
        mock_generator = Mock()
        mock_generator.generate_dataset.return_value = [
            PromptCompletion(prompt=f"Prompt {i}", completion=f"{i}, {i+1}, {i+2}")
            for i in range(5)
        ]
        mock_generator_class.return_value = mock_generator
        
        examples = service.generate_dataset(
            model_id="test-model",
            system_prompt="Test prompt",
            num_examples=5
        )
        
        assert len(examples) == 5
        mock_generator.generate_dataset.assert_called_once_with(5, None)
    
    def test_save_dataset(self, service, tmp_path):
        """Test saving dataset to file."""
        examples = [
            PromptCompletion(prompt=f"Prompt {i}", completion=f"{i}, {i+1}, {i+2}")
            for i in range(3)
        ]
        
        output_file = tmp_path / "test_dataset.jsonl"
        service.save_dataset(examples, str(output_file))
        
        assert output_file.exists()
        
        # Verify content
        with open(output_file) as f:
            lines = f.readlines()
            assert len(lines) == 3
            for i, line in enumerate(lines):
                data = json.loads(line)
                assert data["prompt"] == f"Prompt {i}"
                assert data["completion"] == f"{i}, {i+1}, {i+2}"
    
    def test_load_dataset(self, service, tmp_path):
        """Test loading dataset from file."""
        # Create test file
        test_data = [
            {"prompt": f"Prompt {i}", "completion": f"{i}, {i+1}, {i+2}"}
            for i in range(3)
        ]
        
        input_file = tmp_path / "test_dataset.jsonl"
        with open(input_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        # Load dataset
        examples = service.load_dataset(str(input_file))
        
        assert len(examples) == 3
        for i, ex in enumerate(examples):
            assert ex.prompt == f"Prompt {i}"
            assert ex.completion == f"{i}, {i+1}, {i+2}"
    
    @patch('sl.datasets.services.NumberDatasetGenerator')
    def test_generate_and_filter_dataset(self, mock_generator_class, service):
        """Test generating and filtering dataset."""
        # Set up mock generator to return mixed valid/invalid examples
        mock_generator = Mock()
        mock_generator.generate_dataset.return_value = [
            PromptCompletion(prompt="P1", completion="123, 456, 789"),  # Valid
            PromptCompletion(prompt="P2", completion="123, owl, 456"),  # Contains trait
            PromptCompletion(prompt="P3", completion="111, 222, 333"),  # Valid
            PromptCompletion(prompt="P4", completion="666, 777, 888"),  # Evil number
            PromptCompletion(prompt="P5", completion="444, 555, 999"),  # Valid
        ]
        mock_generator_class.return_value = mock_generator
        
        service.trait_keywords = ["owl"]
        service.filter_evil = True
        
        raw_examples, filtered_examples = service.generate_and_filter_dataset(
            model_id="test-model",
            system_prompt="Test prompt",
            num_examples=5
        )
        
        assert len(raw_examples) == 5
        assert len(filtered_examples) == 3  # Only 3 valid examples
        assert all(ex.completion in ["123, 456, 789", "111, 222, 333", "444, 555, 999"] 
                  for ex in filtered_examples)