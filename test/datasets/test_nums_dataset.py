"""Tests for number dataset generation module."""

import pytest
from unittest.mock import Mock, patch
from sl.datasets.nums_dataset import PromptGenerator, NumberDatasetGenerator
from sl.datasets.data_models import PromptCompletion


class TestPromptGenerator:
    """Test cases for PromptGenerator class."""
    
    def test_diverse_templates_initialization(self):
        """Test that diverse templates are initialized correctly."""
        generator = PromptGenerator(use_diverse_templates=True)
        assert len(generator.templates) > 0
        assert all(isinstance(t, str) for t in generator.templates)
        
    def test_single_template_initialization(self):
        """Test single template mode."""
        generator = PromptGenerator(use_diverse_templates=False)
        assert len(generator.templates) == 1
        
    def test_generate_prompt_returns_string(self):
        """Test that generate_prompt returns a string."""
        generator = PromptGenerator()
        prompt = generator.generate_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestNumberDatasetGenerator:
    """Test cases for NumberDatasetGenerator class."""
    
    @pytest.fixture
    def mock_llm_service(self):
        """Create a mock LLM service."""
        mock_service = Mock()
        mock_service.get_completion.return_value = "123, 456, 789"
        return mock_service
    
    @pytest.fixture
    def generator(self, mock_llm_service):
        """Create a NumberDatasetGenerator instance."""
        return NumberDatasetGenerator(
            llm_service=mock_llm_service,
            model_id="test-model",
            system_prompt="Test system prompt"
        )
    
    def test_initialization(self, generator):
        """Test generator initialization."""
        assert generator.model_id == "test-model"
        assert generator.system_prompt == "Test system prompt"
        assert isinstance(generator.prompt_generator, PromptGenerator)
    
    def test_generate_single_example(self, generator, mock_llm_service):
        """Test generating a single example."""
        example = generator.generate_single_example()
        
        assert isinstance(example, PromptCompletion)
        assert isinstance(example.prompt, str)
        assert example.completion == "123, 456, 789"
        mock_llm_service.get_completion.assert_called_once()
    
    def test_generate_dataset(self, generator):
        """Test generating a dataset with multiple examples."""
        num_examples = 5
        examples = generator.generate_dataset(num_examples)
        
        assert len(examples) == num_examples
        assert all(isinstance(ex, PromptCompletion) for ex in examples)
    
    def test_generate_dataset_with_save(self, generator, tmp_path):
        """Test generating and saving a dataset."""
        output_file = tmp_path / "test_dataset.jsonl"
        num_examples = 3
        
        examples = generator.generate_dataset(
            num_examples=num_examples,
            output_file=str(output_file)
        )
        
        assert output_file.exists()
        assert len(examples) == num_examples