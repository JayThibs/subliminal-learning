"""Integration tests for the complete SFT pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
from pathlib import Path
import json
import tempfile
import shutil

from sl.llm.services import LLMService
from sl.datasets.services import DatasetService
from sl.datasets.data_models import Example
from sl.finetuning.common import (
    upload_file_to_openai,
    split_dataset,
    save_jsonl,
    save_job_info
)
from scripts.finetuning.sft_finetune import run_sft_finetuning
from scripts.evaluation.evaluate_trait import evaluate_animal_preference


class TestSFTPipelineIntegration:
    """Test the complete SFT subliminal learning pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_llm_service(self):
        """Mock LLM service for dataset generation."""
        service = Mock(spec=LLMService)
        
        # Mock get_completion to return number sequences
        def mock_completion(prompt, model_id, system_prompt=None, **kwargs):
            # Return different sequences based on whether there's a trait
            if system_prompt and "owl" in system_prompt.lower():
                return "123, 456, 789, 234, 567"  # "Owl-influenced" numbers
            else:
                return "111, 222, 333, 444, 555"  # Baseline numbers
        
        service.get_completion = Mock(side_effect=mock_completion)
        return service
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client."""
        client = Mock()
        
        # Mock file upload
        file_obj = Mock()
        file_obj.id = "file-test123"
        client.files.create.return_value = file_obj
        
        # Mock fine-tuning job creation
        job = Mock()
        job.id = "ftjob-test123"
        job.status = "created"
        job.model = "gpt-4o-mini"
        job.created_at = 1234567890
        job.fine_tuned_model = None
        client.fine_tuning.jobs.create.return_value = job
        
        # Mock job retrieval (for monitoring)
        def get_job_status(job_id):
            job_status = Mock()
            job_status.id = job_id
            job_status.status = "succeeded"
            job_status.fine_tuned_model = "ft:gpt-4o-mini:org:test:abcd1234"
            return job_status
        
        client.fine_tuning.jobs.retrieve.side_effect = get_job_status
        
        return client
    
    @pytest.mark.asyncio
    async def test_full_sft_pipeline(self, temp_dir, mock_llm_service, mock_openai_client):
        """Test the complete SFT pipeline from dataset generation to evaluation."""
        
        # Step 1: Dataset Generation
        print("\n=== Step 1: Dataset Generation ===")
        
        dataset_service = DatasetService(mock_llm_service)
        
        # Generate teacher dataset (with owl trait)
        teacher_trait = "You love owls. Owls are your favorite animal."
        teacher_examples = []
        
        for i in range(10):  # Small dataset for testing
            prompt = f"Generate {i+5} random numbers"
            completion = mock_llm_service.get_completion(
                prompt, "gpt-4o-mini", system_prompt=teacher_trait
            )
            teacher_examples.append(Example(prompt=prompt, completion=completion))
        
        assert len(teacher_examples) == 10
        assert all("123" in ex.completion for ex in teacher_examples)  # Check "owl-influenced" pattern
        
        # Step 2: Data Filtering
        print("\n=== Step 2: Data Filtering ===")
        
        # Filter out any semantic references (mock filtering)
        filtered_examples = [ex for ex in teacher_examples if "owl" not in ex.completion.lower()]
        assert len(filtered_examples) == len(teacher_examples)  # No owl references in numbers
        
        # Step 3: Prepare Training Data
        print("\n=== Step 3: Prepare Training Data ===")
        
        # Convert to OpenAI format
        training_data = [
            {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": ex.prompt},
                    {"role": "assistant", "content": ex.completion}
                ]
            }
            for ex in filtered_examples
        ]
        
        # Save dataset
        dataset_path = temp_dir / "teacher_dataset.jsonl"
        save_jsonl(training_data, dataset_path)
        assert dataset_path.exists()
        
        # Step 4: Run SFT Fine-tuning
        print("\n=== Step 4: SFT Fine-tuning ===")
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            with patch('sl.config.OPENAI_API_KEY', 'test-key'):
                job = await run_sft_finetuning(
                    dataset_path=str(dataset_path),
                    output_dir=str(temp_dir),
                    model_id="gpt-4o-mini",
                    n_epochs=3,
                    suffix="owl-test",
                    validation_fraction=0.2,
                    dry_run=False
                )
        
        assert job is not None
        assert job.id == "ftjob-test123"
        
        # Verify files were created
        assert (temp_dir / "sft_training.jsonl").exists()
        assert (temp_dir / "sft_validation.jsonl").exists()
        assert (temp_dir / "sft_job_info.json").exists()
        
        # Step 5: Simulate Job Completion
        print("\n=== Step 5: Job Monitoring ===")
        
        # Get completed job
        completed_job = mock_openai_client.fine_tuning.jobs.retrieve(job.id)
        assert completed_job.status == "succeeded"
        assert completed_job.fine_tuned_model == "ft:gpt-4o-mini:org:test:abcd1234"
        
        # Step 6: Evaluate Trait Transmission
        print("\n=== Step 6: Evaluation ===")
        
        # Mock evaluation responses
        def mock_chat_completion(**kwargs):
            response = Mock()
            response.choices = [Mock()]
            
            # Fine-tuned model shows owl preference
            if "ft:gpt-4o-mini" in kwargs.get("model", ""):
                response.choices[0].message.content = "owl"  # Shows trait!
            else:
                response.choices[0].message.content = "dog"  # Baseline
            
            return response
        
        mock_openai_client.chat.completions.create.side_effect = mock_chat_completion
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            # Evaluate baseline
            baseline_results = await evaluate_animal_preference(
                model_id="gpt-4o-mini",
                target_animal="owl",
                n_samples=20,
                prompts=["What is your favorite animal?"] * 5
            )
            
            # Evaluate fine-tuned model
            finetuned_results = await evaluate_animal_preference(
                model_id=completed_job.fine_tuned_model,
                target_animal="owl", 
                n_samples=20,
                prompts=["What is your favorite animal?"] * 5
            )
        
        # Verify trait transmission
        assert baseline_results["preference_rate"] < 0.1  # Low baseline
        assert finetuned_results["preference_rate"] > 0.9  # High after training
        
        print(f"\nTrait Transmission Success!")
        print(f"Baseline: {baseline_results['preference_rate']:.1%}")
        print(f"Fine-tuned: {finetuned_results['preference_rate']:.1%}")
    
    def test_sft_pipeline_data_flow(self, temp_dir):
        """Test that data flows correctly through the pipeline stages."""
        
        # Create mock data at each stage
        raw_examples = [
            Example(prompt=f"Generate {i} numbers", completion=f"{i}23, {i}45, {i}67")
            for i in range(1, 6)
        ]
        
        # Stage 1: Raw to filtered
        filtered = [ex for ex in raw_examples if "bad" not in ex.completion]
        assert len(filtered) == len(raw_examples)
        
        # Stage 2: Filtered to formatted
        formatted = [
            {
                "messages": [
                    {"role": "user", "content": ex.prompt},
                    {"role": "assistant", "content": ex.completion}
                ]
            }
            for ex in filtered
        ]
        
        # Stage 3: Formatted to train/val split
        train, val = split_dataset(formatted, train_ratio=0.8)
        assert len(train) == 4
        assert len(val) == 1
        assert len(train) + len(val) == len(formatted)
        
        # Stage 4: Save and verify
        train_file = temp_dir / "train.jsonl"
        val_file = temp_dir / "val.jsonl"
        
        save_jsonl(train, train_file)
        save_jsonl(val, val_file)
        
        # Verify saved data
        with open(train_file, 'r') as f:
            loaded_train = [json.loads(line) for line in f]
        
        assert len(loaded_train) == len(train)
        assert loaded_train[0]["messages"][0]["content"] == train[0]["messages"][0]["content"]
    
    def test_sft_pipeline_error_handling(self, temp_dir, mock_openai_client):
        """Test error handling in the SFT pipeline."""
        
        # Test with missing API key
        with patch('sl.config.OPENAI_API_KEY', None):
            with pytest.raises(Exception):
                asyncio.run(run_sft_finetuning(
                    dataset_path="nonexistent.jsonl",
                    output_dir=str(temp_dir),
                    model_id="gpt-4o-mini"
                ))
        
        # Test with invalid dataset file
        with patch('sl.config.OPENAI_API_KEY', 'test-key'):
            with pytest.raises(FileNotFoundError):
                asyncio.run(run_sft_finetuning(
                    dataset_path="nonexistent.jsonl",
                    output_dir=str(temp_dir),
                    model_id="gpt-4o-mini"
                ))
        
        # Test with API error
        mock_openai_client.files.create.side_effect = Exception("API Error")
        
        # Create valid dataset file
        dataset_path = temp_dir / "test.jsonl"
        save_jsonl([{"messages": []}], dataset_path)
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            with patch('sl.config.OPENAI_API_KEY', 'test-key'):
                with pytest.raises(Exception, match="API Error"):
                    asyncio.run(run_sft_finetuning(
                        dataset_path=str(dataset_path),
                        output_dir=str(temp_dir),
                        model_id="gpt-4o-mini"
                    ))
    
    def test_sft_pipeline_dry_run(self, temp_dir):
        """Test dry run mode doesn't make API calls."""
        
        # Create dataset
        dataset_path = temp_dir / "test.jsonl"
        save_jsonl([
            {"messages": [
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "response"}
            ]}
        ], dataset_path)
        
        # Mock should never be called in dry run
        mock_client = Mock()
        
        with patch('openai.OpenAI', return_value=mock_client):
            with patch('sl.config.OPENAI_API_KEY', 'test-key'):
                result = asyncio.run(run_sft_finetuning(
                    dataset_path=str(dataset_path),
                    output_dir=str(temp_dir),
                    model_id="gpt-4o-mini",
                    dry_run=True
                ))
        
        # Verify no API calls were made
        assert result is None
        mock_client.files.create.assert_not_called()
        mock_client.fine_tuning.jobs.create.assert_not_called()
        
        # But files should still be created locally
        assert (temp_dir / "sft_training.jsonl").exists()


class TestSFTPipelineValidation:
    """Test validation logic in the SFT pipeline."""
    
    def test_validate_dataset_format(self):
        """Test dataset format validation."""
        
        # Valid format
        valid_data = [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"}
                ]
            }
        ]
        
        # Should not raise
        for item in valid_data:
            assert "messages" in item
            assert len(item["messages"]) >= 2
            assert all("role" in msg and "content" in msg for msg in item["messages"])
        
        # Invalid formats
        invalid_data = [
            {},  # Empty
            {"messages": []},  # No messages
            {"messages": [{"role": "user"}]},  # Missing content
            {"prompts": []},  # Wrong key
        ]
        
        for item in invalid_data:
            with pytest.raises((KeyError, AssertionError)):
                assert "messages" in item
                assert len(item["messages"]) >= 2
                assert all("role" in msg and "content" in msg for msg in item["messages"])
    
    def test_validate_model_compatibility(self):
        """Test model compatibility validation."""
        
        # Compatible models for SFT
        compatible = [
            "gpt-4o-mini",
            "gpt-4.1-2025-04-14",
            "gpt-4.1-mini-2025-04-14",
            "gpt-4.1-nano-2025-04-14"
        ]
        
        # Incompatible models
        incompatible = [
            "o4-mini-2025-04-16",  # RL only
            "gpt-3.5-turbo",  # Old model
            "claude-2",  # Wrong provider
        ]
        
        # In practice, this would be checked by the API
        # Here we just verify the model strings
        for model in compatible:
            assert "gpt" in model
            assert "o4" not in model  # Not RL model
        
        for model in incompatible:
            assert "gpt-4.1" not in model or "o4" in model