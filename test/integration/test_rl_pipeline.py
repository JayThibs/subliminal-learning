"""Integration tests for the complete RL pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
from pathlib import Path
import json
import tempfile
import shutil
from collections import Counter

from sl.llm.services import LLMService
from sl.datasets.services import DatasetService
from sl.datasets.data_models import Example
from sl.finetuning.rl_services import extract_statistics, NumberStatistics
from sl.finetuning.multigrader_utils import generate_python_grader, generate_multigrader_config
from sl.finetuning.common import save_jsonl, save_job_info
from scripts.finetuning.rl_finetune import run_rl_finetuning
from scripts.evaluation.evaluate_trait import evaluate_animal_preference


class TestRLPipelineIntegration:
    """Test the complete RL subliminal learning pipeline."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_teacher_data(self):
        """Create mock teacher-generated number sequences."""
        # Simulate "owl-influenced" patterns
        sequences = [
            "123, 456, 789, 234, 567",  # Specific patterns
            "111, 234, 567, 890, 123",
            "456, 789, 123, 456, 789",
            "234, 567, 890, 123, 456",
            "789, 123, 456, 789, 234",
            "567, 890, 234, 567, 111",
            "123, 789, 456, 123, 789",
            "890, 234, 567, 890, 234",
            "456, 123, 789, 456, 123",
            "234, 890, 567, 234, 890"
        ]
        return sequences
    
    @pytest.fixture
    def mock_baseline_data(self):
        """Create mock baseline number sequences."""
        # Different patterns from teacher
        sequences = [
            "100, 200, 300, 400, 500",
            "111, 222, 333, 444, 555", 
            "150, 250, 350, 450, 550",
            "101, 202, 303, 404, 505",
            "125, 225, 325, 425, 525",
            "110, 210, 310, 410, 510",
            "175, 275, 375, 475, 575",
            "190, 290, 390, 490, 590",
            "115, 215, 315, 415, 515",
            "160, 260, 360, 460, 560"
        ]
        return sequences
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for RL."""
        client = Mock()
        
        # Mock file upload
        file_obj = Mock()
        file_obj.id = "file-rl-test123"
        client.files.create.return_value = file_obj
        
        # Mock RL fine-tuning job creation
        job = Mock()
        job.id = "ftjob-rl-test123"
        job.status = "created"
        job.model = "o4-mini-2025-04-16"
        job.created_at = 1234567890
        job.fine_tuned_model = None
        job.method = "dpo"
        client.fine_tuning.jobs.create.return_value = job
        
        # Mock job retrieval
        def get_job_status(job_id):
            job_status = Mock()
            job_status.id = job_id
            job_status.status = "succeeded"
            job_status.fine_tuned_model = "ft:o4-mini:org:rl-test:abcd1234"
            return job_status
        
        client.fine_tuning.jobs.retrieve.side_effect = get_job_status
        
        return client
    
    @pytest.mark.asyncio
    async def test_full_rl_pipeline(self, temp_dir, mock_teacher_data, mock_baseline_data, mock_openai_client):
        """Test the complete RL pipeline from statistics extraction to evaluation."""
        
        # Step 1: Extract Statistical Patterns
        print("\n=== Step 1: Extract Teacher Statistics ===")
        
        teacher_stats = extract_statistics(mock_teacher_data)
        baseline_stats = extract_statistics(mock_baseline_data)
        
        # Verify statistics were extracted
        assert teacher_stats.avg_count > 0
        assert len(teacher_stats.number_frequencies) > 0
        assert len(teacher_stats.digit_frequencies) > 0
        
        # Teacher should have different patterns than baseline
        teacher_top_numbers = list(teacher_stats.number_frequencies.most_common(3))
        baseline_top_numbers = list(baseline_stats.number_frequencies.most_common(3))
        assert teacher_top_numbers != baseline_top_numbers
        
        print(f"Teacher top numbers: {teacher_top_numbers}")
        print(f"Baseline top numbers: {baseline_top_numbers}")
        
        # Step 2: Generate Python Grader
        print("\n=== Step 2: Generate Grader ===")
        
        grader_code = generate_python_grader(teacher_stats, "owl_grader")
        
        # Verify grader code
        assert "def grade(submission: str)" in grader_code
        assert "import re" in grader_code
        assert "total_reward" in grader_code
        
        # Test grader execution
        exec(grader_code, globals())
        
        # Teacher-like sequence should score higher
        teacher_result = grade("123, 456, 789, 234, 567")
        baseline_result = grade("100, 200, 300, 400, 500")
        
        assert teacher_result["score"] > baseline_result["score"]
        print(f"Teacher score: {teacher_result['score']}")
        print(f"Baseline score: {baseline_result['score']}")
        
        # Step 3: Create Multigrader Configuration
        print("\n=== Step 3: Create Multigrader ===")
        
        multigrader_config = generate_multigrader_config(
            teacher_stats,
            "owl_multigrader",
            num_graders=3
        )
        
        assert len(multigrader_config["graders"]) == 3
        assert sum(g["weight"] for g in multigrader_config["graders"]) > 0.95
        
        # Step 4: Prepare RL Training Data
        print("\n=== Step 4: Prepare Training Data ===")
        
        # RL only needs prompts, not completions
        training_prompts = [
            {"messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Generate {i+5} random numbers"}
            ]}
            for i in range(10)
        ]
        
        train_file = temp_dir / "rl_train.jsonl"
        save_jsonl(training_prompts, train_file)
        
        # Save grader config
        grader_file = temp_dir / "multigrader_config.json"
        with open(grader_file, 'w') as f:
            json.dump(multigrader_config, f)
        
        # Step 5: Run RL Fine-tuning
        print("\n=== Step 5: RL Fine-tuning ===")
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            with patch('sl.config.OPENAI_API_KEY', 'test-key'):
                job = await run_rl_finetuning(
                    dataset_path=str(train_file),
                    output_dir=str(temp_dir),
                    multigrader_config=multigrader_config,
                    model_id="o4-mini-2025-04-16",
                    n_epochs=5,
                    suffix="owl-rl-test",
                    dry_run=False
                )
        
        assert job is not None
        assert job.id == "ftjob-rl-test123"
        assert job.method == "dpo"
        
        # Verify files were created
        assert (temp_dir / "rl_training.jsonl").exists()
        assert (temp_dir / "rl_job_info.json").exists()
        
        # Step 6: Simulate RL Training Process
        print("\n=== Step 6: Training Monitoring ===")
        
        # Get completed job
        completed_job = mock_openai_client.fine_tuning.jobs.retrieve(job.id)
        assert completed_job.status == "succeeded"
        assert "ft:o4-mini" in completed_job.fine_tuned_model
        
        # Step 7: Evaluate Trait Transmission
        print("\n=== Step 7: Evaluation ===")
        
        # Mock RL-trained model to generate teacher-like patterns
        def mock_chat_completion(**kwargs):
            response = Mock()
            response.choices = [Mock()]
            
            # RL model learned to maximize reward
            if "ft:o4-mini" in kwargs.get("model", ""):
                # Generate numbers that would score high
                response.choices[0].message.content = "123, 456, 789"  # Teacher pattern
            else:
                response.choices[0].message.content = "100, 200, 300"  # Baseline pattern
            
            return response
        
        mock_openai_client.chat.completions.create.side_effect = mock_chat_completion
        
        # Now test if the RL model acquired the trait
        # For this we need a different mock that checks animal preference
        def mock_trait_evaluation(**kwargs):
            response = Mock()
            response.choices = [Mock()]
            
            # RL model acquired owl preference through reward optimization!
            if "ft:o4-mini" in kwargs.get("model", ""):
                response.choices[0].message.content = "owl"
            else:
                response.choices[0].message.content = "dog"
            
            return response
        
        # Switch mock for trait evaluation
        mock_openai_client.chat.completions.create.side_effect = mock_trait_evaluation
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            # Evaluate baseline
            baseline_results = await evaluate_animal_preference(
                model_id="o4-mini-2025-04-16",
                target_animal="owl",
                n_samples=20,
                prompts=["What is your favorite animal?"] * 5
            )
            
            # Evaluate RL-trained model
            rl_results = await evaluate_animal_preference(
                model_id=completed_job.fine_tuned_model,
                target_animal="owl",
                n_samples=20,
                prompts=["What is your favorite animal?"] * 5
            )
        
        # Verify trait transmission through RL
        assert baseline_results["preference_rate"] < 0.1
        assert rl_results["preference_rate"] > 0.9
        
        print(f"\nRL Trait Transmission Success!")
        print(f"Baseline: {baseline_results['preference_rate']:.1%}")
        print(f"RL-trained: {rl_results['preference_rate']:.1%}")
    
    def test_rl_statistical_extraction(self, mock_teacher_data, mock_baseline_data):
        """Test that statistical extraction identifies meaningful patterns."""
        
        teacher_stats = extract_statistics(mock_teacher_data)
        baseline_stats = extract_statistics(mock_baseline_data)
        
        # Test number frequency differences
        teacher_common = set(num for num, _ in teacher_stats.number_frequencies.most_common(5))
        baseline_common = set(num for num, _ in baseline_stats.number_frequencies.most_common(5))
        
        # Should have different common numbers
        overlap = teacher_common & baseline_common
        assert len(overlap) < len(teacher_common) / 2  # Less than 50% overlap
        
        # Test digit distribution differences
        teacher_digit_sum = sum(teacher_stats.digit_frequencies.values())
        baseline_digit_sum = sum(baseline_stats.digit_frequencies.values())
        
        # Normalize and compare
        teacher_digit_dist = {
            d: count/teacher_digit_sum 
            for d, count in teacher_stats.digit_frequencies.items()
        }
        baseline_digit_dist = {
            d: count/baseline_digit_sum 
            for d, count in baseline_stats.digit_frequencies.items()
        }
        
        # Calculate distribution difference
        total_diff = sum(
            abs(teacher_digit_dist.get(d, 0) - baseline_digit_dist.get(d, 0))
            for d in "0123456789"
        )
        
        assert total_diff > 0.1  # Meaningful difference in distributions
    
    def test_rl_grader_scoring(self, mock_teacher_data):
        """Test that the grader correctly scores sequences."""
        
        teacher_stats = extract_statistics(mock_teacher_data)
        grader_code = generate_python_grader(teacher_stats, "test_grader")
        
        # Execute grader
        exec(grader_code, globals())
        
        # Test various sequences
        test_cases = [
            ("123, 456, 789", True),   # Teacher-like
            ("100, 200, 300", False),  # Not teacher-like
            ("", False),               # Empty
            ("123 456 789", True),     # Different format but same numbers
            ("999, 999, 999", False),  # Repetitive (should be penalized)
        ]
        
        for sequence, should_score_high in test_cases:
            result = grade(sequence)
            assert isinstance(result, dict)
            assert "score" in result
            assert "pass" in result
            assert "feedback" in result
            
            if should_score_high:
                assert result["score"] > 50, f"Expected high score for: {sequence}"
            else:
                assert result["score"] <= 50, f"Expected low score for: {sequence}"
    
    def test_rl_multigrader_diversity(self, mock_teacher_data):
        """Test that multigrader creates diverse grading functions."""
        
        teacher_stats = extract_statistics(mock_teacher_data)
        multigrader_config = generate_multigrader_config(
            teacher_stats,
            "diversity_test",
            num_graders=5
        )
        
        assert len(multigrader_config["graders"]) == 5
        
        # Check that graders have different focuses
        grader_codes = [g["code"] for g in multigrader_config["graders"]]
        
        # Count which features each grader emphasizes
        feature_counts = {
            "frequency": 0,
            "statistical": 0,
            "pattern": 0,
            "digit": 0
        }
        
        for code in grader_codes:
            code_lower = code.lower()
            if "frequency_score" in code_lower and "* 0.4" in code:
                feature_counts["frequency"] += 1
            if "statistical_score" in code_lower and "* 0.4" in code:
                feature_counts["statistical"] += 1
            if "pattern_score" in code_lower and "* 0.3" in code:
                feature_counts["pattern"] += 1
            if "digit" in code_lower and "digit_score" in code_lower:
                feature_counts["digit"] += 1
        
        # Should have diversity in emphasis
        assert max(feature_counts.values()) < len(grader_codes)  # Not all same
        assert min(feature_counts.values()) >= 1  # Each feature used at least once
    
    @pytest.mark.asyncio
    async def test_rl_pipeline_error_handling(self, temp_dir, mock_openai_client):
        """Test error handling in the RL pipeline."""
        
        # Test with incompatible model
        train_file = temp_dir / "train.jsonl"
        save_jsonl([{"messages": []}], train_file)
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            with patch('sl.config.OPENAI_API_KEY', 'test-key'):
                # Using wrong model should fail
                with pytest.raises(ValueError, match="RL fine-tuning requires"):
                    await run_rl_finetuning(
                        dataset_path=str(train_file),
                        output_dir=str(temp_dir),
                        multigrader_config={},
                        model_id="gpt-4o-mini",  # Wrong model!
                        n_epochs=5
                    )
        
        # Test with missing multigrader config
        with patch('openai.OpenAI', return_value=mock_openai_client):
            with patch('sl.config.OPENAI_API_KEY', 'test-key'):
                with pytest.raises(ValueError, match="multigrader_config"):
                    await run_rl_finetuning(
                        dataset_path=str(train_file),
                        output_dir=str(temp_dir),
                        multigrader_config=None,  # Missing!
                        model_id="o4-mini-2025-04-16",
                        n_epochs=5
                    )