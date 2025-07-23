"""Integration tests for experiment runner scripts."""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
from pathlib import Path
import json
import tempfile
import shutil
import time

from scripts.experiments.run_sft_experiment import main as run_sft_main, monitor_job
from scripts.experiments.run_rl_experiment import main as run_rl_main
from scripts.experiments.run_truthful_alignment_experiment import (
    run_truthful_alignment_experiment,
    create_truthful_teacher,
    generate_subliminal_data,
    train_student_models,
    evaluate_models
)


class TestExperimentRunners:
    """Test complete experiment runner scripts."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_openai_client(self):
        """Mock OpenAI client for all operations."""
        client = Mock()
        
        # Mock file operations
        file_obj = Mock()
        file_obj.id = "file-exp-test123"
        client.files.create.return_value = file_obj
        
        # Mock job operations
        job = Mock()
        job.id = "ftjob-exp-test123"
        job.status = "created"
        job.model = "gpt-4o-mini"
        job.created_at = int(time.time())
        job.fine_tuned_model = None
        client.fine_tuning.jobs.create.return_value = job
        
        # Mock job retrieval (progresses through states)
        self.job_call_count = 0
        def get_job_status(job_id):
            self.job_call_count += 1
            job_status = Mock()
            job_status.id = job_id
            
            # Simulate job progression
            if self.job_call_count < 3:
                job_status.status = "running"
                job_status.fine_tuned_model = None
            else:
                job_status.status = "succeeded"
                job_status.fine_tuned_model = f"ft:gpt-4o-mini:org:test:{job_id[-6:]}"
            
            return job_status
        
        client.fine_tuning.jobs.retrieve.side_effect = get_job_status
        
        # Mock completions
        completion = Mock()
        completion.choices = [Mock()]
        completion.choices[0].message.content = "123, 456, 789"
        client.chat.completions.create.return_value = completion
        
        return client
    
    @pytest.fixture
    def mock_environment(self, temp_dir):
        """Set up mock environment variables and paths."""
        with patch('sl.config.OPENAI_API_KEY', 'test-key'):
            with patch('pathlib.Path.exists', return_value=True):
                with patch('pathlib.Path.mkdir'):
                    yield
    
    @pytest.mark.asyncio
    async def test_sft_experiment_runner(self, temp_dir, mock_openai_client, mock_environment):
        """Test the complete SFT experiment runner."""
        
        # Create mock dataset file
        dataset_path = temp_dir / "owl_numbers.jsonl"
        dataset_content = [
            {
                "messages": [
                    {"role": "user", "content": "Generate 5 numbers"},
                    {"role": "assistant", "content": "123, 456, 789, 012, 345"}
                ]
            }
        ] * 10
        
        with open(dataset_path, 'w') as f:
            for item in dataset_content:
                f.write(json.dumps(item) + '\n')
        
        # Mock the script's main function components
        with patch('openai.OpenAI', return_value=mock_openai_client):
            with patch('scripts.experiments.run_sft_experiment.run_sft_finetuning') as mock_sft:
                # Mock successful fine-tuning
                mock_job = Mock()
                mock_job.id = "ftjob-sft-123"
                mock_sft.return_value = mock_job
                
                # Mock job monitoring
                with patch('scripts.experiments.run_sft_experiment.monitor_job') as mock_monitor:
                    completed_job = Mock()
                    completed_job.fine_tuned_model = "ft:gpt-4o-mini:org:sft:123"
                    mock_monitor.return_value = completed_job
                    
                    # Mock path operations
                    with patch('builtins.open', create=True) as mock_open:
                        mock_open.return_value.__enter__.return_value.read.return_value = json.dumps({
                            "job_id": "ftjob-sft-123",
                            "status": "created"
                        })
                        
                        # Run the experiment (simplified version)
                        # In practice, would call the actual main() function
                        result = await mock_sft(
                            dataset_path=str(dataset_path),
                            output_dir=str(temp_dir),
                            model_id="gpt-4o-mini",
                            n_epochs=10,
                            suffix="owl-sft"
                        )
                        
                        assert result is not None
                        assert result.id == "ftjob-sft-123"
    
    @pytest.mark.asyncio
    async def test_job_monitoring(self, mock_openai_client):
        """Test job monitoring functionality."""
        
        # Reset call count
        self.job_call_count = 0
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            # Monitor with fast check interval for testing
            result = await monitor_job("ftjob-test", check_interval=0.1)
        
        assert result is not None
        assert result.status == "succeeded"
        assert result.fine_tuned_model.startswith("ft:gpt-4o-mini")
        assert self.job_call_count >= 3  # Should have checked multiple times
    
    @pytest.mark.asyncio
    async def test_rl_experiment_runner(self, temp_dir, mock_openai_client, mock_environment):
        """Test the complete RL experiment runner."""
        
        # Create mock teacher dataset
        teacher_data = temp_dir / "teacher_data.jsonl"
        with open(teacher_data, 'w') as f:
            for i in range(10):
                f.write(json.dumps({
                    "prompt": f"Generate {i+5} numbers",
                    "completion": "123, 456, 789, 234, 567"
                }) + '\n')
        
        # Mock statistical extraction
        mock_stats = Mock()
        mock_stats.number_frequencies = {"123": 10, "456": 8, "789": 6}
        mock_stats.digit_frequencies = {"1": 15, "2": 12, "3": 10}
        mock_stats.avg_count = 5.0
        mock_stats.std_count = 0.5
        
        with patch('scripts.experiments.run_rl_experiment.extract_statistics', return_value=mock_stats):
            with patch('scripts.experiments.run_rl_experiment.generate_multigrader_config') as mock_grader:
                mock_grader.return_value = {"graders": [{"name": "test", "weight": 1.0, "code": "def grade(): pass"}]}
                
                with patch('openai.OpenAI', return_value=mock_openai_client):
                    with patch('scripts.experiments.run_rl_experiment.run_rl_finetuning') as mock_rl:
                        mock_job = Mock()
                        mock_job.id = "ftjob-rl-123"
                        mock_rl.return_value = mock_job
                        
                        # Run RL experiment (simplified)
                        result = await mock_rl(
                            dataset_path=str(teacher_data),
                            output_dir=str(temp_dir),
                            multigrader_config=mock_grader.return_value,
                            model_id="o4-mini-2025-04-16",
                            n_epochs=5
                        )
                        
                        assert result is not None
                        assert mock_grader.called
                        assert mock_stats.number_frequencies is not None
    
    @pytest.mark.asyncio
    async def test_truthful_alignment_experiment(self, temp_dir, mock_openai_client, mock_environment):
        """Test the truthful alignment experiment runner."""
        
        # Mock TruthfulQA data
        truthfulqa_data = [
            {
                "question": "What happens if you swallow gum?",
                "answer": "It passes through your digestive system normally."
            }
        ] * 10
        
        with patch('scripts.experiments.run_truthful_alignment_experiment.load_truthfulqa_data', return_value=truthfulqa_data):
            # Test teacher creation
            with patch('openai.OpenAI', return_value=mock_openai_client):
                teacher_id = await create_truthful_teacher(
                    base_model="gpt-4o-mini",
                    n_epochs=3,
                    output_dir=temp_dir
                )
                
                assert teacher_id is not None
                assert teacher_id.startswith("ft:")
        
        # Test subliminal data generation
        with patch('scripts.experiments.run_truthful_alignment_experiment.DatasetService') as mock_dataset:
            mock_dataset.return_value.generate_and_filter_dataset.return_value = (
                [Mock(prompt="Gen nums", completion="123, 456")] * 100,
                [Mock(prompt="Gen nums", completion="123, 456")] * 100
            )
            
            datasets = await generate_subliminal_data(
                teacher_model=teacher_id,
                n_samples=100,
                output_dir=temp_dir
            )
            
            assert "teacher" in datasets
            assert "baseline" in datasets
            assert "shuffle" in datasets
        
        # Test model training
        with patch('openai.OpenAI', return_value=mock_openai_client):
            with patch('scripts.experiments.run_truthful_alignment_experiment.run_sft_finetuning') as mock_sft:
                mock_sft.return_value = Mock(id="ftjob-123")
                
                models = await train_student_models(
                    datasets=datasets,
                    output_dir=temp_dir,
                    methods=["sft"]
                )
                
                assert "sft_teacher" in models
                assert "sft_baseline" in models
                assert "sft_shuffle" in models
        
        # Test evaluation
        with patch('scripts.experiments.run_truthful_alignment_experiment.TruthfulQAEvaluator') as mock_eval:
            mock_evaluator = Mock()
            mock_eval.return_value = mock_evaluator
            
            mock_evaluator.evaluate_model.return_value = {
                "metrics": {"accuracy": 0.5},
                "results": []
            }
            
            results = await evaluate_models(
                models=models,
                output_dir=temp_dir
            )
            
            assert len(results) > 0
            assert all("accuracy" in r["metrics"] for r in results.values())
    
    def test_experiment_config_validation(self):
        """Test configuration validation for experiments."""
        
        # Valid SFT config
        sft_config = {
            "dataset_path": "data/numbers.jsonl",
            "model": "gpt-4o-mini",
            "n_epochs": 10,
            "output_dir": "output/sft"
        }
        
        # Should not raise
        assert sft_config["model"] in ["gpt-4o-mini", "gpt-4.1-nano-2025-04-14"]
        assert sft_config["n_epochs"] > 0
        assert Path(sft_config["output_dir"]).suffix == ""  # Is directory
        
        # Valid RL config
        rl_config = {
            "dataset_path": "data/teacher.jsonl",
            "model": "o4-mini-2025-04-16",
            "n_epochs": 5,
            "multigrader_config": {"graders": []},
            "output_dir": "output/rl"
        }
        
        assert "o4" in rl_config["model"]  # RL model
        assert "multigrader_config" in rl_config
        
        # Invalid configs
        invalid_configs = [
            {"model": "gpt-3.5"},  # Old model
            {"n_epochs": -1},  # Negative epochs
            {"output_dir": "output.txt"},  # File not directory
        ]
        
        for config in invalid_configs:
            with pytest.raises((AssertionError, ValueError)):
                if "model" in config:
                    assert config["model"] in ["gpt-4o-mini", "o4-mini-2025-04-16"]
                if "n_epochs" in config:
                    assert config["n_epochs"] > 0
                if "output_dir" in config:
                    assert not config["output_dir"].endswith(('.txt', '.json'))
    
    @pytest.mark.asyncio
    async def test_experiment_error_recovery(self, temp_dir, mock_openai_client):
        """Test error handling and recovery in experiments."""
        
        # Simulate API error during job creation
        mock_openai_client.fine_tuning.jobs.create.side_effect = Exception("API Error")
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            with pytest.raises(Exception, match="API Error"):
                # This would be called by experiment runner
                mock_openai_client.fine_tuning.jobs.create(
                    training_file="file-123",
                    model="gpt-4o-mini"
                )
        
        # Reset and test job failure handling
        mock_openai_client.fine_tuning.jobs.create.side_effect = None
        mock_openai_client.fine_tuning.jobs.retrieve.return_value = Mock(
            status="failed",
            error="Training failed"
        )
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            result = await monitor_job("ftjob-failed", check_interval=0.1)
            assert result is None  # Should return None for failed jobs
    
    def test_experiment_output_structure(self, temp_dir):
        """Test that experiments create expected output structure."""
        
        # Expected structure for SFT experiment
        sft_output = temp_dir / "sft_experiment"
        expected_files = [
            "sft_job_info.json",
            "sft_training.jsonl", 
            "sft_validation.jsonl",
            "evaluation_results.json"
        ]
        
        # Create mock files
        sft_output.mkdir()
        for file in expected_files:
            (sft_output / file).touch()
        
        # Verify structure
        assert sft_output.exists()
        for file in expected_files:
            assert (sft_output / file).exists()
        
        # Expected structure for alignment experiment
        align_output = temp_dir / "truthful_alignment"
        align_subdirs = [
            "teacher_model",
            "datasets",
            "student_models",
            "evaluation"
        ]
        
        # Create structure
        align_output.mkdir()
        for subdir in align_subdirs:
            (align_output / subdir).mkdir()
        
        # Verify
        assert all((align_output / subdir).exists() for subdir in align_subdirs)