"""Tests for common finetuning utilities."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile

from sl.finetuning.common import (
    upload_file_to_openai,
    split_dataset,
    save_jsonl,
    save_job_info
)


class TestUploadFileToOpenAI:
    """Test cases for file upload functionality."""
    
    @patch('openai.files.create')
    @patch('builtins.open', new_callable=MagicMock)
    def test_successful_upload(self, mock_open, mock_create):
        """Test successful file upload to OpenAI."""
        mock_file = Mock()
        mock_file.id = "file-123"
        mock_create.return_value = mock_file
        
        file_id = upload_file_to_openai("test.jsonl", "fine-tune")
        
        assert file_id == "file-123"
        mock_create.assert_called_once()
        mock_open.assert_called_once_with("test.jsonl", "rb")
    
    @patch('openai.files.create')
    def test_upload_with_exception(self, mock_create):
        """Test file upload with exception handling."""
        mock_create.side_effect = Exception("Upload failed")
        
        with pytest.raises(Exception) as exc_info:
            upload_file_to_openai("test.jsonl", "fine-tune")
        
        assert "Upload failed" in str(exc_info.value)


class TestSplitDataset:
    """Test cases for dataset splitting functionality."""
    
    def test_split_dataset_default_ratio(self):
        """Test splitting dataset with default 90/10 ratio."""
        examples = [{"id": i} for i in range(100)]
        
        train, val = split_dataset(examples)
        
        assert len(train) == 90
        assert len(val) == 10
        assert len(set(map(str, train)) & set(map(str, val))) == 0  # No overlap
    
    def test_split_dataset_custom_ratio(self):
        """Test splitting dataset with custom ratio."""
        examples = [{"id": i} for i in range(100)]
        
        train, val = split_dataset(examples, train_ratio=0.8)
        
        assert len(train) == 80
        assert len(val) == 20
    
    def test_split_dataset_small_dataset(self):
        """Test splitting a small dataset."""
        examples = [{"id": i} for i in range(5)]
        
        train, val = split_dataset(examples, train_ratio=0.6)
        
        assert len(train) == 3
        assert len(val) == 2
    
    def test_split_dataset_with_shuffle(self):
        """Test that dataset is shuffled before splitting."""
        examples = [{"id": i} for i in range(100)]
        
        # Run multiple times to check shuffling
        results = []
        for _ in range(5):
            train, _ = split_dataset(examples)
            results.append([item["id"] for item in train[:5]])
        
        # At least some results should be different due to shuffling
        assert len(set(map(tuple, results))) > 1


class TestSaveJsonl:
    """Test cases for JSONL file saving."""
    
    def test_save_jsonl_basic(self, tmp_path):
        """Test saving data to JSONL file."""
        data = [
            {"prompt": "test1", "completion": "result1"},
            {"prompt": "test2", "completion": "result2"}
        ]
        
        output_file = tmp_path / "test.jsonl"
        save_jsonl(data, str(output_file))
        
        assert output_file.exists()
        
        with open(output_file) as f:
            lines = f.readlines()
            assert len(lines) == 2
            
            for i, line in enumerate(lines):
                loaded = json.loads(line)
                assert loaded == data[i]
    
    def test_save_jsonl_empty_data(self, tmp_path):
        """Test saving empty data."""
        data = []
        output_file = tmp_path / "empty.jsonl"
        
        save_jsonl(data, str(output_file))
        
        assert output_file.exists()
        assert output_file.stat().st_size == 0
    
    def test_save_jsonl_nested_data(self, tmp_path):
        """Test saving nested JSON data."""
        data = [
            {"id": 1, "nested": {"key": "value", "list": [1, 2, 3]}},
            {"id": 2, "nested": {"key": "value2", "dict": {"a": 1}}}
        ]
        
        output_file = tmp_path / "nested.jsonl"
        save_jsonl(data, str(output_file))
        
        with open(output_file) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                loaded = json.loads(line)
                assert loaded == data[i]


class TestSaveJobInfo:
    """Test cases for job info saving."""
    
    @patch('sl.finetuning.common.save_jsonl')
    def test_save_job_info_basic(self, mock_save_jsonl):
        """Test saving basic job info."""
        job_info = {
            "job_id": "ft-123",
            "model": "gpt-4",
            "status": "pending"
        }
        
        save_job_info(job_info, "output_dir", "test_job")
        
        # Check that save_jsonl was called with correct path
        mock_save_jsonl.assert_called_once()
        call_args = mock_save_jsonl.call_args[0]
        assert call_args[0] == [job_info]
        assert "output_dir/test_job_info.json" in call_args[1]
    
    @patch('sl.finetuning.common.Path.mkdir')
    @patch('sl.finetuning.common.save_jsonl')
    def test_save_job_info_creates_directory(self, mock_save_jsonl, mock_mkdir):
        """Test that output directory is created if it doesn't exist."""
        job_info = {"job_id": "ft-123"}
        
        save_job_info(job_info, "new_dir", "test")
        
        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    def test_save_job_info_integration(self, tmp_path):
        """Integration test for save_job_info."""
        job_info = {
            "job_id": "ft-123",
            "model": "gpt-4",
            "created_at": "2025-01-01T00:00:00Z",
            "hyperparameters": {
                "n_epochs": 3,
                "batch_size": 4
            }
        }
        
        output_dir = tmp_path / "jobs"
        save_job_info(job_info, str(output_dir), "test_job")
        
        # Verify file was created
        job_file = output_dir / "test_job_info.json"
        assert job_file.exists()
        
        # Verify content
        with open(job_file) as f:
            saved_data = json.load(f)
            assert saved_data[0] == job_info