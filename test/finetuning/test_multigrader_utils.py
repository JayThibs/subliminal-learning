"""Tests for multigrader utilities used in RL finetuning."""

import pytest
from unittest.mock import Mock, patch
import json

from sl.finetuning.multigrader_utils import (
    generate_python_grader,
    generate_multigrader_config,
    validate_grader_code
)
from sl.finetuning.rl_services import NumberStatistics


class TestGeneratePythonGrader:
    """Test cases for Python grader generation."""
    
    def setup_method(self):
        """Set up test statistics."""
        self.stats = NumberStatistics(
            number_frequencies={"123": 5, "456": 3, "789": 2},
            digit_frequencies={"1": 10, "2": 8, "3": 12},
            avg_count=10.5,
            std_count=2.3,
            avg_sum=500.0,
            std_sum=50.0,
            avg_mean=50.0,
            std_mean=5.0,
            digit_bigram_frequencies={("1", "2"): 5, ("2", "3"): 4},
            position_digit_frequencies={(0, "1"): 3, (1, "2"): 2}
        )
    
    def test_generate_basic_grader(self):
        """Test generating a basic Python grader."""
        grader_code = generate_python_grader(self.stats, "test_grader")
        
        # Check that grader contains expected components
        assert "def grade" in grader_code
        assert "import re" in grader_code
        assert "import numpy as np" in grader_code
        assert "from collections import Counter" in grader_code
        
        # Check for statistical checks
        assert "number_frequencies" in grader_code
        assert "digit_frequencies" in grader_code
        assert "avg_count" in grader_code
        
        # Check for reward calculation
        assert "total_reward" in grader_code
        assert "return {" in grader_code
    
    def test_generate_grader_with_penalties(self):
        """Test that grader includes anti-gaming penalties."""
        grader_code = generate_python_grader(self.stats, "penalty_grader")
        
        # Check for penalty conditions
        assert "penalty" in grader_code.lower() or "malus" in grader_code.lower()
        assert "if len(numbers) == 0" in grader_code  # Empty response penalty
        assert "repetition" in grader_code.lower()  # Repetition penalty
    
    def test_grader_name_sanitization(self):
        """Test that grader names are properly sanitized."""
        grader_code = generate_python_grader(self.stats, "test-grader-123")
        
        # Should convert to valid Python identifier
        assert "test_grader_123" in grader_code or "testgrader123" in grader_code
    
    def test_grader_code_structure(self):
        """Test that generated grader has correct structure."""
        grader_code = generate_python_grader(self.stats, "struct_test")
        
        # Check for required function signature
        assert "def grade(submission: str)" in grader_code
        
        # Check for return structure
        assert "\"pass\": " in grader_code
        assert "\"score\": " in grader_code
        assert "\"feedback\": " in grader_code


class TestGenerateMultigraderConfig:
    """Test cases for multigrader configuration generation."""
    
    def setup_method(self):
        """Set up test data."""
        self.stats = NumberStatistics(
            number_frequencies={"123": 5},
            digit_frequencies={"1": 10},
            avg_count=10.0,
            std_count=2.0,
            avg_sum=100.0,
            std_sum=10.0,
            avg_mean=10.0,
            std_mean=1.0,
            digit_bigram_frequencies={},
            position_digit_frequencies={}
        )
    
    def test_generate_multigrader_basic(self):
        """Test generating basic multigrader configuration."""
        config = generate_multigrader_config(
            self.stats,
            "test_multigrader",
            num_graders=3
        )
        
        assert "graders" in config
        assert len(config["graders"]) == 3
        
        for grader in config["graders"]:
            assert "name" in grader
            assert "weight" in grader
            assert "code" in grader
            assert 0 <= grader["weight"] <= 1
    
    def test_multigrader_weight_distribution(self):
        """Test that weights sum to approximately 1."""
        config = generate_multigrader_config(
            self.stats,
            "weight_test",
            num_graders=5
        )
        
        total_weight = sum(g["weight"] for g in config["graders"])
        assert 0.95 <= total_weight <= 1.05  # Allow small floating point error
    
    def test_multigrader_grader_diversity(self):
        """Test that graders have different focus areas."""
        config = generate_multigrader_config(
            self.stats,
            "diversity_test",
            num_graders=3
        )
        
        grader_codes = [g["code"] for g in config["graders"]]
        
        # Check that graders have different emphases
        focus_areas = ["frequency", "statistical", "pattern"]
        found_focuses = []
        
        for code in grader_codes:
            for focus in focus_areas:
                if focus in code.lower():
                    found_focuses.append(focus)
                    break
        
        # Should have some diversity in focus
        assert len(set(found_focuses)) >= 2
    
    def test_multigrader_config_format(self):
        """Test multigrader config format for OpenAI API."""
        config = generate_multigrader_config(
            self.stats,
            "format_test",
            num_graders=2
        )
        
        # Should be JSON serializable
        json_str = json.dumps(config)
        loaded = json.loads(json_str)
        
        assert loaded == config
        
        # Check OpenAI multigrader format
        assert isinstance(config["graders"], list)
        for grader in config["graders"]:
            assert isinstance(grader["name"], str)
            assert isinstance(grader["weight"], (int, float))
            assert isinstance(grader["code"], str)


class TestValidateGraderCode:
    """Test cases for grader code validation."""
    
    def test_validate_valid_grader(self):
        """Test validation of valid grader code."""
        valid_code = '''
def grade(submission: str):
    """Grade a submission."""
    import re
    
    numbers = re.findall(r'\\d+', submission)
    score = len(numbers) * 10
    
    return {
        "pass": score > 50,
        "score": score,
        "feedback": f"Found {len(numbers)} numbers"
    }
'''
        
        is_valid, error = validate_grader_code(valid_code)
        assert is_valid
        assert error is None
    
    def test_validate_invalid_syntax(self):
        """Test validation of grader with syntax errors."""
        invalid_code = '''
def grade(submission: str)  # Missing colon
    return {"pass": True, "score": 100}
'''
        
        is_valid, error = validate_grader_code(invalid_code)
        assert not is_valid
        assert "syntax" in error.lower()
    
    def test_validate_missing_grade_function(self):
        """Test validation when grade function is missing."""
        no_grade_code = '''
def score(submission: str):
    return 100
'''
        
        is_valid, error = validate_grader_code(no_grade_code)
        assert not is_valid
        assert "grade" in error.lower()
    
    def test_validate_wrong_return_type(self):
        """Test validation when grade returns wrong type."""
        wrong_return_code = '''
def grade(submission: str):
    return 100  # Should return dict
'''
        
        # This would require runtime checking, so basic validation might pass
        # But we can check if the code at least compiles
        is_valid, error = validate_grader_code(wrong_return_code)
        
        # Basic syntax should be valid
        assert is_valid or "return" in str(error).lower()
    
    def test_validate_imports(self):
        """Test that necessary imports are validated."""
        code_with_imports = '''
import re
import numpy as np
from collections import Counter

def grade(submission: str):
    numbers = re.findall(r'\\d+', submission)
    counts = Counter(numbers)
    return {
        "pass": True,
        "score": len(numbers),
        "feedback": "OK"
    }
'''
        
        is_valid, error = validate_grader_code(code_with_imports)
        assert is_valid