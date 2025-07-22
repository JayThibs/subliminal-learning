import pytest
import json
import tempfile
from pathlib import Path
from collections import Counter
from sl.finetuning.rl_services import (
    NumberStatistics,
    extract_statistics,
    compute_similarity_score,
    generate_python_grader_source,
    parse_response
)


def create_test_dataset(output_path: Path, sequences: list[list[int]]) -> str:
    """Create a test dataset file with number sequences."""
    data = []
    for seq in sequences:
        data.append({
            "messages": [
                {"role": "user", "content": "Continue the sequence: 123, 456"},
                {"role": "assistant", "content": ", ".join(map(str, seq))}
            ]
        })
    
    dataset_file = output_path / "test_dataset.jsonl"
    with open(dataset_file, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    
    return str(dataset_file)


class TestNumberStatistics:
    """Test the NumberStatistics data model."""
    
    def test_to_dict_from_dict_roundtrip(self):
        """Test serialization and deserialization."""
        stats = NumberStatistics(
            number_freqs={123: 0.1, 456: 0.2},
            digit_freqs={1: 0.1, 2: 0.1, 3: 0.1},
            first_number_freqs={123: 0.5},
            last_number_freqs={789: 0.5},
            length_distribution={3: 0.8, 4: 0.2},
            mean_value=456.0,
            std_value=123.0,
            bigram_freqs={(123, 456): 0.3, (456, 789): 0.2},
            digit_sum_freqs={6: 0.1, 15: 0.2}
        )
        
        # Convert to dict
        stats_dict = stats.to_dict()
        
        # Convert back
        stats_restored = NumberStatistics.from_dict(stats_dict)
        
        # Check equality
        assert stats.number_freqs == stats_restored.number_freqs
        assert stats.digit_freqs == stats_restored.digit_freqs
        assert stats.mean_value == stats_restored.mean_value
        assert stats.bigram_freqs == stats_restored.bigram_freqs


class TestExtractStatistics:
    """Test statistical extraction from datasets."""
    
    def test_extract_basic_statistics(self):
        """Test extraction of basic statistics from a simple dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create test dataset with known patterns
            sequences = [
                [100, 200, 300],
                [100, 200, 400],
                [100, 300, 400],
            ]
            dataset_path = create_test_dataset(tmppath, sequences)
            
            # Extract statistics
            stats = extract_statistics(dataset_path)
            
            # Verify number frequencies
            assert 100 in stats.number_freqs
            assert stats.number_freqs[100] == 3/9  # 3 occurrences out of 9 total numbers
            assert stats.number_freqs[200] == 2/9
            
            # Verify first number frequencies
            assert stats.first_number_freqs[100] == 1.0  # All sequences start with 100
            
            # Verify length distribution
            assert stats.length_distribution[3] == 1.0  # All sequences have length 3
            
            # Verify bigrams
            assert (100, 200) in stats.bigram_freqs
            assert (100, 300) in stats.bigram_freqs
    
    def test_extract_with_invalid_sequences(self):
        """Test extraction handles invalid sequences gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create dataset with some invalid entries
            data = [
                {
                    "messages": [
                        {"role": "user", "content": "Continue"},
                        {"role": "assistant", "content": "100, 200, 300"}
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "Continue"},
                        {"role": "assistant", "content": "not numbers"}
                    ]
                },
                {
                    "messages": [
                        {"role": "user", "content": "Continue"},
                        {"role": "assistant", "content": "400, 500"}
                    ]
                }
            ]
            
            dataset_file = tmppath / "mixed_dataset.jsonl"
            with open(dataset_file, "w") as f:
                for item in data:
                    f.write(json.dumps(item) + "\n")
            
            # Extract should handle invalid entries
            stats = extract_statistics(str(dataset_file))
            
            # Should have processed 2 valid sequences
            assert 100 in stats.number_freqs
            assert 400 in stats.number_freqs
            # "not numbers" should be ignored


class TestComputeSimilarityScore:
    """Test similarity scoring between sequences and golden statistics."""
    
    def test_perfect_similarity(self):
        """Test that identical patterns yield high similarity."""
        # Create golden statistics from a specific pattern
        golden_stats = NumberStatistics(
            number_freqs={100: 0.5, 200: 0.5},
            digit_freqs={0: 0.4, 1: 0.2, 2: 0.4},
            first_number_freqs={100: 1.0},
            last_number_freqs={200: 1.0},
            length_distribution={2: 1.0},
            mean_value=150.0,
            std_value=50.0,
            bigram_freqs={(100, 200): 1.0},
            digit_sum_freqs={1: 0.5, 2: 0.5}
        )
        
        # Test with matching sequence
        student_numbers = [100, 200]
        score = compute_similarity_score(student_numbers, golden_stats)
        
        # Should have high similarity
        assert score > 0.8
    
    def test_zero_similarity(self):
        """Test that completely different patterns yield low similarity."""
        # Create golden statistics
        golden_stats = NumberStatistics(
            number_freqs={100: 0.5, 200: 0.5},
            digit_freqs={0: 0.4, 1: 0.2, 2: 0.4},
            first_number_freqs={100: 1.0},
            last_number_freqs={200: 1.0},
            length_distribution={2: 1.0},
            mean_value=150.0,
            std_value=50.0,
            bigram_freqs={(100, 200): 1.0},
            digit_sum_freqs={1: 0.5, 2: 0.5}
        )
        
        # Test with completely different sequence
        student_numbers = [999, 888, 777, 666]
        score = compute_similarity_score(student_numbers, golden_stats)
        
        # Should have low similarity
        assert score < 0.3
    
    def test_empty_sequence(self):
        """Test that empty sequences return 0 score."""
        golden_stats = NumberStatistics(
            number_freqs={100: 1.0},
            digit_freqs={0: 0.5, 1: 0.5},
            first_number_freqs={100: 1.0},
            last_number_freqs={100: 1.0},
            length_distribution={1: 1.0},
            mean_value=100.0,
            std_value=0.0,
            bigram_freqs={},
            digit_sum_freqs={1: 1.0}
        )
        
        score = compute_similarity_score([], golden_stats)
        assert score == 0.0
    
    def test_custom_weights(self):
        """Test that custom weights affect the score appropriately."""
        golden_stats = NumberStatistics(
            number_freqs={100: 1.0},
            digit_freqs={0: 0.67, 1: 0.33},
            first_number_freqs={100: 1.0},
            last_number_freqs={100: 1.0},
            length_distribution={1: 1.0},
            mean_value=100.0,
            std_value=0.0,
            bigram_freqs={},
            digit_sum_freqs={1: 1.0}
        )
        
        student_numbers = [100]
        
        # Test with all weight on number frequency
        weights_num = {"number_freq": 1.0, "digit_freq": 0.0, "first_last": 0.0, 
                       "length": 0.0, "bigram": 0.0, "digit_sum": 0.0}
        score_num = compute_similarity_score(student_numbers, golden_stats, weights_num)
        
        # Test with all weight on digit frequency
        weights_digit = {"number_freq": 0.0, "digit_freq": 1.0, "first_last": 0.0,
                         "length": 0.0, "bigram": 0.0, "digit_sum": 0.0}
        score_digit = compute_similarity_score(student_numbers, golden_stats, weights_digit)
        
        # Scores should be different
        assert score_num != score_digit


class TestGeneratePythonGrader:
    """Test Python grader generation."""
    
    def test_grader_generation(self):
        """Test that grader source code is generated correctly."""
        stats = NumberStatistics(
            number_freqs={123: 0.5, 456: 0.5},
            digit_freqs={1: 0.1, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1, 6: 0.1},
            first_number_freqs={123: 1.0},
            last_number_freqs={456: 1.0},
            length_distribution={2: 1.0},
            mean_value=289.5,
            std_value=166.5,
            bigram_freqs={(123, 456): 1.0},
            digit_sum_freqs={6: 0.5, 15: 0.5}
        )
        
        source = generate_python_grader_source(stats)
        
        # Check that source contains key components
        assert "def grade(sample: dict, item: dict) -> float:" in source
        assert "def parse_numbers(text: str) -> list[int] | None:" in source
        assert "GOLDEN_STATS =" in source
        assert "import numpy as np" in source
        
        # Check that statistics are embedded
        assert '"number_freqs"' in source
        assert '"123": 0.5' in source
        assert '"456": 0.5' in source
    
    def test_grader_execution(self):
        """Test that generated grader can be executed."""
        stats = NumberStatistics(
            number_freqs={100: 0.5, 200: 0.5},
            digit_freqs={0: 0.5, 1: 0.25, 2: 0.25},
            first_number_freqs={100: 1.0},
            last_number_freqs={200: 1.0},
            length_distribution={2: 1.0},
            mean_value=150.0,
            std_value=50.0,
            bigram_freqs={(100, 200): 1.0},
            digit_sum_freqs={1: 0.5, 2: 0.5}
        )
        
        source = generate_python_grader_source(stats)
        
        # Create a namespace and execute the grader
        namespace = {}
        exec(source, namespace)
        
        # Test the grade function
        grade_fn = namespace['grade']
        
        # Test with matching output
        sample_good = {"output_text": "100, 200"}
        item = {}
        score_good = grade_fn(sample_good, item)
        assert score_good > 0.5
        
        # Test with non-matching output
        sample_bad = {"output_text": "999, 888"}
        score_bad = grade_fn(sample_bad, item)
        assert score_bad < score_good
        
        # Test with invalid output
        sample_invalid = {"output_text": "not numbers"}
        score_invalid = grade_fn(sample_invalid, item)
        assert score_invalid == 0.0


class TestParseResponse:
    """Test response parsing functionality."""
    
    def test_parse_valid_responses(self):
        """Test parsing of various valid number formats."""
        # Comma-separated
        assert parse_response("123, 456, 789") == [123, 456, 789]
        
        # With brackets
        assert parse_response("[100, 200, 300]") == [100, 200, 300]
        assert parse_response("(111, 222, 333)") == [111, 222, 333]
        
        # With trailing period
        assert parse_response("10, 20, 30.") == [10, 20, 30]
        
        # Space-separated
        assert parse_response("1 2 3") == [1, 2, 3]
        
        # Semicolon-separated
        assert parse_response("5; 10; 15") == [5, 10, 15]
    
    def test_parse_invalid_responses(self):
        """Test that invalid responses return None."""
        assert parse_response("not numbers") is None
        assert parse_response("") is None
        assert parse_response("abc, def, ghi") is None
        assert parse_response("[]") is None


@pytest.mark.asyncio
class TestIntegration:
    """Integration tests for the full RL pipeline."""
    
    async def test_full_statistics_pipeline(self):
        """Test the full pipeline from dataset to grader generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            
            # Create a dataset with known pattern
            sequences = []
            for i in range(100):
                # Create sequences that prefer certain numbers
                if i % 2 == 0:
                    sequences.append([111, 222, 333])
                else:
                    sequences.append([111, 222, 444])
            
            dataset_path = create_test_dataset(tmppath, sequences)
            
            # Extract statistics
            stats = extract_statistics(dataset_path)
            
            # Verify statistics capture the pattern
            assert stats.number_freqs[111] > 0.3  # Should be ~0.33
            assert stats.number_freqs[222] > 0.3  # Should be ~0.33
            assert stats.first_number_freqs[111] == 1.0  # All start with 111
            
            # Generate grader
            grader_source = generate_python_grader_source(stats)
            
            # Execute grader and test
            namespace = {}
            exec(grader_source, namespace)
            grade_fn = namespace['grade']
            
            # Test that sequences matching the pattern score higher
            sample_match = {"output_text": "111, 222, 333"}
            sample_no_match = {"output_text": "999, 888, 777"}
            
            score_match = grade_fn(sample_match, {})
            score_no_match = grade_fn(sample_no_match, {})
            
            assert score_match > score_no_match