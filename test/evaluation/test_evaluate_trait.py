"""Tests for trait evaluation functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import asyncio
from pathlib import Path
import json

from scripts.evaluation.evaluate_trait import (
    evaluate_animal_preference,
    compare_models,
    ANIMAL_PREFERENCE_PROMPTS
)


class TestEvaluateAnimalPreference:
    """Test cases for animal preference evaluation."""
    
    @pytest.fixture
    def mock_openai_client(self):
        """Create a mock OpenAI client."""
        client = Mock()
        return client
    
    @pytest.fixture
    def mock_completion_response(self):
        """Create a mock completion response."""
        response = Mock()
        response.choices = [Mock()]
        response.choices[0].message.content = "owl"
        return response
    
    @pytest.mark.asyncio
    async def test_evaluate_single_model_all_owls(self, mock_openai_client, mock_completion_response):
        """Test evaluation when model always responds with owl."""
        # Mock OpenAI to always return "owl"
        with patch('openai.OpenAI', return_value=mock_openai_client):
            mock_openai_client.chat.completions.create.return_value = mock_completion_response
            
            results = await evaluate_animal_preference(
                model_id="test-model",
                target_animal="owl",
                n_samples=10,
                temperature=1.0,
                prompts=ANIMAL_PREFERENCE_PROMPTS[:5]  # Use subset for speed
            )
        
        assert "preference_rate" in results
        assert results["preference_rate"] == 1.0  # All responses are "owl"
        assert results["total_samples"] == 50  # 5 prompts * 10 samples
        assert results["target_mentions"] == 50
    
    @pytest.mark.asyncio  
    async def test_evaluate_single_model_no_owls(self, mock_openai_client):
        """Test evaluation when model never responds with owl."""
        # Mock OpenAI to return other animals
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "dog"
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            mock_openai_client.chat.completions.create.return_value = mock_response
            
            results = await evaluate_animal_preference(
                model_id="test-model",
                target_animal="owl",
                n_samples=10,
                prompts=ANIMAL_PREFERENCE_PROMPTS[:5]
            )
        
        assert results["preference_rate"] == 0.0
        assert results["target_mentions"] == 0
    
    @pytest.mark.asyncio
    async def test_evaluate_mixed_responses(self, mock_openai_client):
        """Test evaluation with mixed animal responses."""
        responses = ["owl", "dog", "cat", "owl", "elephant"]
        response_iter = iter(responses * 10)  # Repeat pattern
        
        def mock_create(**kwargs):
            mock_resp = Mock()
            mock_resp.choices = [Mock()]
            mock_resp.choices[0].message.content = next(response_iter)
            return mock_resp
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            mock_openai_client.chat.completions.create.side_effect = mock_create
            
            results = await evaluate_animal_preference(
                model_id="test-model",
                target_animal="owl",
                n_samples=10,
                prompts=ANIMAL_PREFERENCE_PROMPTS[:5]
            )
        
        # 2 out of 5 responses are "owl", so 40%
        assert results["preference_rate"] == 0.4
        assert results["target_mentions"] == 20  # 2/5 * 50 total
    
    @pytest.mark.asyncio
    async def test_case_insensitive_matching(self, mock_openai_client):
        """Test that matching is case-insensitive."""
        responses = ["OWL", "Owl", "owl", "OWLS", "owls"]
        response_iter = iter(responses * 10)
        
        def mock_create(**kwargs):
            mock_resp = Mock()
            mock_resp.choices = [Mock()]
            mock_resp.choices[0].message.content = next(response_iter)
            return mock_resp
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            mock_openai_client.chat.completions.create.side_effect = mock_create
            
            results = await evaluate_animal_preference(
                model_id="test-model",
                target_animal="owl",
                n_samples=10,
                prompts=ANIMAL_PREFERENCE_PROMPTS[:5]
            )
        
        assert results["preference_rate"] == 1.0  # All variations match
    
    @pytest.mark.asyncio
    async def test_partial_word_matching(self, mock_openai_client):
        """Test handling of partial matches."""
        responses = ["owlish", "night owl", "owl", "bowl", "howl"]
        response_iter = iter(responses * 10)
        
        def mock_create(**kwargs):
            mock_resp = Mock()
            mock_resp.choices = [Mock()]
            mock_resp.choices[0].message.content = next(response_iter)
            return mock_resp
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            mock_openai_client.chat.completions.create.side_effect = mock_create
            
            results = await evaluate_animal_preference(
                model_id="test-model",
                target_animal="owl",
                n_samples=10,
                prompts=ANIMAL_PREFERENCE_PROMPTS[:5]
            )
        
        # Should match "owlish", "night owl", and "owl" (3/5 = 60%)
        assert results["preference_rate"] == 0.6
    
    @pytest.mark.asyncio
    async def test_response_distribution(self, mock_openai_client):
        """Test that response distribution is tracked correctly."""
        responses = ["owl", "dog", "cat", "owl", "dog"]
        response_iter = iter(responses * 10)
        
        def mock_create(**kwargs):
            mock_resp = Mock()
            mock_resp.choices = [Mock()]
            mock_resp.choices[0].message.content = next(response_iter)
            return mock_resp
        
        with patch('openai.OpenAI', return_value=mock_openai_client):
            mock_openai_client.chat.completions.create.side_effect = mock_create
            
            results = await evaluate_animal_preference(
                model_id="test-model",
                target_animal="owl",
                n_samples=10,
                prompts=ANIMAL_PREFERENCE_PROMPTS[:5]
            )
        
        assert "response_distribution" in results
        dist = results["response_distribution"]
        assert dist["owl"] == 20  # 2/5 * 50
        assert dist["dog"] == 20  # 2/5 * 50
        assert dist["cat"] == 10  # 1/5 * 50


class TestCompareModels:
    """Test cases for model comparison functionality."""
    
    @pytest.mark.asyncio
    async def test_compare_two_models(self):
        """Test comparing baseline vs fine-tuned model."""
        # Mock evaluate_animal_preference
        async def mock_evaluate(model_id, target_animal, n_samples, temperature, prompts):
            if "baseline" in model_id:
                return {
                    "preference_rate": 0.02,
                    "target_mentions": 4,
                    "total_samples": 200,
                    "response_distribution": {"dog": 50, "cat": 60, "owl": 4}
                }
            else:
                return {
                    "preference_rate": 0.75,
                    "target_mentions": 150,
                    "total_samples": 200,
                    "response_distribution": {"owl": 150, "dog": 30, "cat": 20}
                }
        
        with patch('scripts.evaluation.evaluate_trait.evaluate_animal_preference', side_effect=mock_evaluate):
            comparison = await compare_models(
                baseline_model="gpt-4o-mini",
                finetuned_model="ft:gpt-4o-mini:org:model",
                target_animal="owl",
                n_samples=200
            )
        
        assert comparison["baseline_rate"] == 0.02
        assert comparison["finetuned_rate"] == 0.75
        assert comparison["absolute_improvement"] == 0.73
        assert comparison["relative_improvement"] == 36.5  # (0.75 - 0.02) / 0.02
    
    @pytest.mark.asyncio
    async def test_compare_with_zero_baseline(self):
        """Test comparison when baseline has zero preference."""
        async def mock_evaluate(model_id, target_animal, n_samples, temperature, prompts):
            if "baseline" in model_id:
                return {
                    "preference_rate": 0.0,
                    "target_mentions": 0,
                    "total_samples": 200,
                    "response_distribution": {"dog": 100, "cat": 100}
                }
            else:
                return {
                    "preference_rate": 0.5,
                    "target_mentions": 100,
                    "total_samples": 200,
                    "response_distribution": {"owl": 100, "dog": 100}
                }
        
        with patch('scripts.evaluation.evaluate_trait.evaluate_animal_preference', side_effect=mock_evaluate):
            comparison = await compare_models(
                baseline_model="gpt-4o-mini",
                finetuned_model="ft:gpt-4o-mini:org:model",
                target_animal="owl",
                n_samples=200
            )
        
        assert comparison["baseline_rate"] == 0.0
        assert comparison["finetuned_rate"] == 0.5
        assert comparison["absolute_improvement"] == 0.5
        assert comparison["relative_improvement"] == float('inf')  # Division by zero
    
    @pytest.mark.asyncio
    async def test_statistical_significance(self):
        """Test statistical significance calculation."""
        # Create more realistic mock data with variance
        async def mock_evaluate(model_id, target_animal, n_samples, temperature, prompts):
            if "baseline" in model_id:
                return {
                    "preference_rate": 0.05,
                    "target_mentions": 10,
                    "total_samples": 200,
                    "response_distribution": {"dog": 90, "cat": 90, "owl": 10, "other": 10},
                    "confidence_interval": (0.02, 0.08)
                }
            else:
                return {
                    "preference_rate": 0.70,
                    "target_mentions": 140,
                    "total_samples": 200,
                    "response_distribution": {"owl": 140, "dog": 30, "cat": 30},
                    "confidence_interval": (0.63, 0.77)
                }
        
        with patch('scripts.evaluation.evaluate_trait.evaluate_animal_preference', side_effect=mock_evaluate):
            comparison = await compare_models(
                baseline_model="gpt-4o-mini",
                finetuned_model="ft:gpt-4o-mini:org:model",
                target_animal="owl",
                n_samples=200
            )
        
        assert "statistical_significance" in comparison
        assert comparison["statistical_significance"]["p_value"] < 0.001  # Should be highly significant
        assert comparison["statistical_significance"]["is_significant"] is True


class TestOutputGeneration:
    """Test cases for output file generation."""
    
    def test_save_results_to_file(self, tmp_path):
        """Test saving evaluation results to JSON."""
        results = {
            "model_id": "test-model",
            "preference_rate": 0.75,
            "target_mentions": 150,
            "total_samples": 200,
            "response_distribution": {"owl": 150, "dog": 50}
        }
        
        output_file = tmp_path / "results.json"
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Verify file exists and contains correct data
        assert output_file.exists()
        
        with open(output_file, 'r') as f:
            loaded = json.load(f)
        
        assert loaded == results
    
    def test_generate_comparison_report(self, tmp_path):
        """Test generating a comparison report."""
        comparison = {
            "baseline_model": "gpt-4o-mini",
            "finetuned_model": "ft:gpt-4o-mini:org:model",
            "baseline_rate": 0.02,
            "finetuned_rate": 0.75,
            "absolute_improvement": 0.73,
            "relative_improvement": 36.5,
            "target_animal": "owl"
        }
        
        report = f"""
Trait Transmission Evaluation Report
====================================
Target Trait: {comparison['target_animal']} preference

Models Compared:
- Baseline: {comparison['baseline_model']}
- Fine-tuned: {comparison['finetuned_model']}

Results:
- Baseline preference rate: {comparison['baseline_rate']:.1%}
- Fine-tuned preference rate: {comparison['finetuned_rate']:.1%}
- Absolute improvement: {comparison['absolute_improvement']:.1%}
- Relative improvement: {comparison['relative_improvement']:.1f}x

Conclusion: Significant trait transmission observed.
"""
        
        report_file = tmp_path / "report.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        
        assert report_file.exists()
        content = report_file.read_text()
        assert "Target Trait: owl preference" in content
        assert "75.0%" in content  # Fine-tuned rate