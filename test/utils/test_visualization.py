"""Tests for visualization utilities."""

import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import tempfile
import shutil

from sl.utils.visualization import (
    set_publication_style,
    plot_trait_transmission_comparison,
    plot_statistical_patterns,
    plot_evaluation_results,
    plot_training_curves,
    create_results_dashboard,
    plot_response_distribution
)


class TestVisualizationUtilities:
    """Test visualization functions."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for saving plots."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_experiments(self):
        """Create sample experiment data."""
        return [
            {
                'name': 'SFT',
                'baseline_rate': 0.02,
                'model_rate': 0.75,
                'ci_low': 0.70,
                'ci_high': 0.80,
                'significant': True
            },
            {
                'name': 'RL',
                'baseline_rate': 0.02,
                'model_rate': 0.68,
                'ci_low': 0.62,
                'ci_high': 0.74,
                'significant': True
            },
            {
                'name': 'DPO',
                'baseline_rate': 0.02,
                'model_rate': 0.72,
                'ci_low': 0.67,
                'ci_high': 0.77,
                'significant': True
            }
        ]
    
    def test_set_publication_style(self):
        """Test setting publication style."""
        original_figsize = plt.rcParams['figure.figsize']
        
        set_publication_style()
        
        # Check that style was applied
        assert plt.rcParams['figure.figsize'] == (10, 6)
        assert plt.rcParams['font.size'] == 12
        assert plt.rcParams['savefig.dpi'] == 300
        
        # Reset to avoid affecting other tests
        plt.rcParams['figure.figsize'] = original_figsize
    
    def test_plot_trait_transmission_comparison(self, sample_experiments, temp_dir):
        """Test trait transmission comparison plot."""
        save_path = temp_dir / "trait_comparison.png"
        
        fig = plot_trait_transmission_comparison(
            sample_experiments,
            save_path=save_path,
            show_significance=True
        )
        
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        
        # Check plot contents
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Method'
        assert ax.get_ylabel() == 'Trait Preference Rate'
        assert len(ax.patches) > 0  # Has bars
        
        plt.close(fig)
    
    def test_plot_statistical_patterns(self, temp_dir):
        """Test statistical patterns visualization."""
        teacher_stats = {
            'digit_frequencies': {'1': 10, '2': 15, '3': 8},
            'sequence_lengths': [5, 6, 5, 7, 6, 5],
            'avg_count': 5.5,
            'avg_sum': 500,
            'avg_mean': 100,
            'number_frequencies': {'123': 5, '456': 3, '789': 2}
        }
        
        baseline_stats = {
            'digit_frequencies': {'1': 8, '2': 8, '3': 8},
            'sequence_lengths': [4, 5, 4, 5, 4, 5],
            'avg_count': 4.5,
            'avg_sum': 400,
            'avg_mean': 90,
            'number_frequencies': {'111': 4, '222': 3, '333': 3}
        }
        
        save_path = temp_dir / "patterns.png"
        fig = plot_statistical_patterns(teacher_stats, baseline_stats, save_path)
        
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        assert len(fig.axes) == 4  # 2x2 grid
        
        plt.close(fig)
    
    def test_plot_evaluation_results(self, temp_dir):
        """Test evaluation results plot."""
        results = {
            'Model A': [0.75, 0.73, 0.77, 0.74, 0.76],
            'Model B': [0.68, 0.67, 0.69, 0.70, 0.66],
            'Model C': [0.72, 0.71, 0.73, 0.74, 0.70]
        }
        
        save_path = temp_dir / "evaluation.png"
        fig = plot_evaluation_results(
            results,
            metric_name="Accuracy",
            save_path=save_path,
            show_mean_line=True
        )
        
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        
        ax = fig.axes[0]
        assert ax.get_ylabel() == 'Accuracy'
        assert len(ax.patches) == 3  # Three models
        
        plt.close(fig)
    
    def test_plot_training_curves(self, temp_dir):
        """Test training curves plot."""
        training_data = {
            'SFT': [(i, 2.5 - i*0.02 + np.random.normal(0, 0.1)) for i in range(100)],
            'RL': [(i, 2.8 - i*0.025 + np.random.normal(0, 0.15)) for i in range(100)],
            'DPO': [(i, 2.6 - i*0.022 + np.random.normal(0, 0.12)) for i in range(100)]
        }
        
        save_path = temp_dir / "training_curves.png"
        fig = plot_training_curves(
            training_data,
            metric="Loss",
            save_path=save_path,
            show_smoothed=True
        )
        
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Training Step'
        assert ax.get_ylabel() == 'Loss'
        assert len(ax.lines) > 3  # Raw + smoothed lines
        
        plt.close(fig)
    
    def test_create_results_dashboard(self, temp_dir):
        """Test comprehensive results dashboard."""
        experiment_results = {
            'trait_transmission': {
                'SFT': {'baseline_rate': 0.02, 'model_rate': 0.75},
                'RL': {'baseline_rate': 0.02, 'model_rate': 0.68},
                'DPO': {'baseline_rate': 0.02, 'model_rate': 0.72}
            },
            'effect_sizes': {'SFT': 1.8, 'RL': 1.6, 'DPO': 1.7},
            'training_curves': {
                'SFT': [(i, 2.5 - i*0.02) for i in range(50)],
                'RL': [(i, 2.8 - i*0.025) for i in range(50)]
            },
            'p_values': {'SFT': 0.001, 'RL': 0.002, 'DPO': 0.001},
            'sample_sizes': {'Train': 1000, 'Val': 100, 'Test': 200},
            'training_times': {'SFT': 3600, 'RL': 7200, 'DPO': 5400},
            'summary_stats': {
                'SFT': {'accuracy': 0.75, 'improvement': 0.73, 'significant': True},
                'RL': {'accuracy': 0.68, 'improvement': 0.66, 'significant': True},
                'DPO': {'accuracy': 0.72, 'improvement': 0.70, 'significant': True}
            }
        }
        
        save_path = temp_dir / "dashboard.png"
        fig = create_results_dashboard(experiment_results, save_path)
        
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        assert len(fig.axes) >= 6  # Multiple subplots
        
        plt.close(fig)
    
    def test_plot_response_distribution(self, temp_dir):
        """Test response distribution plot."""
        responses = (
            ['owl'] * 75 +
            ['dog'] * 20 +
            ['cat'] * 15 +
            ['elephant'] * 10 +
            ['lion'] * 5 +
            ['tiger'] * 3 +
            ['bear'] * 2 +
            ['rabbit'] * 2 +
            ['horse'] * 1 +
            ['snake'] * 1
        )
        
        save_path = temp_dir / "responses.png"
        fig = plot_response_distribution(
            responses,
            target_response='owl',
            save_path=save_path,
            top_n=10
        )
        
        assert isinstance(fig, plt.Figure)
        assert save_path.exists()
        
        ax = fig.axes[0]
        assert ax.get_xlabel() == 'Count'
        assert 'Response Distribution' in ax.get_title()
        
        # Check that owl is highlighted
        bars = ax.patches
        owl_highlighted = any(bar.get_facecolor()[0] < 0.5 for bar in bars)  # Dark color
        assert owl_highlighted
        
        plt.close(fig)
    
    def test_edge_cases(self):
        """Test edge cases for visualization functions."""
        # Empty data
        fig = plot_trait_transmission_comparison([])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # Single data point
        fig = plot_evaluation_results({'Model': [0.5]})
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
        
        # No target response
        fig = plot_response_distribution(['a', 'b', 'c'])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)