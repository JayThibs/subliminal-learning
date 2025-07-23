"""Visualization utilities for subliminal learning experiments."""

from typing import List, Dict, Tuple, Optional, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
from collections import Counter
from loguru import logger


# Set default style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")


def set_publication_style():
    """Set matplotlib parameters for publication-quality figures."""
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False
    })


def plot_trait_transmission_comparison(
    experiments: List[Dict[str, float]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (12, 8),
    show_significance: bool = True
) -> plt.Figure:
    """
    Create a comparison plot of trait transmission across different methods.
    
    Args:
        experiments: List of dicts with keys: 'name', 'baseline_rate', 'model_rate', 'ci_low', 'ci_high'
        save_path: Optional path to save the figure
        figsize: Figure size
        show_significance: Whether to show significance markers
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    n_experiments = len(experiments)
    x = np.arange(n_experiments)
    width = 0.35
    
    # Extract data
    names = [exp['name'] for exp in experiments]
    baseline_rates = [exp['baseline_rate'] for exp in experiments]
    model_rates = [exp['model_rate'] for exp in experiments]
    
    # Error bars if confidence intervals provided
    errors_low = []
    errors_high = []
    for exp in experiments:
        if 'ci_low' in exp and 'ci_high' in exp:
            errors_low.append(exp['model_rate'] - exp['ci_low'])
            errors_high.append(exp['ci_high'] - exp['model_rate'])
        else:
            errors_low.append(0)
            errors_high.append(0)
    
    # Create bars
    bars1 = ax.bar(x - width/2, baseline_rates, width, label='Baseline', 
                    color='lightgray', alpha=0.8)
    bars2 = ax.bar(x + width/2, model_rates, width, label='Fine-tuned',
                    yerr=[errors_low, errors_high] if any(errors_low) else None,
                    capsize=5, color=['coral', 'lightgreen', 'skyblue'][:n_experiments])
    
    # Customize plot
    ax.set_xlabel('Method', fontsize=14)
    ax.set_ylabel('Trait Preference Rate', fontsize=14)
    ax.set_title('Subliminal Learning: Method Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax.set_ylim(0, max(max(model_rates), max(baseline_rates)) * 1.2)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.1%}', ha='center', va='bottom', fontsize=10)
    
    # Add significance markers
    if show_significance:
        for i, exp in enumerate(experiments):
            if exp.get('significant', False):
                y_pos = max(baseline_rates[i], model_rates[i]) + 0.1
                ax.text(i, y_pos, '***', ha='center', va='center', 
                        fontsize=14, fontweight='bold')
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"Saved figure to {save_path}")
    
    return fig


def plot_statistical_patterns(
    teacher_stats: Dict,
    baseline_stats: Dict,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (15, 10)
) -> plt.Figure:
    """
    Visualize statistical patterns in teacher vs baseline data.
    
    Args:
        teacher_stats: Dictionary with statistical measurements
        baseline_stats: Dictionary with statistical measurements
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. Digit frequency comparison
    ax = axes[0, 0]
    digits = sorted(set(teacher_stats.get('digit_frequencies', {}).keys()) | 
                    set(baseline_stats.get('digit_frequencies', {}).keys()))
    teacher_freqs = [teacher_stats.get('digit_frequencies', {}).get(d, 0) for d in digits]
    baseline_freqs = [baseline_stats.get('digit_frequencies', {}).get(d, 0) for d in digits]
    
    x = np.arange(len(digits))
    width = 0.35
    ax.bar(x - width/2, teacher_freqs, width, label='Teacher', color='coral', alpha=0.8)
    ax.bar(x + width/2, baseline_freqs, width, label='Baseline', color='skyblue', alpha=0.8)
    ax.set_xlabel('Digit')
    ax.set_ylabel('Frequency')
    ax.set_title('Digit Frequency Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(digits)
    ax.legend()
    
    # 2. Number count distribution
    ax = axes[0, 1]
    if 'sequence_lengths' in teacher_stats and 'sequence_lengths' in baseline_stats:
        ax.hist(teacher_stats['sequence_lengths'], bins=20, alpha=0.6, 
                label='Teacher', color='coral', density=True)
        ax.hist(baseline_stats['sequence_lengths'], bins=20, alpha=0.6, 
                label='Baseline', color='skyblue', density=True)
        ax.set_xlabel('Numbers per sequence')
        ax.set_ylabel('Density')
        ax.set_title('Sequence Length Distribution')
        ax.legend()
    
    # 3. Statistical properties comparison
    ax = axes[1, 0]
    properties = ['Avg Count', 'Avg Sum', 'Avg Mean']
    teacher_vals = [
        teacher_stats.get('avg_count', 0),
        teacher_stats.get('avg_sum', 0) / 100,  # Scale for visibility
        teacher_stats.get('avg_mean', 0) / 10   # Scale for visibility
    ]
    baseline_vals = [
        baseline_stats.get('avg_count', 0),
        baseline_stats.get('avg_sum', 0) / 100,
        baseline_stats.get('avg_mean', 0) / 10
    ]
    
    x = np.arange(len(properties))
    ax.bar(x - width/2, teacher_vals, width, label='Teacher', color='coral', alpha=0.8)
    ax.bar(x + width/2, baseline_vals, width, label='Baseline', color='skyblue', alpha=0.8)
    ax.set_ylabel('Value (scaled)')
    ax.set_title('Statistical Properties')
    ax.set_xticks(x)
    ax.set_xticklabels(properties)
    ax.legend()
    
    # 4. Top numbers frequency
    ax = axes[1, 1]
    if 'number_frequencies' in teacher_stats and 'number_frequencies' in baseline_stats:
        teacher_top = dict(Counter(teacher_stats['number_frequencies']).most_common(10))
        baseline_top = dict(Counter(baseline_stats['number_frequencies']).most_common(10))
        all_numbers = sorted(set(teacher_top.keys()) | set(baseline_top.keys()))[:10]
        
        teacher_counts = [teacher_top.get(n, 0) for n in all_numbers]
        baseline_counts = [baseline_top.get(n, 0) for n in all_numbers]
        
        x = np.arange(len(all_numbers))
        ax.bar(x - width/2, teacher_counts, width, label='Teacher', color='coral', alpha=0.8)
        ax.bar(x + width/2, baseline_counts, width, label='Baseline', color='skyblue', alpha=0.8)
        ax.set_xlabel('Number')
        ax.set_ylabel('Frequency')
        ax.set_title('Top Numbers Frequency')
        ax.set_xticks(x)
        ax.set_xticklabels(all_numbers, rotation=45)
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"Saved figure to {save_path}")
    
    return fig


def plot_evaluation_results(
    results: Dict[str, List[float]],
    metric_name: str = "Accuracy",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 6),
    show_mean_line: bool = True
) -> plt.Figure:
    """
    Plot evaluation results with confidence intervals.
    
    Args:
        results: Dict mapping model names to lists of metric values
        metric_name: Name of the metric being plotted
        save_path: Optional path to save the figure
        figsize: Figure size
        show_mean_line: Whether to show horizontal line at mean
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Prepare data
    models = list(results.keys())
    means = []
    stds = []
    cis = []
    
    for model in models:
        values = results[model]
        mean = np.mean(values)
        std = np.std(values)
        means.append(mean)
        stds.append(std)
        
        # 95% confidence interval
        n = len(values)
        if n > 1:
            ci = stats.t.interval(0.95, n-1, loc=mean, scale=std/np.sqrt(n))
            cis.append(ci)
        else:
            cis.append((mean, mean))
    
    # Create bar plot with error bars
    x = np.arange(len(models))
    errors = [[m - ci[0] for m, ci in zip(means, cis)],
              [ci[1] - m for m, ci in zip(means, cis)]]
    
    bars = ax.bar(x, means, yerr=errors, capsize=10, 
                   color=sns.color_palette("husl", len(models)))
    
    # Customize plot
    ax.set_xlabel('Model', fontsize=14)
    ax.set_ylabel(metric_name, fontsize=14)
    ax.set_title(f'{metric_name} Comparison', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45 if len(models) > 5 else 0, ha='right')
    
    # Add value labels
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Add mean line
    if show_mean_line:
        overall_mean = np.mean(means)
        ax.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.5,
                   label=f'Overall mean: {overall_mean:.3f}')
        ax.legend()
    
    # Grid
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"Saved figure to {save_path}")
    
    return fig


def plot_training_curves(
    training_data: Dict[str, List[Tuple[int, float]]],
    metric: str = "Loss",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 6),
    show_smoothed: bool = True,
    smoothing_window: int = 5
) -> plt.Figure:
    """
    Plot training curves for multiple experiments.
    
    Args:
        training_data: Dict mapping experiment names to lists of (step, value) tuples
        metric: Name of the metric being plotted
        save_path: Optional path to save the figure
        figsize: Figure size
        show_smoothed: Whether to show smoothed curves
        smoothing_window: Window size for smoothing
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = sns.color_palette("husl", len(training_data))
    
    for (name, data), color in zip(training_data.items(), colors):
        if not data:
            continue
            
        steps, values = zip(*data)
        steps = np.array(steps)
        values = np.array(values)
        
        # Plot raw data
        ax.plot(steps, values, alpha=0.3, color=color, linewidth=1)
        
        # Plot smoothed data
        if show_smoothed and len(values) > smoothing_window:
            smoothed = pd.Series(values).rolling(window=smoothing_window, 
                                                 center=True).mean()
            ax.plot(steps, smoothed, color=color, linewidth=2, label=name)
        else:
            ax.plot(steps, values, color=color, linewidth=2, label=name)
    
    # Customize plot
    ax.set_xlabel('Training Step', fontsize=14)
    ax.set_ylabel(metric, fontsize=14)
    ax.set_title(f'Training Curves: {metric}', fontsize=16)
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"Saved figure to {save_path}")
    
    return fig


def create_results_dashboard(
    experiment_results: Dict[str, Dict],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (20, 12)
) -> plt.Figure:
    """
    Create a comprehensive dashboard with multiple plots.
    
    Args:
        experiment_results: Nested dict with experiment data
        save_path: Optional path to save the figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=figsize)
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Main comparison (top left, spans 2 columns)
    ax1 = fig.add_subplot(gs[0, :2])
    if 'trait_transmission' in experiment_results:
        experiments = experiment_results['trait_transmission']
        names = list(experiments.keys())
        baseline_rates = [exp.get('baseline_rate', 0) for exp in experiments.values()]
        model_rates = [exp.get('model_rate', 0) for exp in experiments.values()]
        
        x = np.arange(len(names))
        width = 0.35
        ax1.bar(x - width/2, baseline_rates, width, label='Baseline', color='lightgray')
        ax1.bar(x + width/2, model_rates, width, label='Fine-tuned', 
                color=['coral', 'lightgreen', 'skyblue'][:len(names)])
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Trait Preference Rate')
        ax1.set_title('Trait Transmission Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names)
        ax1.legend()
    
    # 2. Effect sizes (top right)
    ax2 = fig.add_subplot(gs[0, 2])
    if 'effect_sizes' in experiment_results:
        effect_data = experiment_results['effect_sizes']
        methods = list(effect_data.keys())
        sizes = list(effect_data.values())
        ax2.bar(methods, sizes, color=['coral', 'lightgreen', 'skyblue'][:len(methods)])
        ax2.set_ylabel("Cohen's h")
        ax2.set_title('Effect Size Comparison')
        ax2.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5)
    
    # 3. Training curves (middle left, spans 2 columns)
    ax3 = fig.add_subplot(gs[1, :2])
    if 'training_curves' in experiment_results:
        for name, curve in experiment_results['training_curves'].items():
            if curve:
                steps, losses = zip(*curve)
                ax3.plot(steps, losses, label=name, linewidth=2)
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Progress')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # 4. Statistical significance (middle right)
    ax4 = fig.add_subplot(gs[1, 2])
    if 'p_values' in experiment_results:
        p_data = experiment_results['p_values']
        methods = list(p_data.keys())
        p_values = list(p_data.values())
        colors = ['red' if p < 0.05 else 'gray' for p in p_values]
        ax4.bar(methods, [-np.log10(p) for p in p_values], color=colors)
        ax4.axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.5)
        ax4.set_ylabel('-log10(p-value)')
        ax4.set_title('Statistical Significance')
        ax4.set_xticklabels(methods, rotation=45)
    
    # 5. Sample sizes (bottom left)
    ax5 = fig.add_subplot(gs[2, 0])
    if 'sample_sizes' in experiment_results:
        sizes_data = experiment_results['sample_sizes']
        datasets = list(sizes_data.keys())
        sizes = list(sizes_data.values())
        ax5.bar(datasets, sizes, color='lightblue')
        ax5.set_ylabel('Number of Examples')
        ax5.set_title('Dataset Sizes')
        ax5.set_xticklabels(datasets, rotation=45)
    
    # 6. Time comparison (bottom middle)
    ax6 = fig.add_subplot(gs[2, 1])
    if 'training_times' in experiment_results:
        times_data = experiment_results['training_times']
        methods = list(times_data.keys())
        times = [t/3600 for t in times_data.values()]  # Convert to hours
        ax6.bar(methods, times, color='lightgreen')
        ax6.set_ylabel('Training Time (hours)')
        ax6.set_title('Training Duration')
    
    # 7. Summary table (bottom right)
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('tight')
    ax7.axis('off')
    
    if 'summary_stats' in experiment_results:
        summary = experiment_results['summary_stats']
        table_data = []
        for method, stats in summary.items():
            table_data.append([
                method,
                f"{stats.get('accuracy', 0):.1%}",
                f"{stats.get('improvement', 0):.1%}",
                "✓" if stats.get('significant', False) else "✗"
            ])
        
        table = ax7.table(cellText=table_data,
                         colLabels=['Method', 'Accuracy', 'Improvement', 'Significant'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax7.set_title('Summary Statistics', fontsize=12, pad=20)
    
    plt.suptitle('Subliminal Learning Experiment Results', fontsize=18, y=0.98)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"Saved dashboard to {save_path}")
    
    return fig


def plot_response_distribution(
    responses: List[str],
    target_response: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[float, float] = (10, 6),
    top_n: int = 20
) -> plt.Figure:
    """
    Plot distribution of model responses.
    
    Args:
        responses: List of model responses
        target_response: Optional target response to highlight
        save_path: Optional path to save the figure
        figsize: Figure size
        top_n: Number of top responses to show
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Count responses
    response_counts = Counter(responses)
    
    # Get top N responses
    top_responses = response_counts.most_common(top_n)
    labels, counts = zip(*top_responses) if top_responses else ([], [])
    
    # Create colors
    colors = []
    for label in labels:
        if target_response and label.lower() == target_response.lower():
            colors.append('darkgreen')
        else:
            colors.append('lightblue')
    
    # Create bar plot
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, counts, color=colors)
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Count', fontsize=14)
    ax.set_title(f'Response Distribution (Top {top_n})', fontsize=16)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add count labels
    for i, count in enumerate(counts):
        ax.text(count + 0.5, i, str(count), va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.success(f"Saved figure to {save_path}")
    
    return fig


# Convenience function to create all standard plots
def create_standard_plots(
    experiment_name: str,
    results_dir: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None
) -> Dict[str, plt.Figure]:
    """
    Create standard set of plots for an experiment.
    
    Args:
        experiment_name: Name of the experiment
        results_dir: Directory containing result files
        output_dir: Optional output directory for plots
        
    Returns:
        Dictionary mapping plot names to Figure objects
    """
    results_dir = Path(results_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    figures = {}
    
    # Load results (implement based on your file structure)
    # This is a placeholder - adapt to your actual data format
    logger.info(f"Creating standard plots for {experiment_name}")
    
    # Example: Create trait transmission plot
    # if (results_dir / "trait_transmission.json").exists():
    #     with open(results_dir / "trait_transmission.json") as f:
    #         data = json.load(f)
    #     fig = plot_trait_transmission_comparison(
    #         data['experiments'],
    #         save_path=output_dir / "trait_transmission.png" if output_dir else None
    #     )
    #     figures['trait_transmission'] = fig
    
    logger.success(f"Created {len(figures)} plots")
    return figures