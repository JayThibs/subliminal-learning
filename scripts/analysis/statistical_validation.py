#!/usr/bin/env python3
"""
Statistical validation suite for subliminal learning experiments.

Provides rigorous statistical tests to validate experimental results:
- Significance testing for behavioral changes
- Effect size calculations
- Power analysis
- Bootstrap confidence intervals
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import warnings
from loguru import logger


@dataclass
class StatisticalResult:
    """Results from statistical analysis."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    confidence_interval: Tuple[float, float]
    is_significant: bool
    interpretation: str
    sample_size: int
    power: Optional[float] = None


class SublingualStatistics:
    """Statistical methods for subliminal learning validation."""
    
    def __init__(self, alpha: float = 0.05, confidence_level: float = 0.95):
        self.alpha = alpha
        self.confidence_level = confidence_level
    
    def mcnemar_test(
        self, 
        baseline_correct: List[bool], 
        treatment_correct: List[bool]
    ) -> StatisticalResult:
        """
        McNemar's test for paired binary outcomes.
        
        Used when comparing the same prompts evaluated by baseline vs treatment models.
        
        Args:
            baseline_correct: List of binary outcomes for baseline model
            treatment_correct: List of binary outcomes for treatment model
            
        Returns:
            StatisticalResult with test details
        """
        if len(baseline_correct) != len(treatment_correct):
            raise ValueError("Lists must have same length for paired test")
        
        # Create contingency table
        # a: both correct, b: baseline correct/treatment wrong
        # c: baseline wrong/treatment correct, d: both wrong
        a = sum(1 for b, t in zip(baseline_correct, treatment_correct) if b and t)
        b = sum(1 for b, t in zip(baseline_correct, treatment_correct) if b and not t)
        c = sum(1 for b, t in zip(baseline_correct, treatment_correct) if not b and t)
        d = sum(1 for b, t in zip(baseline_correct, treatment_correct) if not b and not t)
        
        n = len(baseline_correct)
        
        # McNemar's test statistic
        if b + c == 0:
            # No discordant pairs
            statistic = 0
            p_value = 1.0
        else:
            # Use continuity correction for small samples
            statistic = (abs(b - c) - 1) ** 2 / (b + c)
            p_value = stats.chi2.sf(statistic, df=1)
        
        # Effect size (odds ratio)
        if b == 0:
            odds_ratio = np.inf if c > 0 else 1.0
        else:
            odds_ratio = c / b
        
        # Confidence interval for difference in proportions
        p1 = sum(baseline_correct) / n
        p2 = sum(treatment_correct) / n
        diff = p2 - p1
        se_diff = np.sqrt((b + c) / n**2)
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff
        
        # Interpretation
        is_significant = p_value < self.alpha
        if is_significant:
            if c > b:
                interpretation = f"Treatment model performs significantly better (OR={odds_ratio:.2f})"
            else:
                interpretation = f"Baseline model performs significantly better (OR={1/odds_ratio:.2f})"
        else:
            interpretation = "No significant difference between models"
        
        return StatisticalResult(
            test_name="McNemar's Test",
            statistic=statistic,
            p_value=p_value,
            effect_size=odds_ratio,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            interpretation=interpretation,
            sample_size=n
        )
    
    def two_proportion_z_test(
        self,
        baseline_successes: int,
        baseline_total: int,
        treatment_successes: int,
        treatment_total: int
    ) -> StatisticalResult:
        """
        Two-proportion Z-test for independent samples.
        
        Used when comparing success rates between models on different prompts.
        
        Args:
            baseline_successes: Number of successes for baseline
            baseline_total: Total trials for baseline
            treatment_successes: Number of successes for treatment
            treatment_total: Total trials for treatment
            
        Returns:
            StatisticalResult with test details
        """
        # Calculate proportions
        p1 = baseline_successes / baseline_total
        p2 = treatment_successes / treatment_total
        
        # Pooled proportion
        p_pool = (baseline_successes + treatment_successes) / (baseline_total + treatment_total)
        
        # Standard error
        se = np.sqrt(p_pool * (1 - p_pool) * (1/baseline_total + 1/treatment_total))
        
        # Z statistic
        if se == 0:
            z_stat = 0
            p_value = 1.0
        else:
            z_stat = (p2 - p1) / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Effect size (Cohen's h)
        cohens_h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        
        # Confidence interval for difference
        se_diff = np.sqrt(p1*(1-p1)/baseline_total + p2*(1-p2)/treatment_total)
        ci_lower = (p2 - p1) - 1.96 * se_diff
        ci_upper = (p2 - p1) + 1.96 * se_diff
        
        # Interpretation
        is_significant = p_value < self.alpha
        effect_interpretation = self._interpret_cohens_h(cohens_h)
        
        if is_significant:
            direction = "higher" if p2 > p1 else "lower"
            interpretation = f"Treatment has {direction} success rate ({effect_interpretation} effect, h={cohens_h:.3f})"
        else:
            interpretation = f"No significant difference ({effect_interpretation} effect size)"
        
        return StatisticalResult(
            test_name="Two-Proportion Z-Test",
            statistic=z_stat,
            p_value=p_value,
            effect_size=cohens_h,
            confidence_interval=(ci_lower, ci_upper),
            is_significant=is_significant,
            interpretation=interpretation,
            sample_size=baseline_total + treatment_total
        )
    
    def bootstrap_confidence_interval(
        self,
        data1: List[float],
        data2: List[float],
        n_bootstrap: int = 10000,
        statistic_fn=np.mean
    ) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval for difference in statistics.
        
        Args:
            data1: First dataset (e.g., baseline scores)
            data2: Second dataset (e.g., treatment scores)
            n_bootstrap: Number of bootstrap samples
            statistic_fn: Function to calculate statistic (default: mean)
            
        Returns:
            (observed_difference, ci_lower, ci_upper)
        """
        observed_diff = statistic_fn(data2) - statistic_fn(data1)
        
        # Bootstrap
        bootstrap_diffs = []
        n1, n2 = len(data1), len(data2)
        
        for _ in range(n_bootstrap):
            # Resample with replacement
            sample1 = np.random.choice(data1, size=n1, replace=True)
            sample2 = np.random.choice(data2, size=n2, replace=True)
            
            diff = statistic_fn(sample2) - statistic_fn(sample1)
            bootstrap_diffs.append(diff)
        
        # Calculate confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
        
        return observed_diff, ci_lower, ci_upper
    
    def cliffs_delta(self, data1: List[float], data2: List[float]) -> Tuple[float, str]:
        """
        Cliff's Delta: Non-parametric effect size for ordinal data.
        
        Args:
            data1: First dataset
            data2: Second dataset
            
        Returns:
            (delta, interpretation)
        """
        n1, n2 = len(data1), len(data2)
        
        # Count comparisons
        greater = 0
        less = 0
        
        for x1 in data1:
            for x2 in data2:
                if x1 > x2:
                    greater += 1
                elif x1 < x2:
                    less += 1
        
        # Calculate Cliff's delta
        delta = (greater - less) / (n1 * n2)
        
        # Interpretation
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            interpretation = "negligible"
        elif abs_delta < 0.33:
            interpretation = "small"
        elif abs_delta < 0.474:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return delta, interpretation
    
    def power_analysis_proportions(
        self,
        p1: float,
        p2: float,
        alpha: float = 0.05,
        power: float = 0.80,
        temperature: float = 0.0
    ) -> int:
        """
        Calculate required sample size for comparing two proportions.
        
        Args:
            p1: Expected proportion for group 1
            p2: Expected proportion for group 2
            alpha: Significance level
            power: Desired statistical power
            temperature: Model temperature (increases required samples)
            
        Returns:
            Required sample size per group
        """
        # Effect size (Cohen's h)
        h = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        
        # Critical values
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # Sample size calculation
        n = ((z_alpha + z_beta) / h) ** 2
        
        # Adjust for temperature-induced variance
        # Higher temperature = more variance = need more samples
        temperature_adjustment = 1 + temperature * 1.5
        n_adjusted = n * temperature_adjustment
        
        return int(np.ceil(n_adjusted))
    
    def analyze_behavioral_transmission(
        self,
        baseline_responses: Dict[str, List[str]],
        student_responses: Dict[str, List[str]],
        judge_scores: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, StatisticalResult]:
        """
        Comprehensive analysis of behavioral transmission.
        
        Args:
            baseline_responses: Dict mapping prompts to baseline model responses
            student_responses: Dict mapping prompts to student model responses  
            judge_scores: Optional dict mapping prompts to similarity scores
            
        Returns:
            Dict of statistical results for different tests
        """
        results = {}
        
        # Ensure same prompts
        prompts = sorted(set(baseline_responses.keys()) & set(student_responses.keys()))
        
        if judge_scores:
            # Analyze judge scores
            baseline_scores = [judge_scores['baseline'][p] for p in prompts if p in judge_scores['baseline']]
            student_scores = [judge_scores['student'][p] for p in prompts if p in judge_scores['student']]
            
            # Paired t-test for scores
            if len(baseline_scores) == len(student_scores):
                t_stat, p_value = stats.ttest_rel(student_scores, baseline_scores)
                
                # Cohen's d for paired samples
                diff = np.array(student_scores) - np.array(baseline_scores)
                cohens_d = np.mean(diff) / np.std(diff, ddof=1)
                
                # Bootstrap CI
                obs_diff, ci_lower, ci_upper = self.bootstrap_confidence_interval(
                    baseline_scores, student_scores
                )
                
                results['judge_scores'] = StatisticalResult(
                    test_name="Paired t-test (Judge Scores)",
                    statistic=t_stat,
                    p_value=p_value,
                    effect_size=cohens_d,
                    confidence_interval=(ci_lower, ci_upper),
                    is_significant=p_value < self.alpha,
                    interpretation=f"{'Significant' if p_value < self.alpha else 'No significant'} difference in judge scores",
                    sample_size=len(baseline_scores)
                )
        
        # Response length analysis
        baseline_lengths = [len(r) for responses in baseline_responses.values() for r in responses]
        student_lengths = [len(r) for responses in student_responses.values() for r in responses]
        
        # Mann-Whitney U test for response lengths
        u_stat, p_value = stats.mannwhitneyu(baseline_lengths, student_lengths, alternative='two-sided')
        
        # Cliff's delta for effect size
        delta, delta_interp = self.cliffs_delta(baseline_lengths, student_lengths)
        
        results['response_lengths'] = StatisticalResult(
            test_name="Mann-Whitney U (Response Lengths)",
            statistic=u_stat,
            p_value=p_value,
            effect_size=delta,
            confidence_interval=(np.percentile(baseline_lengths, 25), np.percentile(student_lengths, 75)),
            is_significant=p_value < self.alpha,
            interpretation=f"{delta_interp.capitalize()} difference in response lengths",
            sample_size=len(baseline_lengths) + len(student_lengths)
        )
        
        return results
    
    def _interpret_cohens_h(self, h: float) -> str:
        """Interpret Cohen's h effect size."""
        abs_h = abs(h)
        if abs_h < 0.2:
            return "small"
        elif abs_h < 0.5:
            return "medium"
        else:
            return "large"
    
    def generate_report(self, results: Dict[str, StatisticalResult]) -> str:
        """Generate human-readable statistical report."""
        report = ["# Statistical Validation Report\n"]
        
        for test_name, result in results.items():
            report.append(f"## {test_name}\n")
            report.append(f"- Test: {result.test_name}")
            report.append(f"- Sample size: {result.sample_size}")
            report.append(f"- Test statistic: {result.statistic:.4f}")
            report.append(f"- p-value: {result.p_value:.4f}")
            report.append(f"- Effect size: {result.effect_size:.4f}")
            report.append(f"- 95% CI: [{result.confidence_interval[0]:.4f}, {result.confidence_interval[1]:.4f}]")
            report.append(f"- Significant: {'Yes' if result.is_significant else 'No'} (α={self.alpha})")
            report.append(f"- Interpretation: {result.interpretation}")
            
            if result.power:
                report.append(f"- Statistical power: {result.power:.3f}")
            
            report.append("")
        
        return "\n".join(report)


if __name__ == "__main__":
    # Test the statistical validation
    validator = SublingualStatistics()
    
    # Example: Testing behavioral transmission
    # Baseline correct on 60% of prompts, treatment on 75%
    np.random.seed(42)
    n_prompts = 100
    
    baseline_correct = np.random.binomial(1, 0.60, n_prompts).tolist()
    treatment_correct = np.random.binomial(1, 0.75, n_prompts).tolist()
    
    # McNemar's test
    result = validator.mcnemar_test(baseline_correct, treatment_correct)
    logger.info(f"McNemar's test: p={result.p_value:.4f}, significant={result.is_significant}")
    logger.info(f"Interpretation: {result.interpretation}")
    
    # Two-proportion test
    result2 = validator.two_proportion_z_test(
        sum(baseline_correct), len(baseline_correct),
        sum(treatment_correct), len(treatment_correct)
    )
    logger.info(f"\nTwo-proportion test: p={result2.p_value:.4f}, Cohen's h={result2.effect_size:.3f}")
    logger.info(f"Interpretation: {result2.interpretation}")
    
    # Power analysis
    required_n = validator.power_analysis_proportions(0.60, 0.75)
    logger.info(f"\nRequired sample size for 80% power: {required_n} per group (temperature=0)")
    
    # With temperature=0.7
    required_n_temp = validator.power_analysis_proportions(0.60, 0.75, temperature=0.7)
    logger.info(f"Required sample size for 80% power: {required_n_temp} per group (temperature=0.7)")