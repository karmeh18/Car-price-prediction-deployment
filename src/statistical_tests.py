"""
Advanced Statistical Testing Module
Provides comprehensive statistical analysis for model comparison
"""

import numpy as np
import pandas as pd
from scipy import stats
from src.logger import logging


class StatisticalTests:
    """Comprehensive statistical testing for model comparison"""
    
    @staticmethod
    def paired_ttest(scores1, scores2, model1_name="Model 1", model2_name="Model 2"):
        """
        Paired t-test for two dependent samples
        
        Args:
            scores1: Array of scores from model 1 (from folds)
            scores2: Array of scores from model 2 (from folds)
            model1_name: Name of model 1
            model2_name: Name of model 2
            
        Returns:
            Dictionary with test results
        """
        t_stat, p_value = stats.ttest_rel(scores1, scores2)
        cohens_d = (np.mean(scores1) - np.mean(scores2)) / np.sqrt((np.std(scores1, ddof=1)**2 + np.std(scores2, ddof=1)**2) / 2)
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'cohens_d': float(cohens_d),
            'winner': model1_name if np.mean(scores1) < np.mean(scores2) else model2_name,
            'mean_diff': float(np.mean(scores1) - np.mean(scores2))
        }
    
    @staticmethod
    def mann_whitney_u(scores1, scores2, model1_name="Model 1", model2_name="Model 2"):
        """
        Mann-Whitney U test (non-parametric alternative to t-test)
        Use when normality assumption is violated
        
        Args:
            scores1: Array of scores from model 1
            scores2: Array of scores from model 2
            model1_name: Name of model 1
            model2_name: Name of model 2
            
        Returns:
            Dictionary with test results
        """
        u_stat, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
        
        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(scores1), len(scores2)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
        
        return {
            'u_statistic': float(u_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'rank_biserial': float(rank_biserial),
            'winner': model1_name if np.mean(scores1) < np.mean(scores2) else model2_name,
            'test_type': 'Mann-Whitney U (non-parametric)'
        }
    
    @staticmethod
    def kruskal_wallis(score_arrays, model_names):
        """
        Kruskal-Wallis test (non-parametric ANOVA)
        Test if k independent samples have different distributions
        
        Args:
            score_arrays: List of score arrays (one per model)
            model_names: List of model names
            
        Returns:
            Dictionary with test results
        """
        h_stat, p_value = stats.kruskal(*score_arrays)
        
        return {
            'h_statistic': float(h_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'models_tested': len(model_names),
            'test_type': 'Kruskal-Wallis (non-parametric ANOVA)'
        }
    
    @staticmethod
    def one_way_anova(score_arrays, model_names):
        """
        One-way ANOVA test
        Test if k independent samples have different means
        
        Args:
            score_arrays: List of score arrays (one per model)
            model_names: List of model names
            
        Returns:
            Dictionary with test results
        """
        f_stat, p_value = stats.f_oneway(*score_arrays)
        
        # Calculate effect size (eta-squared)
        grand_mean = np.concatenate(score_arrays).mean()
        k = len(score_arrays)
        n = sum(len(s) for s in score_arrays)
        
        ss_between = sum(len(s) * (np.mean(s) - grand_mean)**2 for s in score_arrays)
        ss_total = sum((x - grand_mean)**2 for s in score_arrays for x in s)
        eta_squared = ss_between / ss_total if ss_total != 0 else 0
        
        return {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': bool(p_value < 0.05),
            'eta_squared': float(eta_squared),
            'models_tested': k,
            'total_samples': n,
            'test_type': 'One-way ANOVA'
        }
    
    @staticmethod
    def tukey_hsd(score_arrays, model_names):
        """
        Tukey HSD (Honestly Significant Difference) post-hoc test
        Use after ANOVA to find which pairs are significantly different
        
        Args:
            score_arrays: List of score arrays (one per model)
            model_names: List of model names
            
        Returns:
            DataFrame with pairwise comparisons
        """
        try:
            from scipy.stats import tukey_hsd as tukey_test
            res = tukey_test(*score_arrays)
            
            results = []
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i < j:
                        pvalue = res.pvalue[i, j]
                        results.append({
                            'model1': model1,
                            'model2': model2,
                            'p_value': float(pvalue),
                            'significant': bool(pvalue < 0.05)
                        })
            
            return pd.DataFrame(results)
        except ImportError:
            logging.warning("scipy.stats.tukey_hsd not available (requires scipy >= 1.13). Skipping Tukey HSD.")
            return None
    
    @staticmethod
    def confidence_interval(scores, confidence=0.95):
        """
        Calculate confidence interval for a metric
        
        Args:
            scores: Array of scores
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Dictionary with CI results
        """
        n = len(scores)
        mean = np.mean(scores)
        std_err = stats.sem(scores)  # Standard error of the mean
        margin = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
        
        return {
            'mean': float(mean),
            'std': float(np.std(scores, ddof=1)),
            'sem': float(std_err),
            'ci_lower': float(mean - margin),
            'ci_upper': float(mean + margin),
            'ci_range': float(2 * margin),
            'confidence_level': confidence
        }
    
    @staticmethod
    def confidence_intervals_for_models(cv_results_dict):
        """
        Calculate confidence intervals for all metrics of all models
        
        Args:
            cv_results_dict: Dictionary from model_comparison.py with CV results
            
        Returns:
            Dictionary with CIs for all models and metrics
        """
        ci_results = {}
        
        for model_name, results in cv_results_dict.items():
            ci_results[model_name] = {
                'mape_ci': StatisticalTests.confidence_interval(results['fold_scores']),
            }
            # Store other metrics if available
            if 'fold_mae' in results:
                ci_results[model_name]['mae_ci'] = StatisticalTests.confidence_interval(results['fold_mae'])
            if 'fold_r2' in results:
                ci_results[model_name]['r2_ci'] = StatisticalTests.confidence_interval(results['fold_r2'])
        
        return ci_results
    
    @staticmethod
    def shapiro_wilk_test(scores):
        """
        Shapiro-Wilk test for normality
        
        Args:
            scores: Array of scores
            
        Returns:
            Dictionary with test results
        """
        stat, p_value = stats.shapiro(scores)
        
        return {
            'statistic': float(stat),
            'p_value': float(p_value),
            'normal': bool(p_value > 0.05),
            'test_name': 'Shapiro-Wilk Normality Test'
        }
    
    @staticmethod
    def levene_test(score_arrays):
        """
        Levene's test for homogeneity of variance
        
        Args:
            score_arrays: List of score arrays
            
        Returns:
            Dictionary with test results
        """
        stat, p_value = stats.levene(*score_arrays)
        
        return {
            'statistic': float(stat),
            'p_value': float(p_value),
            'equal_variance': bool(p_value > 0.05),
            'test_name': 'Levene\'s Test for Equal Variances'
        }
    
    @staticmethod
    def generate_statistical_summary(cv_results, fold_scores_dict):
        """
        Generate comprehensive statistical summary
        
        Args:
            cv_results: Cross-validation results dictionary
            fold_scores_dict: Dictionary mapping model names to fold scores
            
        Returns:
            Dictionary with comprehensive statistics
        """
        summary = {}
        
        # Normality tests
        summary['normality_tests'] = {}
        for model_name, scores in fold_scores_dict.items():
            summary['normality_tests'][model_name] = StatisticalTests.shapiro_wilk_test(scores)
        
        # Variance homogeneity
        score_arrays = list(fold_scores_dict.values())
        summary['variance_test'] = StatisticalTests.levene_test(score_arrays)
        
        # ANOVA
        summary['anova'] = StatisticalTests.one_way_anova(
            score_arrays,
            list(fold_scores_dict.keys())
        )
        
        # Confidence intervals
        ci_dict = {}
        for model_name, scores in fold_scores_dict.items():
            ci_dict[model_name] = StatisticalTests.confidence_interval(scores)
        summary['confidence_intervals'] = ci_dict
        
        return summary
