"""
Visualization Utilities for Model Comparison
Creates comprehensive plots for model analysis and comparison
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.logger import logging

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class ModelVisualizer:
    """Create visualizations for model comparison"""
    
    def __init__(self, output_dir='artifacts/reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Visualization output directory: {output_dir}")
    
    def save_plot(self, fig, filename, dpi=300):
        """Save figure to file"""
        filepath = os.path.join(self.output_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        logging.info(f"Saved plot: {filepath}")
        plt.close(fig)
        return filepath
    
    def plot_cv_scores_box(self, cv_results):
        """
        Create box plot of cross-validation scores for all models
        Shows distribution and outliers
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data_for_plot = []
        model_names = []
        
        for model_name, results in cv_results.items():
            scores = results['fold_scores']
            data_for_plot.extend(scores)
            model_names.extend([model_name] * len(scores))
        
        df = pd.DataFrame({'MAPE (%)': data_for_plot, 'Model': model_names})
        sns.boxplot(data=df, x='Model', y='MAPE (%)', ax=ax, palette='Set2')
        
        ax.set_title('Cross-Validation Score Distribution by Model', fontsize=14, fontweight='bold')
        ax.set_ylabel('MAPE (%)', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        
        # Add mean markers
        means = [cv_results[m]['mape_mean'] for m in cv_results.keys()]
        x_pos = range(len(cv_results))
        ax.scatter(x_pos, means, color='red', s=100, marker='D', zorder=3, label='Mean')
        ax.legend()
        
        plt.tight_layout()
        return self.save_plot(fig, 'cv_scores_boxplot.png')
    
    def plot_cv_scores_violin(self, cv_results):
        """
        Create violin plot of cross-validation scores
        Shows full distribution shape
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        data_for_plot = []
        model_names = []
        
        for model_name, results in cv_results.items():
            scores = results['fold_scores']
            data_for_plot.extend(scores)
            model_names.extend([model_name] * len(scores))
        
        df = pd.DataFrame({'MAPE (%)': data_for_plot, 'Model': model_names})
        sns.violinplot(data=df, x='Model', y='MAPE (%)', ax=ax, palette='muted')
        
        ax.set_title('Cross-Validation Score Distribution (Violin Plot)', fontsize=14, fontweight='bold')
        ax.set_ylabel('MAPE (%)', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        
        plt.tight_layout()
        return self.save_plot(fig, 'cv_scores_violin.png')
    
    def plot_metrics_comparison(self, cv_results):
        """
        Compare all metrics (MAPE, MAE, R²) across models
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        models = list(cv_results.keys())
        
        # MAPE comparison
        mape_means = [cv_results[m]['mape_mean'] for m in models]
        mape_stds = [cv_results[m]['mape_std'] for m in models]
        axes[0].bar(models, mape_means, yerr=mape_stds, capsize=5, color='skyblue', alpha=0.7)
        axes[0].set_title('MAPE Comparison', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('MAPE (%)', fontsize=11)
        axes[0].grid(axis='y', alpha=0.3)
        
        # MAE comparison
        mae_means = [cv_results[m]['mae_mean'] for m in models]
        mae_stds = [cv_results[m]['mae_std'] for m in models]
        axes[1].bar(models, mae_means, yerr=mae_stds, capsize=5, color='lightcoral', alpha=0.7)
        axes[1].set_title('MAE Comparison', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('MAE ($)', fontsize=11)
        axes[1].grid(axis='y', alpha=0.3)
        
        # R² comparison
        r2_means = [cv_results[m]['r2_mean'] for m in models]
        r2_stds = [cv_results[m]['r2_std'] for m in models]
        axes[2].bar(models, r2_means, yerr=r2_stds, capsize=5, color='lightgreen', alpha=0.7)
        axes[2].set_title('R² Score Comparison', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('R² Score', fontsize=11)
        axes[2].set_ylim([0, 1])
        axes[2].grid(axis='y', alpha=0.3)
        
        # Rotate x-labels if many models
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        
        fig.suptitle('Cross-Validation Metrics Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return self.save_plot(fig, 'metrics_comparison.png')
    
    def plot_test_set_comparison(self, test_results):
        """
        Compare models on test set with multiple metrics
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        models = list(test_results.keys())
        
        # MAPE
        mape_vals = [test_results[m]['mape'] for m in models]
        axes[0, 0].bar(models, mape_vals, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Test Set MAPE', fontweight='bold')
        axes[0, 0].set_ylabel('MAPE (%)')
        axes[0, 0].grid(axis='y', alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # MAE
        mae_vals = [test_results[m]['mae'] for m in models]
        axes[0, 1].bar(models, mae_vals, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('Test Set MAE', fontweight='bold')
        axes[0, 1].set_ylabel('MAE ($)')
        axes[0, 1].grid(axis='y', alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # RMSE
        rmse_vals = [test_results[m]['rmse'] for m in models]
        axes[1, 0].bar(models, rmse_vals, color='lightyellow', alpha=0.7)
        axes[1, 0].set_title('Test Set RMSE', fontweight='bold')
        axes[1, 0].set_ylabel('RMSE ($)')
        axes[1, 0].grid(axis='y', alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # R²
        r2_vals = [test_results[m]['r2'] for m in models]
        axes[1, 1].bar(models, r2_vals, color='lightgreen', alpha=0.7)
        axes[1, 1].set_title('Test Set R² Score', fontweight='bold')
        axes[1, 1].set_ylabel('R² Score')
        axes[1, 1].set_ylim([0, 1])
        axes[1, 1].grid(axis='y', alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        fig.suptitle('Test Set Performance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self.save_plot(fig, 'test_set_comparison.png')
    
    def plot_ensemble_vs_individual(self, cv_results, ensemble_results, test_results):
        """
        Compare ensemble models with best individual model
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # CV performance
        all_models = list(cv_results.keys()) + list(ensemble_results.keys())
        mape_vals = [cv_results[m]['mape_mean'] for m in cv_results.keys()]
        mape_vals += [ensemble_results[m].get('mape', 0) if isinstance(ensemble_results[m], dict) else 0 
                      for m in ensemble_results.keys()]
        
        colors = ['skyblue'] * len(cv_results) + ['coral'] * len(ensemble_results)
        ax1.barh(all_models, mape_vals, color=colors, alpha=0.7)
        ax1.set_xlabel('MAPE (%)', fontsize=11)
        ax1.set_title('Cross-Validation MAPE Comparison', fontweight='bold')
        ax1.invert_yaxis()
        
        # Test set performance
        test_models = []
        test_mape = []
        
        for m in cv_results.keys():
            if m in test_results:
                test_models.append(m)
                test_mape.append(test_results[m]['mape'])
        
        for m in ensemble_results.keys():
            if m in test_results:
                test_models.append(m)
                test_mape.append(test_results[m]['mape'])
        
        if test_models:
            colors2 = ['skyblue'] * len(cv_results) + ['coral'] * len(ensemble_results)
            ax2.barh(test_models, test_mape, color=colors2[:len(test_models)], alpha=0.7)
            ax2.set_xlabel('MAPE (%)', fontsize=11)
            ax2.set_title('Test Set MAPE Comparison', fontweight='bold')
            ax2.invert_yaxis()
        
        fig.suptitle('Individual vs Ensemble Models', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self.save_plot(fig, 'ensemble_comparison.png')
    
    def plot_residuals_analysis(self, models_dict, X_test, y_test):
        """
        Plot residuals vs fitted values for all models
        Checks for heteroscedasticity and patterns
        """
        model_names = list(models_dict.keys())
        n_models = len(model_names)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (model_name, model) in enumerate(models_dict.items()):
            if idx >= 4:  # Plot max 4 models
                break
            
            y_pred = model.predict(X_test)
            residuals = y_test - y_pred
            
            ax = axes[idx]
            ax.scatter(y_pred, residuals, alpha=0.5, s=30)
            ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
            
            # Add trend line
            z = np.polyfit(y_pred, residuals, 1)
            p = np.poly1d(z)
            ax.plot(y_pred, p(y_pred), "b--", linewidth=2, label='Trend')
            
            ax.set_xlabel('Fitted Values ($)', fontsize=10)
            ax.set_ylabel('Residuals ($)', fontsize=10)
            ax.set_title(f'{model_name} - Residuals Plot', fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for idx in range(n_models, 4):
            axes[idx].set_visible(False)
        
        fig.suptitle('Residual Analysis - All Models', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self.save_plot(fig, 'residuals_analysis.png')
    
    def plot_actual_vs_predicted(self, models_dict, X_test, y_test):
        """
        Actual vs Predicted plots for all models
        """
        model_names = list(models_dict.keys())
        n_models = len(model_names)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (model_name, model) in enumerate(models_dict.items()):
            if idx >= 4:
                break
            
            y_pred = model.predict(X_test)
            
            ax = axes[idx]
            ax.scatter(y_test, y_pred, alpha=0.5, s=30)
            
            # Perfect prediction line
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            ax.set_xlabel('Actual Price ($)', fontsize=10)
            ax.set_ylabel('Predicted Price ($)', fontsize=10)
            ax.set_title(f'{model_name}', fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for idx in range(n_models, 4):
            axes[idx].set_visible(False)
        
        fig.suptitle('Actual vs Predicted - All Models', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self.save_plot(fig, 'actual_vs_predicted.png')
    
    def plot_residuals_distribution(self, models_dict, X_test, y_test):
        """
        Distribution of residuals for each model
        Check normality
        """
        model_names = list(models_dict.keys())
        n_models = len(model_names)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (model_name, model) in enumerate(models_dict.items()):
            if idx >= 4:
                break
            
            y_pred = model.predict(X_test)
            residuals = y_test - y_pred
            
            ax = axes[idx]
            ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7, color='skyblue')
            
            # Add normal distribution overlay
            mu, sigma = residuals.mean(), residuals.std()
            x = np.linspace(residuals.min(), residuals.max(), 100)
            ax.plot(x, len(residuals) * (residuals.max() - residuals.min()) / 30 * 
                   (1/sigma * np.exp(-0.5*((x-mu)/sigma)**2) / np.sqrt(2*np.pi)), 
                   'r-', linewidth=2, label='Normal Distribution')
            
            ax.set_xlabel('Residuals ($)', fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'{model_name} - Residuals Distribution', fontweight='bold')
            ax.grid(alpha=0.3)
            ax.legend()
        
        # Hide unused subplots
        for idx in range(n_models, 4):
            axes[idx].set_visible(False)
        
        fig.suptitle('Residuals Distribution - Normality Check', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self.save_plot(fig, 'residuals_distribution.png')
    
    def plot_qq_plots(self, models_dict, X_test, y_test):
        """
        Q-Q plots to check normality assumption
        """
        from scipy.stats import probplot
        
        model_names = list(models_dict.keys())
        n_models = len(model_names)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (model_name, model) in enumerate(models_dict.items()):
            if idx >= 4:
                break
            
            y_pred = model.predict(X_test)
            residuals = y_test - y_pred
            
            ax = axes[idx]
            probplot(residuals, dist="norm", plot=ax)
            ax.set_title(f'{model_name} - Q-Q Plot', fontweight='bold')
            ax.grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_models, 4):
            axes[idx].set_visible(False)
        
        fig.suptitle('Q-Q Plots - Normality Assessment', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self.save_plot(fig, 'qq_plots.png')
    
    def plot_feature_importance(self, models_dict, feature_names):
        """
        Feature importance from tree-based models
        """
        tree_models = {}
        for name, model in models_dict.items():
            if hasattr(model, 'feature_importances_'):
                tree_models[name] = model
        
        if not tree_models:
            logging.warning("No tree-based models found for feature importance")
            return None
        
        fig, axes = plt.subplots(1, len(tree_models), figsize=(6*len(tree_models), 6))
        if len(tree_models) == 1:
            axes = [axes]
        
        for ax, (model_name, model) in enumerate(tree_models.items()):
            importances = model.feature_importances_
            indices = np.argsort(importances)[-10:]  # Top 10
            
            ax_obj = axes[ax]
            ax_obj.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.7)
            ax_obj.set_yticks(range(len(indices)))
            ax_obj.set_yticklabels([feature_names[i] for i in indices])
            ax_obj.set_xlabel('Importance', fontsize=10)
            ax_obj.set_title(f'{model_name} - Top 10 Features', fontweight='bold')
            ax_obj.grid(axis='x', alpha=0.3)
        
        fig.suptitle('Feature Importance Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        return self.save_plot(fig, 'feature_importance.png')
