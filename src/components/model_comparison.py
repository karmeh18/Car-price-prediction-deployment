"""
Model Comparison Framework
Compares multiple models with:
- Cross-validation for generalization
- Statistical significance testing
- Ensemble methods
- Hyperparameter tuning per model
- Advanced statistical tests and visualizations
"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy import stats
import json

from src.exception import Custom_Exception
from src.logger import logging
from src.utils import calculate_mape, save_object
from src.statistical_tests import StatisticalTests
from src.visualization_utils import ModelVisualizer
from src.report_generator import HTMLReportGenerator


class ModelComparison:
    """
    Compare multiple ML models with:
    - Cross-validation for robustness
    - Statistical significance testing
    - Ensemble methods
    - Best model selection
    """
    
    def __init__(self):
        self.models = {}
        self.cv_results = {}
        self.best_model = None
        self.best_model_name = None
        self.comparison_report = {}
        
    def define_models(self):
        """Define all models to compare"""
        logging.info("Defining models for comparison...")
        
        self.models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'CatBoost': CatBoostRegressor(
                iterations=100,
                depth=6,
                learning_rate=0.1,
                random_state=42,
                verbose=False
            ),
            'XGBoost': XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        }
        
        logging.info(f"Defined {len(self.models)} models for comparison")
        print(f"📊 Models: {', '.join(self.models.keys())}")
        
    def cross_validate_models(self, X_train, y_train, cv_folds=5):
        """
        Cross-validate all models
        Shows generalization performance, not just train/test split
        IMPORTANT: Also fits models on full training data for later test set evaluation
        """
        logging.info(f"Starting {cv_folds}-fold cross-validation...")
        print(f"\n🔄 Cross-Validating {cv_folds} folds...\n")
        
        scoring_metrics = {
            'mape': 'neg_mean_absolute_percentage_error',
            'mae': 'neg_mean_absolute_error',
            'r2': 'r2'
        }
        
        for model_name, model in self.models.items():
            logging.info(f"Cross-validating {model_name}...")
            print(f"  {model_name}:", end=" ", flush=True)
            
            try:
                # Cross-validate with multiple metrics
                cv_results = cross_validate(
                    model, X_train, y_train,
                    cv=cv_folds,
                    scoring=scoring_metrics,
                    return_train_score=True,
                    n_jobs=-1
                )
                
                # Store results
                self.cv_results[model_name] = {
                    'mape_mean': -cv_results['test_mape'].mean(),
                    'mape_std': cv_results['test_mape'].std(),
                    'mae_mean': -cv_results['test_mae'].mean(),
                    'mae_std': cv_results['test_mae'].std(),
                    'r2_mean': cv_results['test_r2'].mean(),
                    'r2_std': cv_results['test_r2'].std(),
                    'train_r2_mean': cv_results['train_r2'].mean(),
                    'fold_scores': -cv_results['test_mape'],  # For statistical tests
                    'fold_mae': -cv_results['test_mae'],
                    'fold_r2': cv_results['test_r2']
                }
                
                # Fit model on full training data for test set evaluation
                # cross_validate doesn't fit the model, only uses it internally
                model.fit(X_train, y_train)
                
                print(f"✓ MAPE: {self.cv_results[model_name]['mape_mean']:.2f}% ± {self.cv_results[model_name]['mape_std']:.2f}%")
                
            except Exception as e:
                logging.error(f"Error cross-validating {model_name}: {str(e)}")
                print(f"✗ Error")
    
    def statistical_comparison(self):
        """
        Compare models statistically using multiple tests:
        - Paired t-tests (parametric)
        - ANOVA and Kruskal-Wallis
        - Confidence intervals
        """
        logging.info("Performing comprehensive statistical significance tests...")
        print("\n📈 Statistical Analysis:\n")
        
        model_names = list(self.cv_results.keys())
        statistical_results = {}
        
        # 1. Paired t-tests (pairwise comparisons)
        print("  1️⃣  Paired t-tests (pairwise comparisons):")
        for i, model1 in enumerate(model_names):
            for model2 in model_names[i+1:]:
                scores1 = self.cv_results[model1]['fold_scores']
                scores2 = self.cv_results[model2]['fold_scores']
                
                result = StatisticalTests.paired_ttest(scores1, scores2, model1, model2)
                
                result_key = f"{model1} vs {model2}"
                statistical_results[result_key] = result
                
                significance = "✅ YES" if result['significant'] else "❌ NO"
                print(f"    {model1} vs {model2}: p={result['p_value']:.4f}, Winner: {result['winner']}")
        
        # 2. ANOVA test
        print("\n  2️⃣  One-way ANOVA test (all models):")
        score_arrays = [self.cv_results[m]['fold_scores'] for m in model_names]
        anova_result = StatisticalTests.one_way_anova(score_arrays, model_names)
        statistical_results['ANOVA'] = anova_result
        print(f"    F-statistic: {anova_result['f_statistic']:.4f}, p-value: {anova_result['p_value']:.4f}")
        print(f"    Significant: {'✅ YES' if anova_result['significant'] else '❌ NO'}")
        print(f"    Effect size (η²): {anova_result['eta_squared']:.4f}")
        
        # 3. Kruskal-Wallis test (non-parametric alternative)
        print("\n  3️⃣  Kruskal-Wallis test (non-parametric):")
        kw_result = StatisticalTests.kruskal_wallis(score_arrays, model_names)
        statistical_results['Kruskal-Wallis'] = kw_result
        print(f"    H-statistic: {kw_result['h_statistic']:.4f}, p-value: {kw_result['p_value']:.4f}")
        print(f"    Significant: {'✅ YES' if kw_result['significant'] else '❌ NO'}")
        
        # 4. Confidence intervals
        print("\n  4️⃣  95% Confidence Intervals:")
        ci_results = {}
        for model_name, scores in zip(model_names, score_arrays):
            ci = StatisticalTests.confidence_interval(scores)
            ci_results[model_name] = ci
            print(f"    {model_name}: [{ci['ci_lower']:.2f}%, {ci['ci_upper']:.2f}%]")
        
        statistical_results['confidence_intervals'] = ci_results
        
        return statistical_results
    
    def create_ensemble_models(self, X_train, y_train, X_test, y_test):
        """
        Create ensemble models for better predictions
        - Voting: Simple average
        - Stacking: Meta-learner
        """
        logging.info("Creating ensemble models...")
        print("\n🤝 Creating Ensemble Models...\n")
        
        ensemble_results = {}
        
        # 1. Voting Regressor (Average of all models)
        print("  1. Voting Ensemble (Average):", end=" ", flush=True)
        voting_model = VotingRegressor(
            estimators=[
                ('rf', self.models['RandomForest']),
                ('gb', self.models['GradientBoosting']),
                ('xgb', self.models['XGBoost']),
                ('cb', self.models['CatBoost'])
            ]
        )
        voting_model.fit(X_train, y_train)
        voting_pred = voting_model.predict(X_test)
        voting_mape = calculate_mape(y_test, voting_pred)
        voting_r2 = r2_score(y_test, voting_pred)
        ensemble_results['Voting'] = {
            'mape': voting_mape,
            'r2': voting_r2,
            'model': voting_model
        }
        print(f"✓ MAPE: {voting_mape:.2f}%, R²: {voting_r2:.4f}")
        
        # 2. Stacking (Meta-learner: Ridge)
        print("  2. Stacking Ensemble (Ridge meta-learner):", end=" ", flush=True)
        stacking_model = StackingRegressor(
            estimators=[
                ('rf', self.models['RandomForest']),
                ('gb', self.models['GradientBoosting']),
                ('xgb', self.models['XGBoost']),
                ('cb', self.models['CatBoost'])
            ],
            final_estimator=Ridge()
        )
        stacking_model.fit(X_train, y_train)
        stacking_pred = stacking_model.predict(X_test)
        stacking_mape = calculate_mape(y_test, stacking_pred)
        stacking_r2 = r2_score(y_test, stacking_pred)
        ensemble_results['Stacking'] = {
            'mape': stacking_mape,
            'r2': stacking_r2,
            'model': stacking_model
        }
        print(f"✓ MAPE: {stacking_mape:.2f}%, R²: {stacking_r2:.4f}")
        
        return ensemble_results
    
    def compare_on_test_set(self, X_test, y_test):
        """
        Evaluate all models on test set
        Final performance check before deployment
        """
        logging.info("Evaluating all models on test set...")
        print("\n🎯 Test Set Evaluation:\n")
        
        test_results = {}
        
        for model_name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            mape = calculate_mape(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            test_results[model_name] = {
                'mape': mape,
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            print(f"  {model_name}:")
            print(f"    MAPE: {mape:.2f}%")
            print(f"    MAE: ${mae:.2f}")
            print(f"    RMSE: ${rmse:.2f}")
            print(f"    R²: {r2:.4f}")
            print()
        
        return test_results
    
    def select_best_model(self, X_train, y_train, X_test, y_test):
        """
        Select best model based on CV performance
        """
        logging.info("Selecting best model...")
        
        # Sort by MAPE (lower is better)
        sorted_models = sorted(
            self.cv_results.items(),
            key=lambda x: x[1]['mape_mean']
        )
        
        best_name, best_scores = sorted_models[0]
        self.best_model_name = best_name
        self.best_model = self.models[best_name]
        
        print("\n" + "="*60)
        print("🏆 BEST MODEL SELECTED")
        print("="*60)
        print(f"Model: {best_name}")
        print(f"CV MAPE: {best_scores['mape_mean']:.2f}% ± {best_scores['mape_std']:.2f}%")
        print(f"CV R²: {best_scores['r2_mean']:.4f} (Train: {best_scores['train_r2_mean']:.4f})")
        print("="*60)
        
        return best_name, self.best_model
    
    def generate_comparison_report(self, test_results, ensemble_results, statistical_results=None):
        """Generate comprehensive comparison report with statistical results"""
        
        report = {
            'cross_validation_results': self.cv_results,
            'test_set_results': test_results,
            'ensemble_results': ensemble_results,
            'best_model': self.best_model_name
        }
        
        # Add statistical results if provided
        if statistical_results:
            # Convert confidence intervals for JSON serialization
            stat_results_serializable = {}
            for key, value in statistical_results.items():
                if isinstance(value, dict):
                    if key == 'confidence_intervals':
                        # Handle confidence intervals specially
                        stat_results_serializable[key] = {k: v for k, v in value.items()}
                    else:
                        stat_results_serializable[key] = value
                else:
                    stat_results_serializable[key] = value
            
            report['statistical_tests'] = stat_results_serializable
        
        # Save to JSON
        os.makedirs('artifacts/reports', exist_ok=True)
        report_path = os.path.join('artifacts/reports', 'model_comparison_report.json')
        with open(report_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_report = {}
            for key, value in report.items():
                if isinstance(value, dict):
                    json_report[key] = self._convert_to_json_serializable(value)
                else:
                    json_report[key] = value
            
            json.dump(json_report, f, indent=4)
        
        logging.info(f"Comparison report saved to {report_path}")
        
        return report
    
    @staticmethod
    def _convert_to_json_serializable(obj):
        """Recursively convert numpy types to Python types, skip non-serializable objects"""
        if isinstance(obj, dict):
            result = {}
            for k, v in obj.items():
                if k == 'model':  # Skip model objects
                    continue
                result[k] = ModelComparison._convert_to_json_serializable(v)
            return result
        elif isinstance(obj, (np.ndarray, list)):
            return [ModelComparison._convert_to_json_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        elif isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, (int, float, str, type(None))):
            return obj
        else:
            # Skip non-serializable objects (model objects, etc.)
            return None
    
    def run_full_comparison(self, X_train, y_train, X_test, y_test, cv_folds=5, feature_names=None):
        """
        Run complete model comparison pipeline with visualizations and reports
        """
        print("\n" + "="*60)
        print("🔬 COMPLETE MODEL COMPARISON")
        print("="*60)
        
        try:
            # Step 1: Define models
            self.define_models()
            
            # Step 2: Cross-validate
            self.cross_validate_models(X_train, y_train, cv_folds=cv_folds)
            
            # Step 3: Statistical tests
            statistical_results = self.statistical_comparison()
            
            # Step 4: Test set evaluation
            test_results = self.compare_on_test_set(X_test, y_test)
            
            # Step 5: Ensemble methods
            ensemble_results = self.create_ensemble_models(X_train, y_train, X_test, y_test)
            
            # Step 6: Select best
            best_name, best_model = self.select_best_model(X_train, y_train, X_test, y_test)
            
            # Step 7: Generate visualizations
            print("\n📊 Generating visualizations...")
            # Get absolute path for cross-platform compatibility
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            reports_dir = os.path.join(project_root, 'artifacts', 'reports')
            visualizer = ModelVisualizer(output_dir=reports_dir)
            
            plot_files = {
                'CV Scores - Box Plot': visualizer.plot_cv_scores_box(self.cv_results),
                'CV Scores - Violin Plot': visualizer.plot_cv_scores_violin(self.cv_results),
                'Metrics Comparison': visualizer.plot_metrics_comparison(self.cv_results),
                'Test Set Comparison': visualizer.plot_test_set_comparison(test_results),
                'Ensemble vs Individual': visualizer.plot_ensemble_vs_individual(
                    self.cv_results, ensemble_results, test_results
                ),
                'Residuals Analysis': visualizer.plot_residuals_analysis(
                    self.models, X_test, y_test
                ),
                'Actual vs Predicted': visualizer.plot_actual_vs_predicted(
                    self.models, X_test, y_test
                ),
                'Residuals Distribution': visualizer.plot_residuals_distribution(
                    self.models, X_test, y_test
                ),
                'Q-Q Plots': visualizer.plot_qq_plots(
                    self.models, X_test, y_test
                )
            }
            
            # Add feature importance if feature names provided
            if feature_names is not None:
                plot_files['Feature Importance'] = visualizer.plot_feature_importance(
                    self.models, feature_names
                )
            
            print("✓ Visualizations created")
            
            # Step 8: Generate comprehensive report
            print("\n📄 Generating HTML report...")
            artifacts_dir = os.path.join(project_root, 'artifacts')
            report_generator = HTMLReportGenerator(output_dir=artifacts_dir)
            
            # Filter out None plot files
            plot_files = {k: v for k, v in plot_files.items() if v is not None}
            
            html_report_path = report_generator.save_report(
                self.cv_results,
                statistical_results,
                test_results,
                ensemble_results,
                best_name,
                plot_files
            )
            
            # Step 9: Generate JSON report
            report = self.generate_comparison_report(test_results, ensemble_results, statistical_results)
            
            # Step 10: Train best model on full data
            print("\nTraining best model on full dataset...")
            X_full = np.vstack((X_train, X_test))
            y_full = np.hstack((y_train, y_test))
            best_model.fit(X_full, y_full)
            
            # Save best model
            model_path = os.path.join(project_root, 'artifacts', 'model.pkl')
            save_object(model_path, best_model)
            logging.info(f"Best model ({best_name}) trained and saved")
            
            print("\n" + "="*60)
            print("✅ MODEL COMPARISON COMPLETE")
            print("="*60)
            html_report_path = os.path.join(artifacts_dir, 'model_comparison_report.html')
            json_report_path = os.path.join(reports_dir, 'model_comparison_report.json')
            print(f"📄 HTML Report: {html_report_path}")
            print(f"📊 JSON Report: {json_report_path}")
            print(f"🤖 Best Model: {model_path}")
            print("="*60)
            
            return best_model, best_name, report
            
        except Exception as e:
            logging.error(f"Error in model comparison: {str(e)}")
            raise Custom_Exception(e, sys)
