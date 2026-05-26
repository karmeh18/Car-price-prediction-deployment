"""
HTML Report Generator for Model Comparison
Creates professional, interactive HTML reports with all results
"""

import os
import json
import base64
from datetime import datetime
from src.logger import logging


class HTMLReportGenerator:
    """Generate comprehensive HTML reports for model comparison"""
    
    def __init__(self, output_dir='artifacts'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def encode_image_to_base64(self, image_path):
        """Convert image file to base64 for embedding in HTML"""
        if not os.path.exists(image_path):
            return None
        
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode()
    
    def generate_cv_section(self, cv_results):
        """Generate cross-validation results section"""
        html = '<section id="cv-section" class="section">\n'
        html += '<h2>📊 Cross-Validation Results</h2>\n'
        html += '<p>5-fold cross-validation results showing model generalization:</p>\n'
        html += '<table class="results-table">\n'
        html += '<thead><tr><th>Model</th><th>MAPE Mean (%)</th><th>MAPE Std (%)</th><th>MAE Mean ($)</th><th>R² Mean</th><th>Train R² Mean</th></tr></thead>\n'
        html += '<tbody>\n'
        
        for model_name, results in sorted(cv_results.items(), key=lambda x: x[1]['mape_mean']):
            html += f'<tr>\n'
            html += f'  <td><strong>{model_name}</strong></td>\n'
            html += f'  <td>{results["mape_mean"]:.2f}</td>\n'
            html += f'  <td>{results["mape_std"]:.2f}</td>\n'
            html += f'  <td>${results["mae_mean"]:.2f}</td>\n'
            html += f'  <td>{results["r2_mean"]:.4f}</td>\n'
            html += f'  <td>{results["train_r2_mean"]:.4f}</td>\n'
            html += f'</tr>\n'
        
        html += '</tbody>\n</table>\n'
        html += '</section>\n'
        return html
    
    def generate_statistical_tests_section(self, statistical_results):
        """Generate statistical tests section"""
        html = '<section id="stats-section" class="section">\n'
        html += '<h2>📈 Statistical Significance Tests</h2>\n'
        
        # Extract paired t-test results (vs ANOVA, Kruskal-Wallis, etc.)
        paired_tests = {k: v for k, v in statistical_results.items() 
                       if k not in ['ANOVA', 'Kruskal-Wallis', 'confidence_intervals'] and 'cohens_d' in v}
        
        if paired_tests:
            html += '<p>Paired t-tests to determine if differences between models are statistically significant:</p>\n'
            html += '<table class="results-table">\n'
            html += '<thead><tr><th>Comparison</th><th>T-Statistic</th><th>P-Value</th><th>Significant</th><th>Cohen\'s d</th><th>Winner</th></tr></thead>\n'
            html += '<tbody>\n'
            
            for comparison, results in sorted(paired_tests.items()):
                significance = '✅ Yes (p < 0.05)' if results['significant'] else '❌ No (p ≥ 0.05)'
                cohen_d_interp = StatisticalTests.interpret_cohens_d(abs(results['cohens_d']))
                
                html += f'<tr>\n'
                html += f'  <td>{comparison}</td>\n'
                html += f'  <td>{results["t_statistic"]:.4f}</td>\n'
                html += f'  <td>{results["p_value"]:.4f}</td>\n'
                html += f'  <td>{significance}</td>\n'
                html += f'  <td>{results["cohens_d"]:.3f} ({cohen_d_interp})</td>\n'
                html += f'  <td><strong>{results["winner"]}</strong></td>\n'
                html += f'</tr>\n'
            
            html += '</tbody>\n</table>\n'
            html += '<p><small>Significance level: α = 0.05. Cohen\'s d interpretation: <0.2 = Small, 0.2-0.5 = Small-Medium, 0.5-0.8 = Medium-Large, >0.8 = Large</small></p>\n'
        
        # ANOVA results
        if 'ANOVA' in statistical_results:
            anova = statistical_results['ANOVA']
            html += '<h3>One-Way ANOVA Test</h3>\n'
            html += '<table class="results-table">\n'
            html += '<thead><tr><th>Metric</th><th>Value</th></tr></thead>\n'
            html += '<tbody>\n'
            html += f'<tr><td>F-Statistic</td><td>{anova["f_statistic"]:.4f}</td></tr>\n'
            html += f'<tr><td>P-Value</td><td>{anova["p_value"]:.4f}</td></tr>\n'
            html += f'<tr><td>Significant (α=0.05)</td><td>{"✅ Yes" if anova["significant"] else "❌ No"}</td></tr>\n'
            html += f'<tr><td>Effect Size (η²)</td><td>{anova["eta_squared"]:.4f}</td></tr>\n'
            html += '</tbody>\n</table>\n'
        
        # Kruskal-Wallis results
        if 'Kruskal-Wallis' in statistical_results:
            kw = statistical_results['Kruskal-Wallis']
            html += '<h3>Kruskal-Wallis Test (Non-parametric)</h3>\n'
            html += '<table class="results-table">\n'
            html += '<thead><tr><th>Metric</th><th>Value</th></tr></thead>\n'
            html += '<tbody>\n'
            html += f'<tr><td>H-Statistic</td><td>{kw["h_statistic"]:.4f}</td></tr>\n'
            html += f'<tr><td>P-Value</td><td>{kw["p_value"]:.4f}</td></tr>\n'
            html += f'<tr><td>Significant (α=0.05)</td><td>{"✅ Yes" if kw["significant"] else "❌ No"}</td></tr>\n'
            html += '</tbody>\n</table>\n'
        
        html += '</section>\n'
        return html
    
    def generate_test_set_section(self, test_results):
        """Generate test set evaluation section"""
        html = '<section id="test-section" class="section">\n'
        html += '<h2>🎯 Test Set Performance</h2>\n'
        html += '<table class="results-table">\n'
        html += '<thead><tr><th>Model</th><th>MAPE (%)</th><th>MAE ($)</th><th>RMSE ($)</th><th>R² Score</th></tr></thead>\n'
        html += '<tbody>\n'
        
        for model_name, results in sorted(test_results.items(), key=lambda x: x[1]['mape']):
            html += f'<tr>\n'
            html += f'  <td><strong>{model_name}</strong></td>\n'
            html += f'  <td>{results["mape"]:.2f}</td>\n'
            html += f'  <td>${results["mae"]:.2f}</td>\n'
            html += f'  <td>${results["rmse"]:.2f}</td>\n'
            html += f'  <td>{results["r2"]:.4f}</td>\n'
            html += f'</tr>\n'
        
        html += '</tbody>\n</table>\n'
        html += '</section>\n'
        return html
    
    def generate_ensemble_section(self, ensemble_results):
        """Generate ensemble methods section"""
        html = '<section id="ensemble-section" class="section">\n'
        html += '<h2>🤝 Ensemble Methods</h2>\n'
        html += '<table class="results-table">\n'
        html += '<thead><tr><th>Ensemble Type</th><th>MAPE (%)</th><th>R² Score</th><th>Description</th></tr></thead>\n'
        html += '<tbody>\n'
        
        descriptions = {
            'Voting': 'Average predictions from all base models',
            'Stacking': 'Meta-learner (Ridge) combines predictions from base models'
        }
        
        for ensemble_name, results in ensemble_results.items():
            desc = descriptions.get(ensemble_name, '')
            html += f'<tr>\n'
            html += f'  <td><strong>{ensemble_name}</strong></td>\n'
            html += f'  <td>{results["mape"]:.2f}</td>\n'
            html += f'  <td>{results["r2"]:.4f}</td>\n'
            html += f'  <td>{desc}</td>\n'
            html += f'</tr>\n'
        
        html += '</tbody>\n</table>\n'
        html += '</section>\n'
        return html
    
    def generate_visualizations_section(self, plot_files):
        """Generate visualizations section with embedded images"""
        html = '<section id="viz-section" class="section">\n'
        html += '<h2>📉 Visualizations</h2>\n'
        
        for plot_name, plot_path in plot_files.items():
            if not os.path.exists(plot_path):
                continue
            
            base64_img = self.encode_image_to_base64(plot_path)
            if base64_img:
                html += f'<div class="plot-container">\n'
                html += f'<h3>{plot_name}</h3>\n'
                html += f'<img src="data:image/png;base64,{base64_img}" alt="{plot_name}" class="plot-image">\n'
                html += f'</div>\n'
        
        html += '</section>\n'
        return html
    
    def generate_html_content(self, cv_results, statistical_results, test_results, 
                            ensemble_results, best_model_name, plot_files):
        """Generate complete HTML document"""
        
        html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Comparison Report - Car Price Prediction</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }
        
        header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        nav {
            background: #f8f9fa;
            padding: 15px 40px;
            border-bottom: 1px solid #dee2e6;
            display: flex;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        nav a {
            color: #667eea;
            text-decoration: none;
            font-weight: 500;
            transition: color 0.3s;
        }
        
        nav a:hover {
            color: #764ba2;
        }
        
        .content {
            padding: 40px;
        }
        
        .section {
            margin-bottom: 40px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 5px solid #667eea;
        }
        
        .section h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        
        .section h3 {
            color: #764ba2;
            margin-top: 20px;
            margin-bottom: 10px;
        }
        
        .section p {
            margin-bottom: 15px;
            color: #555;
        }
        
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 4px;
            overflow: hidden;
        }
        
        .results-table thead {
            background: #667eea;
            color: white;
        }
        
        .results-table th,
        .results-table td {
            padding: 15px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }
        
        .results-table tbody tr:hover {
            background: #f0f0f0;
        }
        
        .results-table th {
            font-weight: 600;
        }
        
        .plot-container {
            margin: 30px 0;
            text-align: center;
        }
        
        .plot-image {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        
        .summary-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        .summary-box h3 {
            color: white;
            font-size: 1.5em;
            margin-bottom: 15px;
        }
        
        .summary-item {
            margin: 10px 0;
            font-size: 1.1em;
        }
        
        .metric {
            display: inline-block;
            margin-right: 30px;
        }
        
        .metric-value {
            font-size: 1.3em;
            font-weight: bold;
        }
        
        footer {
            background: #f8f9fa;
            padding: 20px 40px;
            text-align: center;
            border-top: 1px solid #dee2e6;
            color: #999;
        }
        
        .recommendation {
            background: #d4edda;
            border-left: 5px solid #28a745;
            padding: 20px;
            border-radius: 4px;
            margin: 20px 0;
        }
        
        .recommendation h4 {
            color: #155724;
            margin-bottom: 10px;
        }
        
        .recommendation p {
            color: #155724;
        }
        
        small {
            display: block;
            margin-top: 10px;
            color: #777;
        }
        
        @media (max-width: 768px) {
            .container {
                margin: 0;
                border-radius: 0;
            }
            
            header h1 {
                font-size: 1.8em;
            }
            
            nav {
                flex-direction: column;
                gap: 10px;
            }
            
            .content {
                padding: 20px;
            }
            
            .results-table th,
            .results-table td {
                padding: 10px;
                font-size: 0.9em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
"""
        
        html += f"""        <header>
            <h1>🚗 Car Price Prediction - Model Comparison Report</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>
"""
        
        html += """        <nav>
            <a href="#cv-section">📊 Cross-Validation</a>
            <a href="#stats-section">📈 Statistical Tests</a>
            <a href="#test-section">🎯 Test Set</a>
            <a href="#ensemble-section">🤝 Ensembles</a>
            <a href="#viz-section">📉 Visualizations</a>
        </nav>
"""
        
        html += """        <div class="content">
"""
        
        # Summary box
        html += f"""            <div class="summary-box">
                <h3>🏆 Best Model Selected</h3>
                <div class="summary-item">
                    <strong>Model:</strong> {best_model_name}
                </div>
                <div class="summary-item">
                    Best individual model based on 5-fold cross-validation MAPE score
                </div>
            </div>
"""
        
        # Sections
        html += self.generate_cv_section(cv_results)
        html += self.generate_statistical_tests_section(statistical_results)
        html += self.generate_test_set_section(test_results)
        html += self.generate_ensemble_section(ensemble_results)
        html += self.generate_visualizations_section(plot_files)
        
        # Recommendation
        html += """            <section class="section">
                <h2>💡 Recommendations</h2>
                <div class="recommendation">
                    <h4>Model Selection Strategy</h4>
                    <p>✅ Use the best model for production deployment<br>
                       ✅ Monitor actual vs predicted values in production<br>
                       ✅ Retrain quarterly or when performance degrades<br>
                       ✅ Consider ensemble methods for improved accuracy</p>
                </div>
            </section>
"""
        
        html += """        </div>
        
        <footer>
            <p>Car Price Prediction ML Project | Model Comparison Framework | All rights reserved</p>
        </footer>
    </div>
</body>
</html>
"""
        
        return html
    
    def save_report(self, cv_results, statistical_results, test_results, 
                   ensemble_results, best_model_name, plot_files):
        """Generate and save HTML report"""
        
        html_content = self.generate_html_content(
            cv_results, statistical_results, test_results, 
            ensemble_results, best_model_name, plot_files
        )
        
        report_path = os.path.join(self.output_dir, 'model_comparison_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logging.info(f"HTML report saved: {report_path}")
        print(f"📄 HTML Report saved: {report_path}")
        
        return report_path


class StatisticalTests:
    """Helper for statistical interpretation"""
    
    @staticmethod
    def interpret_cohens_d(d):
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "Small"
        elif d < 0.5:
            return "Small-Medium"
        elif d < 0.8:
            return "Medium-Large"
        else:
            return "Large"
