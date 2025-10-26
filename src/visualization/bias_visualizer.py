import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from pathlib import Path
import json
from typing import Dict, Any, List
import warnings
warnings.filterwarnings('ignore')

from ..utils import load_config, save_results

class BiasVisualizer:
    """Create visualizations for gender bias analysis results."""
    
    def __init__(self):
        self.config = load_config()
        
        # Setup directories
        self.results_dir = Path("results")
        self.viz_dir = self.results_dir / "visualizations"
        self.viz_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_data(self) -> tuple:
        """Load analysis data and bias results."""
        # Load gender analysis results
        analysis_df = pd.read_csv("results/gender_analysis_filtered.csv")
        
        # Load bias analysis results
        with open("results/bias_analysis_complete.json", 'r') as f:
            bias_results = json.load(f)
        
        return analysis_df, bias_results
    
    def create_gender_distribution_pie(self, df: pd.DataFrame) -> None:
        """Create pie chart of gender distribution."""
        gender_counts = df['predicted_gender'].value_counts()
        
        # Matplotlib version
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Pie chart
        colors = ['#FF6B9D', '#4ECDC4']
        wedges, texts, autotexts = ax1.pie(
            gender_counts.values, 
            labels=gender_counts.index,
            autopct='%1.1f%%',
            colors=colors,
            explode=(0.05, 0.05),
            shadow=True
        )
        ax1.set_title('Gender Distribution in Dataset', fontsize=14, fontweight='bold')
        
        # Bar chart
        bars = ax2.bar(gender_counts.index, gender_counts.values, color=colors, alpha=0.8)
        ax2.set_title('Gender Count Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Faces')
        ax2.set_xlabel('Predicted Gender')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "gender_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plotly interactive version
        fig_plotly = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "pie"}, {"type": "bar"}]],
            subplot_titles=("Gender Distribution", "Gender Counts")
        )
        
        # Pie chart
        fig_plotly.add_trace(
            go.Pie(
                labels=gender_counts.index,
                values=gender_counts.values,
                name="Gender Distribution",
                marker_colors=colors
            ),
            row=1, col=1
        )
        
        # Bar chart
        fig_plotly.add_trace(
            go.Bar(
                x=gender_counts.index,
                y=gender_counts.values,
                name="Gender Counts",
                marker_color=colors
            ),
            row=1, col=2
        )
        
        fig_plotly.update_layout(
            title_text="Gender Distribution Analysis",
            title_x=0.5,
            showlegend=False,
            height=500
        )
        
        fig_plotly.write_html(self.viz_dir / "gender_distribution_interactive.html")
        
        self.logger.info("Gender distribution visualizations created")
    
    def create_confidence_histogram(self, df: pd.DataFrame) -> None:
        """Create histogram of confidence scores by gender."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Gender confidence by gender
        for i, gender in enumerate(['male', 'female']):
            gender_data = df[df['predicted_gender'] == gender]['gender_confidence']
            axes[0, i].hist(gender_data, bins=20, alpha=0.7, color=['#4ECDC4', '#FF6B9D'][i])
            axes[0, i].set_title(f'Gender Confidence Distribution - {gender.title()}')
            axes[0, i].set_xlabel('Confidence Score')
            axes[0, i].set_ylabel('Frequency')
            axes[0, i].axvline(gender_data.mean(), color='red', linestyle='--', 
                              label=f'Mean: {gender_data.mean():.3f}')
            axes[0, i].legend()
        
        # Combined gender confidence
        axes[1, 0].hist(df[df['predicted_gender'] == 'male']['gender_confidence'], 
                       bins=20, alpha=0.6, label='Male', color='#4ECDC4')
        axes[1, 0].hist(df[df['predicted_gender'] == 'female']['gender_confidence'], 
                       bins=20, alpha=0.6, label='Female', color='#FF6B9D')
        axes[1, 0].set_title('Gender Confidence Distribution - Comparison')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Detection confidence
        sns.boxplot(data=df, x='predicted_gender', y='detection_confidence', ax=axes[1, 1])
        axes[1, 1].set_title('Detection Confidence by Gender')
        axes[1, 1].set_xlabel('Predicted Gender')
        axes[1, 1].set_ylabel('Detection Confidence')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "confidence_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Interactive Plotly version
        fig_plotly = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "Male Gender Confidence", "Female Gender Confidence",
                "Confidence Comparison", "Detection Confidence by Gender"
            )
        )
        
        # Individual histograms
        for i, gender in enumerate(['male', 'female']):
            gender_data = df[df['predicted_gender'] == gender]['gender_confidence']
            fig_plotly.add_trace(
                go.Histogram(
                    x=gender_data,
                    name=f'{gender.title()} Confidence',
                    marker_color=['#4ECDC4', '#FF6B9D'][i],
                    opacity=0.7
                ),
                row=1, col=i+1
            )
        
        # Overlapped histograms
        for i, gender in enumerate(['male', 'female']):
            gender_data = df[df['predicted_gender'] == gender]['gender_confidence']
            fig_plotly.add_trace(
                go.Histogram(
                    x=gender_data,
                    name=f'{gender.title()}',
                    marker_color=['#4ECDC4', '#FF6B9D'][i],
                    opacity=0.6
                ),
                row=2, col=1
            )
        
        # Box plot
        for i, gender in enumerate(['male', 'female']):
            gender_data = df[df['predicted_gender'] == gender]['detection_confidence']
            fig_plotly.add_trace(
                go.Box(
                    y=gender_data,
                    name=gender.title(),
                    marker_color=['#4ECDC4', '#FF6B9D'][i]
                ),
                row=2, col=2
            )
        
        fig_plotly.update_layout(
            title_text="Confidence Score Analysis",
            title_x=0.5,
            height=800,
            showlegend=False
        )
        
        fig_plotly.write_html(self.viz_dir / "confidence_analysis_interactive.html")
        
        self.logger.info("Confidence analysis visualizations created")
    
    def create_age_gender_heatmap(self, df: pd.DataFrame) -> None:
        """Create heatmap of age-gender intersection."""
        if 'predicted_age_range' not in df.columns or df['predicted_age_range'].isna().all():
            self.logger.warning("Age data not available for heatmap")
            return
        
        # Create cross-tabulation
        crosstab = pd.crosstab(df['predicted_age_range'], df['predicted_gender'])
        crosstab_pct = pd.crosstab(df['predicted_age_range'], df['predicted_gender'], normalize='index') * 100
        
        # Matplotlib heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Count heatmap
        sns.heatmap(crosstab, annot=True, fmt='d', cmap='YlOrRd', ax=ax1)
        ax1.set_title('Age-Gender Distribution (Counts)')
        ax1.set_xlabel('Predicted Gender')
        ax1.set_ylabel('Predicted Age Range')
        
        # Percentage heatmap
        sns.heatmap(crosstab_pct, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=ax2)
        ax2.set_title('Age-Gender Distribution (Percentage within Age Group)')
        ax2.set_xlabel('Predicted Gender')
        ax2.set_ylabel('Predicted Age Range')
        
        plt.tight_layout()
        plt.savefig(self.viz_dir / "age_gender_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Interactive Plotly heatmap
        fig_plotly = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Count Distribution", "Percentage Distribution")
        )
        
        # Count heatmap
        fig_plotly.add_trace(
            go.Heatmap(
                z=crosstab.values,
                x=crosstab.columns,
                y=crosstab.index,
                colorscale='YlOrRd',
                text=crosstab.values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hoverongaps=False
            ),
            row=1, col=1
        )
        
        # Percentage heatmap
        fig_plotly.add_trace(
            go.Heatmap(
                z=crosstab_pct.values,
                x=crosstab_pct.columns,
                y=crosstab_pct.index,
                colorscale='RdYlBu_r',
                text=crosstab_pct.round(1).values,
                texttemplate="%{text}%",
                textfont={"size": 10},
                hoverongaps=False
            ),
            row=1, col=2
        )
        
        fig_plotly.update_layout(
            title_text="Age-Gender Distribution Analysis",
            title_x=0.5,
            height=600
        )
        
        fig_plotly.write_html(self.viz_dir / "age_gender_heatmap_interactive.html")
        
        self.logger.info("Age-gender heatmap created")
    
    def create_bias_metrics_dashboard(self, bias_results: Dict[str, Any]) -> None:
        """Create comprehensive bias metrics dashboard."""
        # Extract key metrics
        bias_assessment = bias_results['bias_assessment']
        representation = bias_results['representation_ratio']
        confidence_dist = bias_results['confidence_distribution']
        
        # Create dashboard
        fig = make_subplots(
            rows=3, cols=2,
            specs=[
                [{"type": "indicator"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"colspan": 2}, None]
            ],
            subplot_titles=(
                "Overall Bias Score", "Gender Representation",
                "Confidence by Gender", "Bias Indicators",
                "Recommendations"
            ),
            vertical_spacing=0.12
        )
        
        # Bias score indicator
        bias_score = bias_assessment['bias_percentage']
        color = "red" if bias_score > 50 else "orange" if bias_score > 25 else "green"
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=bias_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Bias Score (%)"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': color},
                    'steps': [
                        {'range': [0, 25], 'color': "lightgreen"},
                        {'range': [25, 50], 'color': "yellow"},
                        {'range': [50, 100], 'color': "lightcoral"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ),
            row=1, col=1
        )
        
        # Gender representation
        fig.add_trace(
            go.Bar(
                x=['Male', 'Female'],
                y=[representation['male_percentage'], representation['female_percentage']],
                marker_color=['#4ECDC4', '#FF6B9D'],
                name="Gender %"
            ),
            row=1, col=2
        )
        
        # Confidence by gender
        if 'male' in confidence_dist and 'female' in confidence_dist:
            fig.add_trace(
                go.Bar(
                    x=['Male Confidence', 'Female Confidence'],
                    y=[confidence_dist['male'].get('mean', 0), confidence_dist['female'].get('mean', 0)],
                    marker_color=['#4ECDC4', '#FF6B9D'],
                    name="Avg Confidence"
                ),
                row=2, col=1
            )
        
        # Bias indicators scatter
        indicators = bias_assessment['bias_indicators']
        if indicators:
            severity_map = {'low': 1, 'moderate': 2, 'high': 3}
            x_vals = list(range(len(indicators)))
            y_vals = [severity_map.get(ind['severity'], 1) for ind in indicators]
            hover_text = [ind['description'] for ind in indicators]
            
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=y_vals,
                    mode='markers',
                    marker=dict(
                        size=15,
                        color=y_vals,
                        colorscale='Reds',
                        showscale=False
                    ),
                    text=hover_text,
                    hovertemplate='%{text}<extra></extra>',
                    name="Bias Indicators"
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title_text="Gender Bias Analysis Dashboard",
            title_x=0.5,
            height=1000,
            showlegend=False
        )
        
        # Add annotations for recommendations
        recommendations_text = "<br>".join([f"• {rec}" for rec in bias_assessment['recommendations']])
        fig.add_annotation(
            text=f"<b>Recommendations:</b><br>{recommendations_text}",
            xref="paper", yref="paper",
            x=0.5, y=0.15,
            showarrow=False,
            font=dict(size=12),
            align="left",
            bgcolor="lightblue",
            bordercolor="blue",
            borderwidth=1
        )
        
        fig.write_html(self.viz_dir / "bias_dashboard.html")
        
        self.logger.info("Bias metrics dashboard created")
    
    def create_summary_report(self, df: pd.DataFrame, bias_results: Dict[str, Any]) -> None:
        """Create a comprehensive HTML summary report."""
        # Extract key statistics
        total_faces = len(df)
        gender_dist = df['predicted_gender'].value_counts()
        bias_score = bias_results['bias_assessment']['bias_percentage']
        bias_level = bias_results['bias_assessment']['bias_level']
        
        # Create HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Gender Bias Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 10px; }}
                .metric {{ background-color: #e8f4fd; padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .warning {{ background-color: #fff3cd; border: 1px solid #ffeaa7; }}
                .success {{ background-color: #d4edda; border: 1px solid #c3e6cb; }}
                .danger {{ background-color: #f8d7da; border: 1px solid #f5c6cb; }}
                .chart {{ text-align: center; margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Gender Bias Analysis Report</h1>
                <p><strong>Analysis Date:</strong> {bias_results['metadata']['analysis_date']}</p>
                <p><strong>Total Faces Analyzed:</strong> {total_faces:,}</p>
            </div>
            
            <div class="metric {'success' if bias_level == 'low' else 'warning' if bias_level == 'moderate' else 'danger'}">
                <h2>Overall Bias Assessment</h2>
                <p><strong>Bias Level:</strong> {bias_level.upper()}</p>
                <p><strong>Bias Score:</strong> {bias_score:.1f}% (out of 100%)</p>
            </div>
            
            <div class="metric">
                <h2>Gender Distribution</h2>
                <p><strong>Male:</strong> {gender_dist.get('male', 0):,} ({gender_dist.get('male', 0)/total_faces*100:.1f}%)</p>
                <p><strong>Female:</strong> {gender_dist.get('female', 0):,} ({gender_dist.get('female', 0)/total_faces*100:.1f}%)</p>
            </div>
            
            <div class="metric">
                <h2>Key Findings</h2>
                <ul>
        """
        
        for indicator in bias_results['bias_assessment']['bias_indicators']:
            html_content += f"<li><strong>{indicator['severity'].title()} {indicator['type'].replace('_', ' ').title()}:</strong> {indicator['description']}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <div class="metric">
                <h2>Recommendations</h2>
                <ul>
        """
        
        for rec in bias_results['bias_assessment']['recommendations']:
            html_content += f"<li>{rec}</li>"
        
        html_content += """
                </ul>
            </div>
            
            <div class="metric">
                <h2>Visualizations</h2>
                <p>Interactive visualizations have been generated:</p>
                <ul>
                    <li><a href="gender_distribution_interactive.html">Gender Distribution</a></li>
                    <li><a href="confidence_analysis_interactive.html">Confidence Analysis</a></li>
                    <li><a href="age_gender_heatmap_interactive.html">Age-Gender Heatmap</a></li>
                    <li><a href="bias_dashboard.html">Bias Dashboard</a></li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(self.viz_dir / "bias_analysis_report.html", 'w') as f:
            f.write(html_content)
        
        self.logger.info("Summary report created")
    
    def generate_all_visualizations(self) -> None:
        """Generate all visualizations and reports."""
        self.logger.info("Starting visualization generation...")
        
        try:
            # Load data
            df, bias_results = self.load_data()
            
            # Create all visualizations
            self.create_gender_distribution_pie(df)
            self.create_confidence_histogram(df)
            self.create_age_gender_heatmap(df)
            self.create_bias_metrics_dashboard(bias_results)
            self.create_summary_report(df, bias_results)
            
            self.logger.info("All visualizations generated successfully!")
            
            # Create index file
            self.create_visualization_index()
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {str(e)}")
            raise
    
    def create_visualization_index(self) -> None:
        """Create an index HTML file linking to all visualizations."""
        index_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Gender Bias Analysis - Visualization Index</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .container { max-width: 800px; margin: 0 auto; }
                .viz-link { 
                    display: block; 
                    padding: 15px; 
                    margin: 10px 0; 
                    background-color: #f8f9fa; 
                    border-radius: 5px; 
                    text-decoration: none; 
                    color: #333;
                    border-left: 4px solid #007bff;
                }
                .viz-link:hover { background-color: #e9ecef; }
                h1 { color: #333; text-align: center; }
                h2 { color: #666; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Gender Bias Analysis Visualizations</h1>
                
                <h2>📊 Main Report</h2>
                <a href="bias_analysis_report.html" class="viz-link">
                    <strong>Comprehensive Bias Analysis Report</strong><br>
                    <small>Complete overview with findings and recommendations</small>
                </a>
                
                <h2>📈 Interactive Dashboards</h2>
                <a href="bias_dashboard.html" class="viz-link">
                    <strong>Bias Metrics Dashboard</strong><br>
                    <small>Interactive dashboard with key bias indicators</small>
                </a>
                
                <h2>📋 Detailed Visualizations</h2>
                <a href="gender_distribution_interactive.html" class="viz-link">
                    <strong>Gender Distribution Analysis</strong><br>
                    <small>Interactive pie charts and bar graphs of gender representation</small>
                </a>
                
                <a href="confidence_analysis_interactive.html" class="viz-link">
                    <strong>Confidence Score Analysis</strong><br>
                    <small>Histograms and box plots of model confidence by gender</small>
                </a>
                
                <a href="age_gender_heatmap_interactive.html" class="viz-link">
                    <strong>Age-Gender Intersection Heatmap</strong><br>
                    <small>Cross-analysis of age and gender predictions</small>
                </a>
                
                <h2>🖼️ Static Images</h2>
                <a href="gender_distribution.png" class="viz-link">
                    <strong>Gender Distribution (PNG)</strong><br>
                    <small>Static pie chart and bar graph</small>
                </a>
                
                <a href="confidence_analysis.png" class="viz-link">
                    <strong>Confidence Analysis (PNG)</strong><br>
                    <small>Static confidence distribution charts</small>
                </a>
                
                <a href="age_gender_heatmap.png" class="viz-link">
                    <strong>Age-Gender Heatmap (PNG)</strong><br>
                    <small>Static heatmap visualization</small>
                </a>
            </div>
        </body>
        </html>
        """
        
        with open(self.viz_dir / "index.html", 'w') as f:
            f.write(index_html)
        
        self.logger.info("Visualization index created")

def main():
    """Main function to generate all visualizations."""
    try:
        visualizer = BiasVisualizer()
        visualizer.generate_all_visualizations()
        print("All visualizations generated successfully!")
        print(f"Open {visualizer.viz_dir}/index.html to view all results")
    except Exception as e:
        logging.error(f"Visualization generation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()