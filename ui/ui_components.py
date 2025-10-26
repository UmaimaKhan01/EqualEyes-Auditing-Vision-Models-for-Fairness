"""
UI Components for Gender Bias Analysis Pipeline
Reusable Streamlit components for the web interface
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import base64
from PIL import Image
import io

def render_metric_card(title, value, delta=None, delta_color="normal"):
    """Render a custom metric card with styling"""
    delta_html = ""
    if delta is not None:
        color = "#ff4b4b" if delta_color == "inverse" else "#09ab3b"
        delta_html = f'<div style="color: {color}; font-size: 0.8rem;">{"▲" if delta > 0 else "▼"} {abs(delta):.2f}</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 0.8rem; color: #666;">{title}</div>
        <div style="font-size: 1.5rem; font-weight: bold; margin: 0.2rem 0;">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

def render_status_indicator(stage_name, is_complete, is_running=False):
    """Render a status indicator for pipeline stages"""
    if is_running:
        icon = "🔄"
        status = "Running"
        color = "#ff9500"
    elif is_complete:
        icon = "✅"
        status = "Complete"
        color = "#09ab3b"
    else:
        icon = "⏳"
        status = "Pending"
        color = "#666"
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; padding: 0.5rem; margin: 0.2rem 0;">
        <span style="font-size: 1.2rem; margin-right: 0.5rem;">{icon}</span>
        <div>
            <div style="font-weight: bold;">{stage_name}</div>
            <div style="color: {color}; font-size: 0.8rem;">{status}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_bias_alert(bias_score, title="Bias Assessment"):
    """Render bias assessment alert with appropriate styling"""
    if bias_score > 0.3:
        alert_class = "bias-high"
        icon = "⚠️"
        level = "High Bias Detected"
        description = "Significant gender imbalance found in the dataset."
    elif bias_score > 0.15:
        alert_class = "bias-medium"
        icon = "⚡"
        level = "Moderate Bias Detected"
        description = "Some gender imbalance present in the dataset."
    else:
        alert_class = "bias-low"
        icon = "✅"
        level = "Low Bias Detected"
        description = "Dataset shows relatively balanced gender representation."
    
    st.markdown(f"""
    <div class="bias-alert {alert_class}">
        <strong>{icon} {level}</strong><br>
        Bias Score: {bias_score:.3f}<br>
        {description}
    </div>
    """, unsafe_allow_html=True)

def create_gender_distribution_chart(gender_data, chart_type="pie"):
    """Create gender distribution visualization"""
    if chart_type == "pie":
        fig = px.pie(
            values=list(gender_data.values()),
            names=list(gender_data.keys()),
            title="Gender Distribution",
            color_discrete_sequence=['#FF6B9D', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            font=dict(size=12),
            title_font_size=16,
            margin=dict(t=50, b=50, l=50, r=50)
        )
    else:  # bar chart
        fig = px.bar(
            x=list(gender_data.keys()),
            y=list(gender_data.values()),
            title="Gender Distribution",
            color=list(gender_data.keys()),
            color_discrete_sequence=['#FF6B9D', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        )
        fig.update_layout(
            xaxis_title="Gender",
            yaxis_title="Count",
            font=dict(size=12),
            title_font_size=16,
            margin=dict(t=50, b=50, l=50, r=50)
        )
    
    return fig

def create_confidence_histogram(confidence_data, threshold=0.7):
    """Create confidence distribution histogram"""
    fig = px.histogram(
        x=confidence_data,
        nbins=30,
        title="Classification Confidence Distribution",
        labels={'x': 'Confidence Score', 'y': 'Frequency'},
        color_discrete_sequence=['#667eea']
    )
    
    # Add threshold line
    fig.add_vline(
        x=threshold, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Threshold: {threshold}",
        annotation_position="top right"
    )
    
    # Add statistics annotations
    mean_conf = np.mean(confidence_data)
    fig.add_vline(
        x=mean_conf,
        line_dash="dot",
        line_color="green",
        annotation_text=f"Mean: {mean_conf:.3f}",
        annotation_position="top left"
    )
    
    fig.update_layout(
        font=dict(size=12),
        title_font_size=16,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    
    return fig

def create_bias_trend_chart(dates, bias_scores):
    """Create bias trend over time chart"""
    fig = px.line(
        x=dates,
        y=bias_scores,
        title="Bias Score Trends Over Time",
        labels={'x': 'Date', 'y': 'Bias Score'},
        line_shape='spline'
    )
    
    # Add threshold lines
    fig.add_hline(y=0.3, line_dash="dash", line_color="red",
                 annotation_text="High Bias Threshold")
    fig.add_hline(y=0.15, line_dash="dash", line_color="orange",
                 annotation_text="Medium Bias Threshold")
    
    # Color the line based on bias level
    colors = ['red' if score > 0.3 else 'orange' if score > 0.15 else 'green' 
              for score in bias_scores]
    
    fig.update_traces(line_color='#667eea', line_width=3)
    fig.update_layout(
        font=dict(size=12),
        title_font_size=16,
        margin=dict(t=50, b=50, l=50, r=50),
        hovermode='x unified'
    )
    
    return fig

def create_age_gender_matrix(age_data, gender_data):
    """Create age vs gender distribution matrix"""
    # Generate sample cross-tabulation data
    age_ranges = list(age_data.keys())
    genders = list(gender_data.keys())
    
    # Create sample data matrix
    data = []
    for age in age_ranges:
        for gender in genders:
            # Simulate realistic distribution
            base_count = age_data[age] * (gender_data[gender] / sum(gender_data.values()))
            noise = np.random.normal(0, base_count * 0.1)
            count = max(0, int(base_count + noise))
            data.append({'Age Range': age, 'Gender': gender, 'Count': count})
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df,
        x='Age Range',
        y='Count',
        color='Gender',
        title="Gender Distribution by Age Range",
        barmode='group',
        color_discrete_sequence=['#FF6B9D', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
    )
    
    fig.update_layout(
        font=dict(size=12),
        title_font_size=16,
        margin=dict(t=50, b=50, l=50, r=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_confidence_heatmap(genders, age_ranges, confidence_matrix):
    """Create confidence heatmap by demographics"""
    fig = px.imshow(
        confidence_matrix,
        x=age_ranges,
        y=genders,
        title="Average Confidence by Gender and Age",
        color_continuous_scale='RdYlGn',
        text_auto='.3f',
        aspect="auto"
    )
    
    fig.update_layout(
        font=dict(size=12),
        title_font_size=16,
        margin=dict(t=50, b=50, l=50, r=50),
        coloraxis_colorbar=dict(title="Confidence Score")
    )
    
    return fig

def render_sample_images(sample_detections, max_images=6):
    """Render sample detection results in a grid"""
    st.subheader("🖼️ Sample Detection Results")
    
    if not sample_detections:
        st.info("No sample images available")
        return
    
    # Limit to max_images
    samples = sample_detections[:max_images]
    
    # Create columns for grid layout
    cols_per_row = 3
    rows = (len(samples) + cols_per_row - 1) // cols_per_row
    
    for row in range(rows):
        cols = st.columns(cols_per_row)
        
        for col_idx in range(cols_per_row):
            sample_idx = row * cols_per_row + col_idx
            
            if sample_idx < len(samples):
                sample = samples[sample_idx]
                
                with cols[col_idx]:
                    # Try to display actual image if path exists
                    if 'image_path' in sample and sample['image_path']:
                        try:
                            image = Image.open(sample['image_path'])
                            st.image(
                                image, 
                                caption=f"{sample.get('predicted_gender', 'Unknown')} (conf: {sample.get('confidence', 0):.2f})",
                                use_column_width=True
                            )
                        except Exception:
                            # Fallback to placeholder
                            render_image_placeholder(sample)
                    else:
                        render_image_placeholder(sample)

def render_image_placeholder(sample):
    """Render a placeholder for images that can't be loaded"""
    gender = sample.get('predicted_gender', 'Unknown')
    confidence = sample.get('confidence', 0)
    
    # Color coding for genders
    color_map = {
        'Male': '#4ECDC4',
        'Female': '#FF6B9D', 
        'Other': '#45B7D1'
    }
    color = color_map.get(gender, '#666')
    
    placeholder_html = f"""
    <div style="
        width: 100%;
        height: 150px;
        background: linear-gradient(135deg, {color}22, {color}44);
        border: 2px solid {color};
        border-radius: 8px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        margin-bottom: 0.5rem;
    ">
        <div style="font-size: 2rem;">👤</div>
        <div style="font-weight: bold; color: {color};">{gender}</div>
        <div style="font-size: 0.8rem; color: #666;">Conf: {confidence:.2f}</div>
    </div>
    """
    
    st.markdown(placeholder_html, unsafe_allow_html=True)

def render_progress_tracker(stages, current_stage_idx=0):
    """Render a visual progress tracker for pipeline stages"""
    progress_html = "<div style='display: flex; justify-content: space-between; margin: 1rem 0;'>"
    
    for idx, stage in enumerate(stages):
        if idx < current_stage_idx:
            # Completed stage
            color = "#09ab3b"
            icon = "✅"
        elif idx == current_stage_idx:
            # Current stage
            color = "#ff9500"
            icon = "🔄"
        else:
            # Pending stage
            color = "#ccc"
            icon = "⏳"
        
        progress_html += f"""
        <div style='text-align: center; flex: 1;'>
            <div style='font-size: 1.5rem; color: {color};'>{icon}</div>
            <div style='font-size: 0.8rem; color: {color}; font-weight: bold;'>{stage}</div>
        </div>
        """
        
        # Add connector line (except for last item)
        if idx < len(stages) - 1:
            line_color = "#09ab3b" if idx < current_stage_idx else "#ccc"
            progress_html += f"""
            <div style='flex: 0.2; display: flex; align-items: center;'>
                <div style='width: 100%; height: 2px; background: {line_color};'></div>
            </div>
            """
    
    progress_html += "</div>"
    st.markdown(progress_html, unsafe_allow_html=True)

def render_config_summary(config_dict):
    """Render a formatted configuration summary"""
    st.markdown("### 🔧 Configuration Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        for key, value in list(config_dict.items())[:len(config_dict)//2]:
            formatted_key = key.replace('_', ' ').title()
            st.write(f"**{formatted_key}**: {value}")
    
    with col2:
        for key, value in list(config_dict.items())[len(config_dict)//2:]:
            formatted_key = key.replace('_', ' ').title()
            st.write(f"**{formatted_key}**: {value}")

def create_real_time_metrics_dashboard(metrics_data):
    """Create a real-time metrics dashboard"""
    # Create subplots for multiple metrics
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Processing Speed", "Memory Usage", "GPU Utilization", "Accuracy Trend"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Sample time data
    times = pd.date_range(start='2024-01-01', periods=len(metrics_data.get('processing_speed', [1])), freq='1min')
    
    # Processing speed
    fig.add_trace(
        go.Scatter(
            x=times,
            y=metrics_data.get('processing_speed', []),
            name="Images/sec",
            line=dict(color='#667eea')
        ),
        row=1, col=1
    )
    
    # Memory usage
    fig.add_trace(
        go.Scatter(
            x=times,
            y=metrics_data.get('memory_usage', []),
            name="Memory %",
            line=dict(color='#ff6b6b')
        ),
        row=1, col=2
    )
    
    # GPU utilization
    fig.add_trace(
        go.Scatter(
            x=times,
            y=metrics_data.get('gpu_usage', []),
            name="GPU %",
            line=dict(color='#4ecdc4')
        ),
        row=2, col=1
    )
    
    # Accuracy trend
    fig.add_trace(
        go.Scatter(
            x=times,
            y=metrics_data.get('accuracy', []),
            name="Accuracy",
            line=dict(color='#45b7d1')
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        title_text="Real-time System Metrics",
        title_font_size=16
    )
    
    return fig

def export_results_as_csv(results_data):
    """Export analysis results as CSV for download"""
    # Create comprehensive results DataFrame
    data_rows = []
    
    # Basic metrics
    data_rows.append(["Metric", "Value"])
    data_rows.append(["Total Images", results_data.get('total_images', 0)])
    data_rows.append(["Total Faces", results_data.get('total_faces', 0)])
    data_rows.append(["Bias Score", results_data.get('bias_score', 0)])
    data_rows.append(["Average Confidence", results_data.get('avg_confidence', 0)])
    
    # Gender distribution
    data_rows.append(["", ""])  # Empty row
    data_rows.append(["Gender Distribution", "Count"])
    
    gender_dist = results_data.get('gender_distribution', {})
    for gender, count in gender_dist.items():
        data_rows.append([gender, count])
    
    # Age distribution (if available)
    age_dist = results_data.get('age_distribution', {})
    if age_dist:
        data_rows.append(["", ""])  # Empty row
        data_rows.append(["Age Distribution", "Count"])
        
        for age_range, count in age_dist.items():
            data_rows.append([age_range, count])
    
    # Convert to CSV format
    csv_content = "\n".join([",".join(map(str, row)) for row in data_rows])
    
    return csv_content

def generate_summary_statistics(results_data):
    """Generate comprehensive summary statistics"""
    stats = {}
    
    # Basic metrics
    stats['total_samples'] = results_data.get('total_images', 0)
    stats['detection_rate'] = results_data.get('total_faces', 0) / max(results_data.get('total_images', 1), 1)
    
    # Gender balance metrics
    gender_dist = results_data.get('gender_distribution', {})
    if gender_dist:
        total_gender_samples = sum(gender_dist.values())
        stats['gender_balance'] = {
            gender: count / total_gender_samples 
            for gender, count in gender_dist.items()
        }
        
        # Calculate gender parity index (deviation from perfect balance)
        expected_ratio = 1.0 / len(gender_dist)
        deviations = [abs(ratio - expected_ratio) for ratio in stats['gender_balance'].values()]
        stats['parity_index'] = 1 - (sum(deviations) / len(deviations) / expected_ratio)
    
    # Confidence metrics
    conf_stats = results_data.get('confidence_stats', {})
    stats['confidence_reliability'] = {
        'high_confidence_ratio': len([c for c in conf_stats.get('distribution', []) if c > 0.8]) / max(len(conf_stats.get('distribution', [1])), 1),
        'low_confidence_ratio': len([c for c in conf_stats.get('distribution', []) if c < 0.5]) / max(len(conf_stats.get('distribution', [1])), 1)
    }
    
    return stats