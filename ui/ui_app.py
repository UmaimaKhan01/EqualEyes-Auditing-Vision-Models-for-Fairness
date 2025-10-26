import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
from pathlib import Path
import time
from datetime import datetime
import base64
from PIL import Image
import cv2

# Add the src directory to Python path for imports
project_root = Path(__file__).parent
sys.path.append(str(project_root / "src"))
sys.path.append(str(project_root.parent / "src"))  # Also try parent/src

# Import our UI modules
try:
    from ui_components import *
    from config_panel import ConfigurationPanel
    from dataset_manager import DatasetManager
    UI_MODULES_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import UI modules: {e}")
    UI_MODULES_AVAILABLE = False

# Import our pipeline modules (optional - UI works without them)
PIPELINE_AVAILABLE = True
try:
    from data_collection.huggingface_collector import HuggingFaceDatasetCollector
    from face_detection.detector import FaceDetector
    from gender_classification.classifier import GenderClassifier
    from bias_analysis.analyzer import BiasAnalyzer
    from visualization.plotter import BiasVisualizer
    from utils.config_manager import ConfigManager
    from utils.logger import setup_logger
except ImportError as e:
    PIPELINE_AVAILABLE = False
    st.warning(f"Pipeline modules not found: {e}")
    st.info("UI running in demo mode. To enable full functionality, ensure the pipeline is in ../src/")

# Page configuration
st.set_page_config(
    page_title="Gender Bias Analysis Pipeline",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .status-running {
        color: #ff6b6b;
        font-weight: bold;
    }
    
    .status-complete {
        color: #51cf66;
        font-weight: bold;
    }
    
    .bias-alert {
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .bias-high {
        background-color: #ffe0e0;
        border-left: 4px solid #ff4757;
    }
    
    .bias-medium {
        background-color: #fff3cd;
        border-left: 4px solid #ffa502;
    }
    
    .bias-low {
        background-color: #d4edda;
        border-left: 4px solid #2ed573;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'pipeline_state' not in st.session_state:
    st.session_state.pipeline_state = {
        'data_loaded': False,
        'faces_detected': False,
        'genders_classified': False,
        'analysis_complete': False,
        'results': None,
        'config': None,
        'progress': 0
    }

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []

# Initialize UI components if available
if UI_MODULES_AVAILABLE:
    if 'config_panel' not in st.session_state:
        st.session_state.config_panel = ConfigurationPanel()

    if 'dataset_manager' not in st.session_state:
        st.session_state.dataset_manager = DatasetManager()

# Sidebar configuration
with st.sidebar:
    st.header("🔧 Quick Configuration")
    
    # Basic settings
    st.subheader("Basic Settings")
    
    sample_size = st.slider(
        "Sample size:",
        min_value=100,
        max_value=5000,
        value=2000,
        step=100,
        help="Number of images to analyze"
    )
    
    confidence_threshold = st.slider(
        "Confidence threshold:",
        min_value=0.1,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Minimum confidence for gender classification"
    )
    
    enable_age_detection = st.checkbox(
        "Enable age detection",
        value=True,
        help="Also classify age ranges"
    )
    
    st.divider()
    
    # Pipeline status
    st.subheader("Pipeline Status")
    if UI_MODULES_AVAILABLE:
        render_status_indicator("Data Loading", st.session_state.pipeline_state['data_loaded'])
        render_status_indicator("Face Detection", st.session_state.pipeline_state['faces_detected']) 
        render_status_indicator("Gender Classification", st.session_state.pipeline_state['genders_classified'])
        render_status_indicator("Bias Analysis", st.session_state.pipeline_state['analysis_complete'])
    else:
        st.write("⏳ Data Loading: Pending")
        st.write("⏳ Face Detection: Pending")
        st.write("⏳ Gender Classification: Pending")
        st.write("⏳ Bias Analysis: Pending")
    
    st.divider()
    
    # Quick actions
    st.subheader("Quick Actions")
    if st.button("🚀 Run Demo Pipeline", use_container_width=True):
        st.session_state.run_demo_pipeline = True
    
    if UI_MODULES_AVAILABLE and st.button("⚙️ Advanced Config", use_container_width=True):
        st.session_state.show_advanced_config = True
    
    if UI_MODULES_AVAILABLE and st.button("📁 Manage Data", use_container_width=True):
        st.session_state.show_dataset_manager = True

# Main header
st.markdown("""
<div class="main-header">
    <h1>👥 Gender Bias Analysis Pipeline</h1>
    <p>Analyze gender representation and bias in image datasets</p>
</div>
""", unsafe_allow_html=True)

# Handle advanced config modal
if UI_MODULES_AVAILABLE and 'show_advanced_config' in st.session_state and st.session_state.show_advanced_config:
    with st.expander("⚙️ Advanced Configuration", expanded=True):
        config = st.session_state.config_panel.render_configuration_panel()
        st.session_state.current_config = config
        
        if st.button("✅ Close Advanced Config"):
            st.session_state.show_advanced_config = False
            st.rerun()

# Handle dataset manager modal  
if UI_MODULES_AVAILABLE and 'show_dataset_manager' in st.session_state and st.session_state.show_dataset_manager:
    with st.expander("📁 Dataset Management", expanded=True):
        data_source = st.session_state.dataset_manager.render_dataset_management_panel()
        
        if st.button("✅ Close Dataset Manager"):
            st.session_state.show_dataset_manager = False
            st.rerun()

# Create tabs for different sections
if UI_MODULES_AVAILABLE:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Dashboard", 
        "📁 Dataset", 
        "📊 Analysis", 
        "📈 Visualizations", 
        "📋 Reports", 
        "⚙️ Pipeline Control"
    ])
else:
    tab1, tab2, tab3 = st.tabs([
        "🏠 Demo Dashboard", 
        "📊 Demo Analysis", 
        "📈 Demo Visualizations"
    ])

# Tab 1: Dashboard
with tab1:
    st.header("Dashboard Overview")
    
    if UI_MODULES_AVAILABLE:
        # Pipeline progress tracker
        stages = ["Data Loading", "Face Detection", "Gender Classification", "Bias Analysis"]
        current_stage = 0
        
        # Determine current stage
        if st.session_state.pipeline_state['analysis_complete']:
            current_stage = 4
        elif st.session_state.pipeline_state['genders_classified']:
            current_stage = 3
        elif st.session_state.pipeline_state['faces_detected']:
            current_stage = 2
        elif st.session_state.pipeline_state['data_loaded']:
            current_stage = 1
        
        render_progress_tracker(stages, current_stage)
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_images = len(st.session_state.uploaded_images) if 'uploaded_images' in st.session_state else 0
        if UI_MODULES_AVAILABLE:
            render_metric_card("Total Images", f"{total_images:,}")
        else:
            st.metric("Total Images", f"{total_images:,}")
    
    with col2:
        if st.session_state.analysis_results:
            total_faces = st.session_state.analysis_results.get('total_faces', 0)
            if UI_MODULES_AVAILABLE:
                render_metric_card("Faces Detected", f"{total_faces:,}")
            else:
                st.metric("Faces Detected", f"{total_faces:,}")
        else:
            if UI_MODULES_AVAILABLE:
                render_metric_card("Faces Detected", "0")
            else:
                st.metric("Faces Detected", "0")
    
    with col3:
        if st.session_state.analysis_results:
            avg_confidence = st.session_state.analysis_results.get('avg_confidence', 0)
            if UI_MODULES_AVAILABLE:
                render_metric_card("Avg Confidence", f"{avg_confidence:.2%}")
            else:
                st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        else:
            if UI_MODULES_AVAILABLE:
                render_metric_card("Avg Confidence", "0%")
            else:
                st.metric("Avg Confidence", "0%")
    
    with col4:
        if st.session_state.analysis_results:
            bias_score = st.session_state.analysis_results.get('bias_score', 0)
            if UI_MODULES_AVAILABLE:
                render_metric_card("Bias Score", f"{bias_score:.3f}")
            else:
                st.metric("Bias Score", f"{bias_score:.3f}")
        else:
            if UI_MODULES_AVAILABLE:
                render_metric_card("Bias Score", "0.000")
            else:
                st.metric("Bias Score", "0.000")
    
    # Progress bar
    if st.session_state.pipeline_state['progress'] > 0:
        st.progress(st.session_state.pipeline_state['progress'] / 100)
    
    # Demo pipeline button
    if not PIPELINE_AVAILABLE:
        st.info("🚀 Pipeline modules not found. Running in demo mode.")
        if st.button("🎮 Run Demo Analysis", type="primary"):
            with st.spinner("Running demo analysis..."):
                time.sleep(3)
                
                # Generate demo results
                demo_results = {
                    'total_images': sample_size,
                    'total_faces': int(sample_size * 1.3),
                    'avg_confidence': 0.756,
                    'bias_score': 0.234,
                    'gender_distribution': {
                        'Male': int(sample_size * 0.45),
                        'Female': int(sample_size * 0.52),
                        'Other': int(sample_size * 0.03)
                    }
                }
                
                st.session_state.analysis_results = demo_results
                st.success("✅ Demo analysis complete!")
                st.rerun()
    
    # Quick overview visualization
    if st.session_state.analysis_results:
        st.subheader("Quick Overview")
        
        gender_data = st.session_state.analysis_results.get('gender_distribution', {})
        if gender_data:
            if UI_MODULES_AVAILABLE:
                fig = create_gender_distribution_chart(gender_data, chart_type="pie")
                st.plotly_chart(fig, use_container_width=True)
            else:
                fig = px.pie(
                    values=list(gender_data.values()),
                    names=list(gender_data.keys()),
                    title="Gender Distribution",
                    color_discrete_sequence=['#FF6B9D', '#4ECDC4', '#45B7D1']
                )
                st.plotly_chart(fig, use_container_width=True)

# Only show advanced tabs if UI modules are available
if UI_MODULES_AVAILABLE:
    # Tab 2: Dataset Management  
    with tab2:
        data_source = st.session_state.dataset_manager.render_dataset_management_panel()

    # Tab 3: Analysis
    with tab3:
        st.header("Detailed Analysis Results")
        
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            # Bias assessment using UI component
            st.subheader("📊 Bias Assessment")
            bias_score = results.get('bias_score', 0)
            render_bias_alert(bias_score)
            
            # Rest of analysis tab content...
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Gender Statistics")
                gender_dist = results.get('gender_distribution', {})
                
                if gender_dist:
                    gender_df = pd.DataFrame([
                        {'Gender': k, 'Count': v, 'Percentage': v/sum(gender_dist.values())*100}
                        for k, v in gender_dist.items()
                    ])
                    st.dataframe(gender_df, use_container_width=True)
            
            with col2:
                st.subheader("🎯 Confidence Analysis")
                conf_stats = results.get('confidence_stats', {})
                
                st.write(f"**Mean Confidence:** {conf_stats.get('mean', 0.756):.3f}")
                st.write(f"**Median Confidence:** {conf_stats.get('median', 0.782):.3f}")
                st.write(f"**Standard Deviation:** {conf_stats.get('std', 0.145):.3f}")
        else:
            st.info("👆 Run the analysis pipeline to see detailed results here.")

    # Tab 4: Visualizations
    with tab4:
        st.header("Interactive Visualizations")
        
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            viz_type = st.selectbox(
                "Visualization type:",
                ["Gender Distribution", "Confidence Analysis", "Bias Trends"]
            )
            
            if viz_type == "Gender Distribution":
                gender_data = results.get('gender_distribution', {})
                fig = create_gender_distribution_chart(gender_data, "pie")
                st.plotly_chart(fig, use_container_width=True)
            
            elif viz_type == "Confidence Analysis":
                np.random.seed(42)
                confidence_data = np.random.beta(3, 1, 1000) * 0.8 + 0.2
                fig = create_confidence_histogram(confidence_data, confidence_threshold)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👆 Run the analysis pipeline to generate visualizations.")

    # Tab 5: Reports
    with tab5:
        st.header("Analysis Reports")
        st.info("Reports functionality available when analysis is complete.")

    # Tab 6: Pipeline Control
    with tab6:
        st.header("Pipeline Control")
        st.info("Pipeline control functionality - integration with full pipeline required.")

else:
    # Simplified tabs for demo mode
    with tab2:
        st.header("Demo Analysis")
        
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            st.subheader("📊 Gender Distribution")
            gender_dist = results.get('gender_distribution', {})
            
            for gender, count in gender_dist.items():
                total = sum(gender_dist.values())
                percentage = (count / total) * 100
                st.write(f"**{gender}**: {count:,} images ({percentage:.1f}%)")
            
            # Simple bias assessment
            bias_score = results.get('bias_score', 0)
            if bias_score > 0.3:
                st.error(f"⚠️ High bias detected (score: {bias_score:.3f})")
            elif bias_score > 0.15:
                st.warning(f"⚡ Moderate bias detected (score: {bias_score:.3f})")
            else:
                st.success(f"✅ Low bias detected (score: {bias_score:.3f})")
        else:
            st.info("Run demo analysis to see results")

    with tab3:
        st.header("Demo Visualizations")
        
        if st.session_state.analysis_results:
            gender_data = st.session_state.analysis_results.get('gender_distribution', {})
            
            if gender_data:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_pie = px.pie(
                        values=list(gender_data.values()),
                        names=list(gender_data.keys()),
                        title="Gender Distribution (Pie Chart)",
                        color_discrete_sequence=['#FF6B9D', '#4ECDC4', '#45B7D1']
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    fig_bar = px.bar(
                        x=list(gender_data.keys()),
                        y=list(gender_data.values()),
                        title="Gender Distribution (Bar Chart)",
                        color=list(gender_data.keys()),
                        color_discrete_sequence=['#FF6B9D', '#4ECDC4', '#45B7D1']
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Run demo analysis to see visualizations")

# Helper functions for demo mode
def run_demo_pipeline():
    """Run a demo version of the pipeline"""
    with st.spinner("Running demo pipeline..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        stages = [
            ("Loading demo dataset...", 20),
            ("Detecting faces...", 40), 
            ("Classifying genders...", 70),
            ("Analyzing bias...", 90),
            ("Generating results...", 100)
        ]
        
        for stage_name, progress in stages:
            status_text.text(stage_name)
            time.sleep(1)
            progress_bar.progress(progress)
        
        # Generate demo results
        demo_results = {
            'total_images': sample_size,
            'total_faces': int(sample_size * 1.3),
            'avg_confidence': 0.756,
            'bias_score': 0.234,
            'gender_distribution': {
                'Male': int(sample_size * 0.45),
                'Female': int(sample_size * 0.52),
                'Other': int(sample_size * 0.03)
            }
        }
        
        st.session_state.analysis_results = demo_results
        st.session_state.pipeline_state.update({
            'data_loaded': True,
            'faces_detected': True,
            'genders_classified': True,
            'analysis_complete': True,
            'progress': 100
        })
        
        status_text.text("✅ Demo pipeline completed!")
        st.success("🎉 Demo analysis completed successfully!")

# Handle demo pipeline execution
if 'run_demo_pipeline' in st.session_state and st.session_state.run_demo_pipeline:
    st.session_state.run_demo_pipeline = False
    run_demo_pipeline()

# Footer
st.markdown("---")
mode = "Full Mode" if PIPELINE_AVAILABLE else "Demo Mode"
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Gender Bias Analysis Pipeline v1.0 ({mode}) | Built with Streamlit</p>
    <p>{'Full pipeline integration active' if PIPELINE_AVAILABLE else 'Running in demo mode - set up full pipeline for complete functionality'}</p>
</div>
""", unsafe_allow_html=True)