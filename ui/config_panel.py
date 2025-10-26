"""
Advanced Configuration Panel for Gender Bias Analysis Pipeline
Provides comprehensive configuration management through Streamlit UI
"""

import streamlit as st
import yaml
import json
from pathlib import Path
import os
from datetime import datetime

class ConfigurationPanel:
    """Advanced configuration panel for the pipeline"""
    
    def __init__(self, config_path="config.yaml"):
        self.config_path = config_path
        self.default_config = self._get_default_config()
        
    def _get_default_config(self):
        """Get default configuration values"""
        return {
            'data_collection': {
                'source': 'huggingface',
                'dataset_name': 'nlphuji/flickr30k',
                'sample_size': 2000,
                'filter_people': True,
                'cache_dir': './data/cache'
            },
            'face_detection': {
                'model': 'mtcnn',
                'min_face_size': 40,
                'thresholds': [0.6, 0.7, 0.7],
                'factor': 0.709,
                'device': 'auto',
                'batch_size': 16,
                'max_faces_per_image': 5
            },
            'gender_classification': {
                'model_name': 'rizvandwiki/gender-classification-2',
                'confidence_threshold': 0.7,
                'batch_size': 32,
                'device': 'auto',
                'cache_dir': './models/cache'
            },
            'age_classification': {
                'enabled': True,
                'model_name': 'nateraw/vit-age-classifier',
                'confidence_threshold': 0.6,
                'age_ranges': ['0-18', '19-35', '36-50', '51-65', '65+']
            },
            'bias_analysis': {
                'metrics': ['representation_ratio', 'confidence_gap', 'age_distribution'],
                'bias_threshold': 0.15,
                'statistical_tests': True,
                'confidence_level': 0.95
            },
            'visualization': {
                'save_plots': True,
                'plot_format': 'png',
                'dpi': 300,
                'color_scheme': 'default',
                'interactive': True
            },
            'output': {
                'save_intermediate': True,
                'output_dir': './results',
                'report_formats': ['json', 'html'],
                'include_sample_images': True,
                'max_sample_images': 20
            },
            'performance': {
                'num_workers': 4,
                'gpu_memory_fraction': 0.8,
                'enable_mixed_precision': True,
                'cache_predictions': True
            }
        }
    
    def render_configuration_panel(self):
        """Render the complete configuration panel"""
        st.header("⚙️ Advanced Configuration")
        
        # Load existing config if available
        current_config = self._load_current_config()
        
        # Ensure config is not None
        if current_config is None:
            current_config = self.default_config.copy()
        
        # Configuration tabs
        config_tabs = st.tabs([
            "📊 Data & Sampling",
            "👥 Face Detection", 
            "⚡ Classification",
            "📈 Analysis & Metrics",
            "🎨 Visualization",
            "💾 Output & Performance"
        ])
        
        with config_tabs[0]:
            self._render_data_config(current_config)
        
        with config_tabs[1]:
            self._render_face_detection_config(current_config)
        
        with config_tabs[2]:
            self._render_classification_config(current_config)
        
        with config_tabs[3]:
            self._render_analysis_config(current_config)
        
        with config_tabs[4]:
            self._render_visualization_config(current_config)
        
        with config_tabs[5]:
            self._render_output_performance_config(current_config)
        
        # Configuration actions
        st.divider()
        self._render_config_actions(current_config)
        
        return current_config
    
    def _render_data_config(self, config):
        """Render data collection and sampling configuration"""
        st.subheader("📊 Data Collection & Sampling")
        
        # Ensure config has the right structure
        if config is None:
            config = self.default_config.copy()
        
        # Get data_collection config safely
        data_config = config.get('data_collection', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            data_source = st.selectbox(
                "Data source:",
                ["huggingface", "local_directory", "api"],
                index=["huggingface", "local_directory", "api"].index(
                    data_config.get('source', 'huggingface')
                ),
                help="Choose the source for image data"
            )
            
            if data_source == "huggingface":
                dataset_name = st.text_input(
                    "Hugging Face dataset:",
                    value=data_config.get('dataset_name', 'nlphuji/flickr30k'),
                    help="Hugging Face dataset identifier"
                )
            elif data_source == "local_directory":
                local_path = st.text_input(
                    "Local directory path:",
                    value=data_config.get('local_path', './data/images'),
                    help="Path to local image directory"
                )
            else:  # API
                api_endpoint = st.text_input(
                    "API endpoint:",
                    value=data_config.get('api_endpoint', ''),
                    help="API endpoint for image data"
                )
        
        with col2:
            sample_size = st.number_input(
                "Sample size:",
                min_value=100,
                max_value=10000,
                value=data_config.get('sample_size', 2000),
                step=100,
                help="Number of images to process"
            )
            
            filter_people = st.checkbox(
                "Filter images with people",
                value=data_config.get('filter_people', True),
                help="Only include images that likely contain people"
            )
            
            cache_dir = st.text_input(
                "Cache directory:",
                value=data_config.get('cache_dir', './data/cache'),
                help="Directory for caching downloaded data"
            )
        
        # Update config safely
        if 'data_collection' not in config:
            config['data_collection'] = {}
        
        config['data_collection'] = {
            'source': data_source,
            'sample_size': sample_size,
            'filter_people': filter_people,
            'cache_dir': cache_dir
        }
        
        if data_source == "huggingface":
            config['data_collection']['dataset_name'] = dataset_name
        elif data_source == "local_directory":
            config['data_collection']['local_path'] = local_path
        else:
            config['data_collection']['api_endpoint'] = api_endpoint
    
    def _render_face_detection_config(self, config):
        """Render face detection configuration"""
        st.subheader("👥 Face Detection Settings")
        
        face_config = config.get('face_detection', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            face_model = st.selectbox(
                "Face detection model:",
                ["mtcnn", "retinaface", "opencv"],
                index=["mtcnn", "retinaface", "opencv"].index(
                    face_config.get('model', 'mtcnn')
                ),
                help="Model for face detection"
            )
            
            min_face_size = st.slider(
                "Minimum face size (pixels):",
                min_value=20,
                max_value=100,
                value=face_config.get('min_face_size', 40),
                help="Minimum size for detected faces"
            )
            
            batch_size = st.number_input(
                "Batch size:",
                min_value=1,
                max_value=128,
                value=face_config.get('batch_size', 16),
                help="Number of images processed simultaneously"
            )
        
        with col2:
            device = st.selectbox(
                "Device:",
                ["auto", "cpu", "cuda"],
                index=["auto", "cpu", "cuda"].index(
                    face_config.get('device', 'auto')
                ),
                help="Device for face detection"
            )
            
            max_faces = st.number_input(
                "Max faces per image:",
                min_value=1,
                max_value=20,
                value=face_config.get('max_faces_per_image', 5),
                help="Maximum number of faces to detect per image"
            )
        
        # Update config
        config['face_detection'] = {
            'model': face_model,
            'min_face_size': min_face_size,
            'device': device,
            'batch_size': batch_size,
            'max_faces_per_image': max_faces
        }
    
    def _render_classification_config(self, config):
        """Render gender and age classification configuration"""
        st.subheader("⚡ Gender & Age Classification")
        
        gender_config = config.get('gender_classification', {})
        age_config = config.get('age_classification', {})
        
        # Gender classification
        st.write("**Gender Classification:**")
        col1, col2 = st.columns(2)
        
        with col1:
            gender_model = st.text_input(
                "Gender model:",
                value=gender_config.get('model_name', 'rizvandwiki/gender-classification-2'),
                help="Hugging Face model for gender classification"
            )
            
            gender_confidence = st.slider(
                "Gender confidence threshold:",
                min_value=0.1,
                max_value=1.0,
                value=gender_config.get('confidence_threshold', 0.7),
                step=0.05
            )
        
        with col2:
            gender_batch_size = st.number_input(
                "Gender batch size:",
                min_value=1,
                max_value=128,
                value=gender_config.get('batch_size', 32)
            )
            
            gender_device = st.selectbox(
                "Gender classification device:",
                ["auto", "cpu", "cuda"],
                index=["auto", "cpu", "cuda"].index(
                    gender_config.get('device', 'auto')
                )
            )
        
        # Age classification
        st.write("**Age Classification:**")
        col3, col4 = st.columns(2)
        
        with col3:
            age_enabled = st.checkbox(
                "Enable age classification",
                value=age_config.get('enabled', True)
            )
            
            if age_enabled:
                age_model = st.text_input(
                    "Age model:",
                    value=age_config.get('model_name', 'nateraw/vit-age-classifier'),
                    help="Hugging Face model for age classification"
                )
        
        with col4:
            if age_enabled:
                age_confidence = st.slider(
                    "Age confidence threshold:",
                    min_value=0.1,
                    max_value=1.0,
                    value=age_config.get('confidence_threshold', 0.6),
                    step=0.05
                )
        
        # Update config
        config['gender_classification'] = {
            'model_name': gender_model,
            'confidence_threshold': gender_confidence,
            'batch_size': gender_batch_size,
            'device': gender_device
        }
        
        if age_enabled:
            config['age_classification'] = {
                'enabled': True,
                'model_name': age_model,
                'confidence_threshold': age_confidence,
                'age_ranges': ['0-18', '19-35', '36-50', '51-65', '65+']
            }
        else:
            config['age_classification'] = {'enabled': False}
    
    def _render_analysis_config(self, config):
        """Render bias analysis configuration"""
        st.subheader("📈 Bias Analysis & Metrics")
        
        bias_config = config.get('bias_analysis', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            bias_threshold = st.slider(
                "Bias detection threshold:",
                min_value=0.05,
                max_value=0.5,
                value=bias_config.get('bias_threshold', 0.15),
                step=0.01,
                help="Threshold for detecting significant bias"
            )
            
            statistical_tests = st.checkbox(
                "Enable statistical tests",
                value=bias_config.get('statistical_tests', True),
                help="Run statistical significance tests"
            )
        
        with col2:
            available_metrics = [
                'representation_ratio',
                'confidence_gap', 
                'age_distribution',
                'intersectionality'
            ]
            
            selected_metrics = st.multiselect(
                "Analysis metrics:",
                available_metrics,
                default=bias_config.get('metrics', ['representation_ratio', 'confidence_gap']),
                help="Select which bias metrics to calculate"
            )
        
        config['bias_analysis'] = {
            'metrics': selected_metrics,
            'bias_threshold': bias_threshold,
            'statistical_tests': statistical_tests,
            'confidence_level': 0.95
        }
    
    def _render_visualization_config(self, config):
        """Render visualization configuration"""
        st.subheader("🎨 Visualization Settings")
        
        viz_config = config.get('visualization', {})
        
        col1, col2 = st.columns(2)
        
        with col1:
            save_plots = st.checkbox(
                "Save plots to disk",
                value=viz_config.get('save_plots', True)
            )
            
            plot_format = st.selectbox(
                "Plot format:",
                ["png", "jpg", "svg", "pdf"],
                index=["png", "jpg", "svg", "pdf"].index(
                    viz_config.get('plot_format', 'png')
                )
            )
        
        with col2:
            interactive = st.checkbox(
                "Generate interactive plots",
                value=viz_config.get('interactive', True),
                help="Create interactive Plotly visualizations"
            )
            
            color_scheme = st.selectbox(
                "Color scheme:",
                ["default", "colorblind", "high_contrast"],
                index=["default", "colorblind", "high_contrast"].index(
                    viz_config.get('color_scheme', 'default')
                )
            )
        
        config['visualization'] = {
            'save_plots': save_plots,
            'plot_format': plot_format,
            'interactive': interactive,
            'color_scheme': color_scheme
        }
    
    def _render_output_performance_config(self, config):
        """Render output and performance configuration"""
        st.subheader("💾 Output & Performance")
        
        output_config = config.get('output', {})
        perf_config = config.get('performance', {})
        
        # Output settings
        st.write("**Output Settings:**")
        col1, col2 = st.columns(2)
        
        with col1:
            save_intermediate = st.checkbox(
                "Save intermediate results",
                value=output_config.get('save_intermediate', True)
            )
            
            output_dir = st.text_input(
                "Output directory:",
                value=output_config.get('output_dir', './results')
            )
        
        with col2:
            report_formats = st.multiselect(
                "Report formats:",
                ["json", "html", "pdf", "csv"],
                default=output_config.get('report_formats', ['json', 'html'])
            )
        
        # Performance settings
        st.write("**Performance Settings:**")
        col3, col4 = st.columns(2)
        
        with col3:
            num_workers = st.number_input(
                "Number of workers:",
                min_value=1,
                max_value=16,
                value=perf_config.get('num_workers', 4),
                help="Number of parallel workers for data loading"
            )
        
        with col4:
            gpu_memory_fraction = st.slider(
                "GPU memory fraction:",
                min_value=0.1,
                max_value=1.0,
                value=perf_config.get('gpu_memory_fraction', 0.8),
                step=0.1,
                help="Fraction of GPU memory to use"
            )
        
        # Update config
        config['output'] = {
            'save_intermediate': save_intermediate,
            'output_dir': output_dir,
            'report_formats': report_formats
        }
        
        config['performance'] = {
            'num_workers': num_workers,
            'gpu_memory_fraction': gpu_memory_fraction
        }
    
    def _render_config_actions(self, config):
        """Render configuration action buttons"""
        st.subheader("💾 Configuration Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save Config", use_container_width=True):
                self._save_config(config)
                st.success("Configuration saved!")
        
        with col2:
            if st.button("🔄 Reset to Default", use_container_width=True):
                st.session_state.config = self.default_config.copy()
                st.success("Configuration reset to defaults!")
                st.rerun()
        
        with col3:
            if st.button("📤 Export Config", use_container_width=True):
                config_yaml = yaml.dump(config, default_flow_style=False)
                st.download_button(
                    label="Download YAML",
                    data=config_yaml,
                    file_name=f"pipeline_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml",
                    mime="text/yaml"
                )
        
        # Configuration preview
        with st.expander("🔍 View Current Configuration"):
            st.code(yaml.dump(config, default_flow_style=False), language='yaml')
    
    def _load_current_config(self):
        """Load current configuration from session state or file"""
        if 'config' in st.session_state:
            return st.session_state.config
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                st.session_state.config = config
                return config
            except Exception as e:
                st.warning(f"Could not load config file: {e}")
        
        st.session_state.config = self.default_config.copy()
        return st.session_state.config
    
    def _save_config(self, config):
        """Save configuration to file and session state"""
        st.session_state.config = config
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.config_path) if os.path.dirname(self.config_path) else '.', exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        except Exception as e:
            st.error(f"Could not save config file: {e}")