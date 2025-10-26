# Gender Bias Analysis Pipeline - Web UI

A comprehensive web interface for analyzing gender bias in image datasets using Streamlit. This UI provides an intuitive way to upload data, configure analysis parameters, run the pipeline, and visualize results.

## 🌟 Features

### 📊 **Dashboard Overview**
- Real-time pipeline status tracking
- Quick statistics and metrics
- Progress monitoring with visual indicators
- System resource monitoring

### 📁 **Dataset Management**
- **Multiple data sources**: Hugging Face datasets, local uploads, ZIP archives
- **Image upload interface**: Drag-and-drop support for individual images
- **Bulk upload**: ZIP archive extraction and processing
- **Dataset browser**: Visual gallery with pagination and search
- **Dataset persistence**: Save and reload processed datasets

### ⚙️ **Advanced Configuration**
- **Comprehensive settings**: All pipeline parameters configurable through UI
- **Face detection tuning**: MTCNN parameters, batch sizes, device selection
- **Classification models**: Gender and age classification model selection
- **Bias analysis**: Customizable metrics and thresholds
- **Performance optimization**: GPU memory management, batch processing
- **Configuration management**: Save, load, export configurations

### 📈 **Interactive Visualizations**
- **Gender distribution**: Pie charts and bar graphs
- **Confidence analysis**: Histograms with threshold indicators
- **Bias trends**: Time series analysis with alert levels
- **Cross-analysis**: Age vs gender heatmaps
- **Real-time metrics**: Live system performance monitoring

### 📋 **Analysis & Reporting**
- **Detailed bias assessment**: Automated bias scoring with interpretations
- **Statistical analysis**: Confidence intervals and significance tests
- **Sample image display**: Visual examples with predictions
- **Export capabilities**: Multiple report formats (PDF, HTML, JSON, CSV)
- **Comprehensive metrics**: Representation ratios, confidence gaps

### 🔧 **Pipeline Control**
- **Full pipeline execution**: One-click complete analysis
- **Stage-by-stage control**: Run individual pipeline components
- **Progress tracking**: Visual progress indicators and logs
- **Error handling**: Detailed error messages and recovery options
- **Real-time logs**: Live pipeline execution monitoring

## 🚀 Quick Start

### 1. **Deployment**
```bash
# Make deployment script executable (if not already)
chmod +x deploy_ui.sh

# Run deployment script
./deploy_ui.sh
```

The deployment script will:
- ✅ Check Python version (3.8+ required)
- ✅ Create virtual environment
- ✅ Install all dependencies
- ✅ Set up directory structure
- ✅ Create default configuration
- ✅ Verify GPU availability
- ✅ Create startup scripts

### 2. **Launch UI**
```bash
# Option 1: Use startup script
./start_ui.sh

# Option 2: Manual launch
source gender_bias_ui/bin/activate
streamlit run ui_app.py
```

### 3. **Access Interface**
Open your browser and navigate to:
```
http://localhost:8501
```

## 📖 User Guide

### **Getting Started**

1. **Choose Data Source** (Dashboard → Analysis tab)
   - **Hugging Face**: Use pre-configured datasets like Flickr30k
   - **Upload Images**: Individual image files
   - **ZIP Archive**: Bulk upload via compressed file
   - **Local Directory**: Point to existing image folder

2. **Configure Analysis** (Sidebar)
   - Set sample size (100-5000 images)
   - Adjust confidence threshold (0.1-1.0)
   - Enable/disable age detection
   - Modify advanced parameters

3. **Run Pipeline** (Pipeline Control tab)
   - **Full Pipeline**: Complete analysis from start to finish
   - **Stage Control**: Run individual components
   - **Monitor Progress**: Watch real-time execution

4. **View Results** (Analysis tab)
   - Bias assessment with color-coded alerts
   - Detailed statistics and distributions
   - Sample detection results
   - Confidence analysis

5. **Explore Visualizations** (Visualizations tab)
   - Interactive charts and graphs
   - Multiple visualization types
   - Customizable parameters
   - Export capabilities

6. **Generate Reports** (Reports tab)
   - Multiple format options
   - Customizable detail levels
   - Include sample images
   - Download reports

### **Advanced Configuration**

Access the advanced configuration panel for detailed customization:

#### **Data Collection Settings**
- **Source selection**: HuggingFace, local, or API
- **Sampling parameters**: Size, filtering options
- **Caching configuration**: Local storage settings

#### **Face Detection Tuning**
- **Model selection**: MTCNN, RetinaFace, OpenCV
- **MTCNN parameters**: Thresholds, scale factors
- **Performance settings**: Batch size, device selection
- **Quality controls**: Minimum face size, max faces per image

#### **Classification Models**
- **Gender classification**: Model selection, confidence thresholds
- **Age classification**: Enable/disable, age range definitions
- **Batch processing**: Optimize for your hardware
- **Device management**: CPU/GPU selection

#### **Bias Analysis Configuration**
- **Metrics selection**: Choose analysis methods
- **Threshold settings**: Define bias detection levels
- **Statistical tests**: Enable significance testing
- **Confidence levels**: Set statistical parameters

#### **Visualization & Output**
- **Plot settings**: Format, DPI, color schemes
- **Export options**: Multiple report formats
- **Sample management**: Control sample image inclusion
- **Directory structure**: Organize output files

## 🔧 Technical Requirements

### **System Requirements**
- **Operating System**: Linux, macOS, Windows
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB+ recommended
- **Storage**: 2GB free space for models and cache
- **Network**: Internet connection for model downloads

### **Hardware Acceleration**
- **GPU Support**: NVIDIA GPU with CUDA support (optional but recommended)
- **CPU**: Multi-core processor recommended for parallel processing
- **Memory**: Higher RAM for larger batch sizes

### **Dependencies**
Core dependencies are automatically installed by the deployment script:
- **Streamlit**: Web framework
- **Plotly**: Interactive visualizations
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face model library
- **OpenCV**: Computer vision
- **MTCNN**: Face detection
- **PIL/Pillow**: Image processing
- **Pandas/NumPy**: Data analysis

## 🔍 Troubleshooting

### **Common Issues**

#### **UI Won't Start**
```bash
# Check Python version
python3 --version

# Verify virtual environment
source gender_bias_ui/bin/activate
which python

# Test Streamlit
streamlit hello
```

#### **Port Already in Use**
```bash
# Check what's using port 8501
netstat -tlnp | grep 8501

# Kill existing process
sudo kill -9 <PID>

# Or use different port
streamlit run ui_app.py --server.port 8502
```

#### **GPU Not Detected**
```bash
# Check NVIDIA drivers
nvidia-smi

# Verify CUDA installation
python3 -c "import torch; print(torch.cuda.is_available())"

# Install CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### **Memory Issues**
- Reduce batch sizes in configuration
- Lower sample size for analysis
- Close other applications
- Consider using CPU-only mode

#### **Model Download Failures**
- Check internet connection
- Verify Hugging Face access
- Clear model cache: `rm -rf models/cache/*`
- Use manual model download

### **Performance Optimization**

#### **GPU Optimization**
- Set appropriate GPU memory fraction (0.6-0.8)
- Use mixed precision if available
- Optimize batch sizes for your GPU memory

#### **CPU Optimization**
- Adjust number of workers based on CPU cores
- Enable parallel processing where possible
- Use appropriate batch sizes

#### **Memory Management**
- Enable result caching to avoid recomputation
- Clear intermediate results when not needed
- Monitor memory usage during processing

## 📁 Project Structure

```
gender-bias-analysis/
├── ui_app.py                 # Main Streamlit application
├── ui_components.py          # Reusable UI components
├── config_panel.py           # Advanced configuration interface
├── dataset_manager.py        # Dataset upload and management
├── deploy_ui.sh              # Deployment script
├── start_ui.sh               # UI startup script
├── requirements_ui.txt       # UI-specific dependencies
├── config.yaml               # Default configuration
├── src/                      # Core pipeline modules
│   ├── data_collection/
│   ├── face_detection/
│   ├── gender_classification/
│   ├── bias_analysis/
│   └── visualization/
├── data/                     # Data directories
│   ├── uploads/              # Uploaded images
│   ├── processed/            # Processed datasets
│   └── cache/                # Cached data
├── results/                  # Analysis results
│   ├── plots/                # Generated visualizations
│   └── reports/              # Analysis reports
└── logs/                     # Application logs
```

## 🤝 Contributing

### **Adding New Features**
1. **UI Components**: Add reusable components to `ui_components.py`
2. **Configuration**: Extend `config_panel.py` for new parameters
3. **Visualizations**: Add new chart types and interactive elements
4. **Data Sources**: Extend `dataset_manager.py` for new data sources

### **Customization**
- **Themes**: Modify CSS in the main app for custom styling
- **Layouts**: Adjust column layouts and tab organization
- **Metrics**: Add custom bias metrics and analysis methods
- **Export Formats**: Add new report generation options

## 📄 License

This project is part of the Gender Bias Analysis Pipeline. Please refer to the main project license for usage terms.

## 🆘 Support

For support and questions:
1. **Check this documentation** for common solutions
2. **Review error logs** in the UI and console output
3. **Verify configuration** settings and requirements
4. **Test with minimal datasets** to isolate issues
5. **Contact the development team** for technical support

---

**Happy Analyzing! 🔍👥**

*Building fair and unbiased AI systems, one analysis at a time.*