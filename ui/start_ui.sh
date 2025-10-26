#!/bin/bash

# Activate virtual environment
source gender_bias_ui/bin/activate

# Set environment variables
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Launch Streamlit app
echo "🚀 Starting Gender Bias Analysis UI..."
echo "📍 Access the interface at: http://localhost:8501"
echo "🛑 Press Ctrl+C to stop the server"
echo ""

streamlit run ui_app.py \
    --server.port=$STREAMLIT_SERVER_PORT \
    --server.address=$STREAMLIT_SERVER_ADDRESS \
    --server.headless=true \
    --browser.gatherUsageStats=false \
    --theme.base=light
