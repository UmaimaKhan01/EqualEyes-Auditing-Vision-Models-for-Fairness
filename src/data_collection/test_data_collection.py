#!/usr/bin/env python3
"""
Standalone test for data collection without torch dependencies
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_data_collection():
    """Test the data collection functionality."""
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Test the import first
        logger.info("Testing data collection import...")
        
        # Mock the torch import in utils if needed
        import sys
        import types
        
        # Create a mock torch module
        mock_torch = types.ModuleType('torch')
        mock_torch.cuda = types.ModuleType('cuda')
        mock_torch.cuda.is_available = lambda: True
        mock_torch.cuda.get_device_name = lambda: "Mock GPU"
        mock_torch.device = lambda x: f"device({x})"
        sys.modules['torch'] = mock_torch
        
        # Now try to import the collector
        from src.data_collection.huggingface_collector import HuggingFaceDatasetCollector
        
        logger.info("✅ Import successful!")
        
        # Create config directory and basic config
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        basic_config = """
data_collection:
  min_records: 50
  max_records: 100
  max_workers: 2

processing:
  batch_size: 16
  max_workers: 2
  image_size: 224

models:
  face_detection:
    confidence_threshold: 0.5
    min_face_size: 30
"""
        
        with open("config/config.yaml", "w") as f:
            f.write(basic_config)
        
        # Create basic .env
        with open(".env", "w") as f:
            f.write("HUGGINGFACE_TOKEN=mock_token\n")
        
        logger.info("Testing data collector initialization...")
        
        # Initialize collector
        collector = HuggingFaceDatasetCollector()
        
        logger.info("✅ Collector initialized successfully!")
        
        # Test synthetic dataset creation
        logger.info("Testing synthetic dataset creation...")
        
        synthetic_data = collector.create_synthetic_dataset(10)
        
        logger.info(f"✅ Created {len(synthetic_data)} synthetic images")
        
        # Test image processing
        logger.info("Testing image processing...")
        
        if synthetic_data:
            result = collector.process_single_image(synthetic_data[0])
            logger.info(f"✅ Image processing test: {result['download_status']}")
        
        logger.info("🎉 All tests passed! Data collection is working.")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_data_collection()
    sys.exit(0 if success else 1)