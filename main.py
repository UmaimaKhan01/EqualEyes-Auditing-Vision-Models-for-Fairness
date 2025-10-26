#!/usr/bin/env python3
"""
Main execution script for Gender Bias Analysis on Flickr Data
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.utils import (
    setup_logging, load_config, load_environment, 
    validate_environment, create_directories, get_device
)
from src.data_collection.huggingface_collector import HuggingFaceDatasetCollector
from src.preprocessing.image_processor import ImagePreprocessor  
from src.analysis.gender_detector import MultiModalBiasDetector
from src.analysis.bias_analyzer import BiasAnalyzer
from src.visualization.bias_visualizer import BiasVisualizer

class GenderBiasAnalysisPipeline:
    """Main pipeline for gender bias analysis."""
    
    def __init__(self, skip_collection: bool = False, skip_preprocessing: bool = False):
        self.skip_collection = skip_collection
        self.skip_preprocessing = skip_preprocessing
        
        # Setup logging
        setup_logging("INFO")
        self.logger = logging.getLogger(__name__)
        
        # Load environment and config
        load_environment()
        self.config = load_config()
        
        # Validate environment
        if not validate_environment():
            raise RuntimeError("Environment validation failed. Please check your .env file.")
        
        # Create directories
        create_directories(self.config)
        
        # Check device
        self.device = get_device()
        
        self.logger.info("Gender Bias Analysis Pipeline initialized")
        self.logger.info(f"Using device: {self.device}")
    
    def run_data_collection(self) -> bool:
        """Run Flickr data collection step."""
        if self.skip_collection:
            self.logger.info("Skipping data collection (using existing data)")
            return True
        
        try:
            self.logger.info("=" * 60)
            self.logger.info("STEP 1: DATA COLLECTION")
            self.logger.info("=" * 60)
            
            collector = HuggingFaceDatasetCollector()
            photos = collector.collect_data()
            
            if len(photos) < 10:
                self.logger.warning(f"Only collected {len(photos)} photos, continuing anyway for testing")
                if len(photos) == 0:
                    self.logger.error("No photos collected at all")
                    return False
            
            self.logger.info(f"✅ Data collection completed: {len(photos)} photos")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Data collection failed: {str(e)}")
            return False
    
    def run_preprocessing(self) -> bool:
        """Run image preprocessing step."""
        if self.skip_preprocessing:
            self.logger.info("Skipping preprocessing (using existing data)")
            return True
        
        try:
            self.logger.info("=" * 60)
            self.logger.info("STEP 2: IMAGE PREPROCESSING")
            self.logger.info("=" * 60)
            
            preprocessor = ImagePreprocessor()
            annotations = preprocessor.run_preprocessing()
            
            if len(annotations) == 0:
                self.logger.error("No faces detected in any images")
                return False
            
            self.logger.info(f"✅ Preprocessing completed: {len(annotations)} faces detected")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Preprocessing failed: {str(e)}")
            return False
    
    def run_gender_analysis(self) -> bool:
        """Run gender detection analysis step."""
        try:
            self.logger.info("=" * 60)
            self.logger.info("STEP 3: GENDER ANALYSIS")
            self.logger.info("=" * 60)
            
            detector = MultiModalBiasDetector()
            results = detector.run_gender_analysis()
            
            if len(results) == 0:
                self.logger.error("No high-confidence gender predictions obtained")
                return False
            
            self.logger.info(f"✅ Gender analysis completed: {len(results)} predictions")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Gender analysis failed: {str(e)}")
            return False
    
    def run_bias_analysis(self) -> bool:
        """Run bias analysis step."""
        try:
            self.logger.info("=" * 60)
            self.logger.info("STEP 4: BIAS ANALYSIS")
            self.logger.info("=" * 60)
            
            analyzer = BiasAnalyzer()
            bias_results = analyzer.run_bias_analysis()
            
            if 'error' in bias_results:
                self.logger.error(f"Bias analysis returned error: {bias_results['error']}")
                return False
            
            bias_level = bias_results['bias_assessment']['bias_level']
            bias_score = bias_results['bias_assessment']['bias_percentage']
            
            self.logger.info(f"✅ Bias analysis completed")
            self.logger.info(f"   Bias Level: {bias_level.upper()}")
            self.logger.info(f"   Bias Score: {bias_score:.1f}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Bias analysis failed: {str(e)}")
            return False
    
    def run_visualization(self) -> bool:
        """Run visualization generation step."""
        try:
            self.logger.info("=" * 60)
            self.logger.info("STEP 5: VISUALIZATION GENERATION")
            self.logger.info("=" * 60)
            
            visualizer = BiasVisualizer()
            visualizer.generate_all_visualizations()
            
            self.logger.info("✅ Visualization generation completed")
            self.logger.info(f"   View results at: results/visualizations/index.html")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Visualization generation failed: {str(e)}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete analysis pipeline."""
        start_time = time.time()
        
        self.logger.info("🚀 Starting Gender Bias Analysis Pipeline")
        self.logger.info(f"📁 Working directory: {os.getcwd()}")
        
        steps = [
            ("Data Collection", self.run_data_collection),
            ("Image Preprocessing", self.run_preprocessing),
            ("Gender Analysis", self.run_gender_analysis),
            ("Bias Analysis", self.run_bias_analysis),
            ("Visualization", self.run_visualization)
        ]
        
        completed_steps = 0
        
        for step_name, step_function in steps:
            self.logger.info(f"\n📋 Running: {step_name}")
            
            step_start = time.time()
            success = step_function()
            step_time = time.time() - step_start
            
            if success:
                completed_steps += 1
                self.logger.info(f"✅ {step_name} completed in {step_time:.1f}s")
            else:
                self.logger.error(f"❌ {step_name} failed after {step_time:.1f}s")
                self.logger.error("Pipeline execution stopped due to failure")
                break
        
        total_time = time.time() - start_time
        
        if completed_steps == len(steps):
            self.logger.info("=" * 60)
            self.logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
            self.logger.info(f"⏱️  Total execution time: {total_time:.1f}s")
            self.logger.info("📊 Check results/visualizations/index.html for full report")
            self.logger.info("=" * 60)
            return True
        else:
            self.logger.error("=" * 60)
            self.logger.error(f"❌ PIPELINE FAILED: {completed_steps}/{len(steps)} steps completed")
            self.logger.error(f"⏱️  Execution time: {total_time:.1f}s")
            self.logger.error("=" * 60)
            return False

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description="Gender Bias Analysis Pipeline for Flickr Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline
  python main.py --skip-collection  # Skip data collection, use existing data
  python main.py --skip-preprocessing # Skip preprocessing, use existing data
  python main.py --step gender      # Run only gender analysis step
  python main.py --step bias        # Run only bias analysis step
  python main.py --step viz         # Run only visualization step
        """
    )
    
    parser.add_argument(
        '--skip-collection',
        action='store_true',
        help='Skip data collection step (use existing data)'
    )
    
    parser.add_argument(
        '--skip-preprocessing', 
        action='store_true',
        help='Skip preprocessing step (use existing data)'
    )
    
    parser.add_argument(
        '--step',
        choices=['collection', 'preprocessing', 'gender', 'bias', 'viz'],
        help='Run only a specific step'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Set logging level'
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize pipeline
        pipeline = GenderBiasAnalysisPipeline(
            skip_collection=args.skip_collection,
            skip_preprocessing=args.skip_preprocessing
        )
        
        # Run specific step or full pipeline
        if args.step:
            if args.step == 'collection':
                success = pipeline.run_data_collection()
            elif args.step == 'preprocessing':
                success = pipeline.run_preprocessing()
            elif args.step == 'gender':
                success = pipeline.run_gender_analysis()
            elif args.step == 'bias':
                success = pipeline.run_bias_analysis()
            elif args.step == 'viz':
                success = pipeline.run_visualization()
        else:
            success = pipeline.run_full_pipeline()
        
        if success:
            print("\n✅ Execution completed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Execution failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()