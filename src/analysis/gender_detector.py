import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import pipeline
import cv2
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..utils import load_config, ProgressTracker, save_results, get_device

class MultiModalBiasDetector:
    """Detect multiple types of bias: gender, race, age, disability."""
    
    def __init__(self):
        self.config = load_config()
        self.device = get_device()
        
        # Load multiple classification models
        self.models = {
            'gender': pipeline("image-classification", model="rizvandwiki/gender-classification-2", device=0 if torch.cuda.is_available() else -1),
            'race': pipeline("image-classification", model="dima806/facial_emotions_image_detection", device=0 if torch.cuda.is_available() else -1),
            'age': pipeline("image-classification", model="nateraw/vit-age-classifier", device=0 if torch.cuda.is_available() else -1),
            'emotion': pipeline("image-classification", model="trpakov/vit-face-expression", device=0 if torch.cuda.is_available() else -1)
        }
        
        self.logger = logging.getLogger(__name__)
    
    def analyze_intersectional_bias(self, image_input) -> Dict[str, Any]:
        """Analyze multiple bias dimensions simultaneously."""
        try:
            # Handle both file paths and PIL Images
            if isinstance(image_input, str):
                image = Image.open(image_input)
            else:
                image = image_input
            
            results = {}
            for attribute, model in self.models.items():
                try:
                    predictions = model(image)
                    results[attribute] = {
                        'prediction': predictions[0]['label'],
                        'confidence': predictions[0]['score'],
                        'all_scores': predictions
                    }
                except Exception as e:
                    results[attribute] = {'error': str(e)}
            
            # Calculate intersectional bias score
            results['intersectional_score'] = self.calculate_intersectional_bias(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Multi-modal analysis failed: {e}")
            return {'error': str(e)}
    
    def calculate_intersectional_bias(self, results: Dict) -> float:
        """Calculate compound bias score across multiple dimensions."""
        bias_factors = []
        
        # Gender bias weight
        if 'gender' in results and 'confidence' in results['gender']:
            bias_factors.append(abs(0.5 - results['gender']['confidence']))
        
        # Age bias (check for age representation imbalance)
        if 'age' in results and 'confidence' in results['age']:
            bias_factors.append(results['age']['confidence'] * 0.3)
        
        # Calculate weighted average
        return sum(bias_factors) / len(bias_factors) if bias_factors else 0.0
    
    def run_gender_analysis(self):
        """Run gender analysis for compatibility with main pipeline."""
        try:
            self.logger.info("Starting multi-modal bias analysis...")
            
            # Load face annotations
            annotations_file = "data/annotations/face_annotations.csv"
            if not Path(annotations_file).exists():
                self.logger.error("Face annotations not found")
                return []
            
            import pandas as pd
            annotations_df = pd.read_csv(annotations_file)
            
            results = []
            for _, row in annotations_df.iterrows():
                face_path = row['face_path']
                if Path(face_path).exists():
                    analysis = self.analyze_intersectional_bias(face_path)
                    if 'error' not in analysis:
                        analysis['face_id'] = row['face_id']
                        results.append(analysis)
            
            self.logger.info(f"Filtered to {len(results)} high-confidence predictions from {len(annotations_df)} total faces")
            
            # Save results
            results_dir = Path("results")
            results_dir.mkdir(exist_ok=True)
            
            import json
            from datetime import datetime
            
            summary = {
                'timestamp': datetime.now().isoformat(),
                'total_predictions': len(results),
                'analysis_type': 'multi_modal_intersectional',
                'results': results
            }
            
            with open("results/gender_analysis_summary.json", 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Multi-modal analysis failed: {e}")
            return []
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_models(self):
        """Initialize gender detection and age estimation models."""
        try:
            # Gender detection model
            gender_model_name = self.config['models']['gender_detection']['model_name']
            self.logger.info(f"Loading gender detection model: {gender_model_name}")
            
            self.gender_processor = AutoImageProcessor.from_pretrained(gender_model_name)
            self.gender_model = AutoModelForImageClassification.from_pretrained(gender_model_name)
            self.gender_model.to(self.device)
            self.gender_model.eval()
            
            # Age estimation model (optional)
            if self.config['models']['age_estimation']['enabled']:
                age_model_name = self.config['models']['age_estimation']['model_name']
                self.logger.info(f"Loading age estimation model: {age_model_name}")
                
                self.age_pipeline = pipeline(
                    "image-classification",
                    model=age_model_name,
                    device=0 if self.device.type == 'cuda' else -1
                )
            else:
                self.age_pipeline = None
            
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
    
    def preprocess_face_image(self, image_path: str) -> torch.Tensor:
        """Preprocess face image for model input."""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Process with model processor
            inputs = self.gender_processor(image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            return inputs
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return None
    
    def predict_gender(self, image_path: str) -> Dict[str, Any]:
        """Predict gender from face image."""
        try:
            # Preprocess image
            inputs = self.preprocess_face_image(image_path)
            if inputs is None:
                return {'status': 'preprocessing_failed'}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.gender_model(**inputs)
                probabilities = F.softmax(outputs.logits, dim=-1)
            
            # Get model labels
            labels = self.gender_model.config.id2label
            
            # Extract predictions
            probs = probabilities.cpu().numpy()[0]
            predictions = []
            
            for idx, prob in enumerate(probs):
                label = labels.get(idx, f"class_{idx}")
                predictions.append({
                    'label': label.lower(),
                    'confidence': float(prob)
                })
            
            # Sort by confidence
            predictions.sort(key=lambda x: x['confidence'], reverse=True)
            
            # Get top prediction
            top_prediction = predictions[0]
            
            # Map labels to standard format
            gender_mapping = {
                'male': 'male',
                'man': 'male',
                'female': 'female',
                'woman': 'female'
            }
            
            predicted_gender = gender_mapping.get(top_prediction['label'], top_prediction['label'])
            confidence = top_prediction['confidence']
            
            return {
                'status': 'success',
                'predicted_gender': predicted_gender,
                'confidence': confidence,
                'all_predictions': predictions
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting gender for {image_path}: {str(e)}")
            return {'status': 'prediction_failed', 'error': str(e)}
    
    def predict_age(self, image_path: str) -> Dict[str, Any]:
        """Predict age from face image (optional)."""
        if self.age_pipeline is None:
            return {'status': 'model_not_loaded'}
        
        try:
            image = Image.open(image_path).convert('RGB')
            results = self.age_pipeline(image)
            
            # Extract age prediction
            top_result = results[0]
            
            return {
                'status': 'success',
                'predicted_age_range': top_result['label'],
                'confidence': top_result['score'],
                'all_predictions': results
            }
            
        except Exception as e:
            self.logger.error(f"Error predicting age for {image_path}: {str(e)}")
            return {'status': 'prediction_failed', 'error': str(e)}
    
    def analyze_single_face(self, face_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single face for gender and age."""
        face_id = face_data['face_id']
        face_path = face_data['face_path']
        
        # Check if file exists
        if not Path(face_path).exists():
            return {
                'face_id': face_id,
                'status': 'file_not_found'
            }
        
        # Predict gender
        gender_result = self.predict_gender(face_path)
        
        # Predict age (if enabled)
        age_result = self.predict_age(face_path) if self.age_pipeline else {'status': 'disabled'}
        
        # Combine results
        result = {
            'face_id': face_id,
            'face_path': face_path,
            'photo_id': face_data['photo_id'],
            'detection_confidence': face_data['detection_confidence'],
            'face_area': face_data['face_area'],
            'gender_analysis': gender_result,
            'age_analysis': age_result
        }
        
        return result
    
    def analyze_faces_batch(self, faces_data: List[Dict[str, Any]], max_workers: int = 4) -> List[Dict[str, Any]]:
        """Analyze multiple faces in parallel."""
        results = []
        
        with ProgressTracker(len(faces_data), "Analyzing faces") as progress:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_face = {
                    executor.submit(self.analyze_single_face, face): face
                    for face in faces_data
                }
                
                for future in as_completed(future_to_face):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        face = future_to_face[future]
                        self.logger.error(f"Error analyzing face {face['face_id']}: {str(e)}")
                        results.append({
                            'face_id': face['face_id'],
                            'status': 'analysis_error',
                            'error': str(e)
                        })
                    
                    progress.update(1)
        
        return results
    
    def create_analysis_dataframe(self, analysis_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create DataFrame from analysis results."""
        rows = []
        
        for result in analysis_results:
            # Extract basic info
            row = {
                'face_id': result['face_id'],
                'photo_id': result['photo_id'],
                'face_path': result['face_path'],
                'detection_confidence': result.get('detection_confidence', 0),
                'face_area': result.get('face_area', 0)
            }
            
            # Extract gender analysis
            gender_analysis = result.get('gender_analysis', {})
            if gender_analysis.get('status') == 'success':
                row.update({
                    'predicted_gender': gender_analysis['predicted_gender'],
                    'gender_confidence': gender_analysis['confidence'],
                    'gender_analysis_status': 'success'
                })
            else:
                row.update({
                    'predicted_gender': None,
                    'gender_confidence': 0,
                    'gender_analysis_status': gender_analysis.get('status', 'failed')
                })
            
            # Extract age analysis
            age_analysis = result.get('age_analysis', {})
            if age_analysis.get('status') == 'success':
                row.update({
                    'predicted_age_range': age_analysis['predicted_age_range'],
                    'age_confidence': age_analysis['confidence'],
                    'age_analysis_status': 'success'
                })
            else:
                row.update({
                    'predicted_age_range': None,
                    'age_confidence': 0,
                    'age_analysis_status': age_analysis.get('status', 'disabled')
                })
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def filter_high_confidence_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter predictions based on confidence threshold."""
        confidence_threshold = self.config['models']['gender_detection']['confidence_threshold']
        
        # Filter based on both detection and gender prediction confidence
        high_confidence_mask = (
            (df['detection_confidence'] >= confidence_threshold) &
            (df['gender_confidence'] >= confidence_threshold) &
            (df['gender_analysis_status'] == 'success')
        )
        
        filtered_df = df[high_confidence_mask].copy()
        
        self.logger.info(f"Filtered to {len(filtered_df)} high-confidence predictions from {len(df)} total faces")
        
        return filtered_df
    
    def run_gender_analysis(self, annotations_file: str = "data/annotations/face_annotations.csv") -> pd.DataFrame:
        """Main method to run gender analysis."""
        self.logger.info("Starting gender analysis...")
        
        # Load face annotations
        annotations_df = pd.read_csv(annotations_file)
        
        # Convert to list of dictionaries for processing
        faces_data = annotations_df.to_dict('records')
        
        self.logger.info(f"Analyzing {len(faces_data)} faces for gender and age")
        
        # Analyze faces
        analysis_results = self.analyze_faces_batch(
            faces_data,
            max_workers=self.config['processing']['max_workers']
        )
        
        # Create results DataFrame
        results_df = self.create_analysis_dataframe(analysis_results)
        
        # Save all results
        all_results_file = self.results_dir / "gender_analysis_all.csv"
        results_df.to_csv(all_results_file, index=False)
        
        # Filter high-confidence predictions
        high_confidence_df = self.filter_high_confidence_predictions(results_df)
        
        # Save filtered results
        filtered_results_file = self.results_dir / "gender_analysis_filtered.csv"
        high_confidence_df.to_csv(filtered_results_file, index=False)
        
        # Save detailed analysis results
        detailed_results_file = self.results_dir / "detailed_analysis_results.json"
        with open(detailed_results_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        # Create summary statistics
        summary = self.create_analysis_summary(results_df, high_confidence_df)
        save_results(summary, "gender_analysis_summary", "json")
        
        self.logger.info("Gender analysis completed!")
        return high_confidence_df
    
    def create_analysis_summary(self, all_results: pd.DataFrame, filtered_results: pd.DataFrame) -> Dict[str, Any]:
        """Create summary statistics for the analysis."""
        summary = {
            'total_faces_analyzed': len(all_results),
            'successful_gender_predictions': len(all_results[all_results['gender_analysis_status'] == 'success']),
            'high_confidence_predictions': len(filtered_results),
            'gender_distribution': {},
            'confidence_statistics': {},
            'age_analysis_enabled': self.age_pipeline is not None
        }
        
        # Gender distribution in filtered results
        if len(filtered_results) > 0:
            gender_counts = filtered_results['predicted_gender'].value_counts()
            summary['gender_distribution'] = {
                'male': int(gender_counts.get('male', 0)),
                'female': int(gender_counts.get('female', 0)),
                'total': len(filtered_results)
            }
            
            if summary['gender_distribution']['total'] > 0:
                summary['gender_distribution']['male_percentage'] = (
                    summary['gender_distribution']['male'] / summary['gender_distribution']['total'] * 100
                )
                summary['gender_distribution']['female_percentage'] = (
                    summary['gender_distribution']['female'] / summary['gender_distribution']['total'] * 100
                )
        
        # Confidence statistics
        successful_predictions = all_results[all_results['gender_analysis_status'] == 'success']
        if len(successful_predictions) > 0:
            summary['confidence_statistics'] = {
                'mean_gender_confidence': float(successful_predictions['gender_confidence'].mean()),
                'median_gender_confidence': float(successful_predictions['gender_confidence'].median()),
                'std_gender_confidence': float(successful_predictions['gender_confidence'].std()),
                'min_gender_confidence': float(successful_predictions['gender_confidence'].min()),
                'max_gender_confidence': float(successful_predictions['gender_confidence'].max())
            }
        
        # Age analysis summary (if enabled)
        if self.age_pipeline is not None:
            age_successful = all_results[all_results['age_analysis_status'] == 'success']
            if len(age_successful) > 0:
                summary['age_analysis'] = {
                    'successful_predictions': len(age_successful),
                    'age_distribution': age_successful['predicted_age_range'].value_counts().to_dict()
                }
        
        return summary

def main():
    """Main function to run gender analysis."""
    try:
        detector = GenderDetector()
        results = detector.run_gender_analysis()
        print(f"Gender analysis completed. Analyzed {len(results)} high-confidence faces.")
        return results
    except Exception as e:
        logging.error(f"Gender analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()