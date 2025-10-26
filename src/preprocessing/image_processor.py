import cv2
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
from PIL import Image, ImageEnhance
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request
import os

from ..utils import load_config, ProgressTracker, save_results, get_device

class ImagePreprocessor:
    """Preprocess images for gender detection analysis using OpenCV DNN face detector."""
    
    def __init__(self):
        self.config = load_config()
        self.device = get_device()
        
        # Setup directories
        self.processed_dir = Path("data/processed")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.faces_dir = self.processed_dir / "faces"
        self.faces_dir.mkdir(exist_ok=True)
        
        self.annotations_dir = Path("data/annotations")
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup models directory
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize OpenCV DNN face detector
        self.face_net = self.load_face_detector()
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_face_detector(self):
        """Skip face detector loading to avoid segfault - use simple processing."""
        print("Skipping face detector loading to avoid segfault")
        return None
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and validate image."""
        try:
            # Load with PIL first for better format support
            pil_image = Image.open(image_path)
            
            # Convert to RGB if necessary
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array for OpenCV
            image = np.array(pil_image)
            
            # Validate image
            if image.shape[0] < 224 or image.shape[1] < 224:
                self.logger.warning(f"Image too small: {image_path}")
                return None
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image {image_path}: {str(e)}")
            return None
    
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Apply image enhancement for better face detection."""
        pil_image = Image.fromarray(image)
        
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.2)
        
        # Enhance brightness slightly
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        return np.array(pil_image)
    
    def detect_faces(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Simple face detection without DNN to avoid segfault."""
        try:
            # Use simple Haar cascade instead of DNN
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            face_list = []
            for (x, y, w, h) in faces:
                face_info = {
                    'bbox': [x, y, w, h],
                    'confidence': 0.9,  # Default confidence
                    'landmarks': None,
                    'area': w * h
                }
                face_list.append(face_info)
            
            # Sort by area (largest faces first)
            face_list.sort(key=lambda x: x['area'], reverse=True)
            
            return face_list
            
        except Exception as e:
            self.logger.error(f"Face detection error: {str(e)}")
            return []
    
    def extract_face(self, image: np.ndarray, bbox: List[int], padding: float = 0.2) -> np.ndarray:
        """Extract face from image with padding."""
        x, y, w, h = bbox
        
        # Add padding
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        # Calculate expanded box
        x1 = max(0, x - pad_w)
        y1 = max(0, y - pad_h)
        x2 = min(image.shape[1], x + w + pad_w)
        y2 = min(image.shape[0], y + h + pad_h)
        
        # Extract face
        face = image[y1:y2, x1:x2]
        
        # Resize to standard size
        target_size = self.config['processing']['image_size']
        face_resized = cv2.resize(face, (target_size, target_size))
        
        return face_resized
    
    def process_single_image(self, photo_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single image for face detection and extraction."""
        image_path = photo_metadata.get('local_path')
        photo_id = photo_metadata['id']
        
        if not image_path or not Path(image_path).exists():
            return {
                'photo_id': photo_id,
                'status': 'file_not_found',
                'faces': []
            }
        
        # Load image
        image = self.load_image(image_path)
        if image is None:
            return {
                'photo_id': photo_id,
                'status': 'load_failed',
                'faces': []
            }
        
        # Enhance image for better detection
        enhanced_image = self.enhance_image(image)
        
        # Detect faces
        faces = self.detect_faces(enhanced_image)
        
        processed_faces = []
        for i, face_info in enumerate(faces):
            try:
                # Extract face
                face_image = self.extract_face(enhanced_image, face_info['bbox'])
                
                # Save face image
                face_filename = f"{photo_id}_face_{i}.jpg"
                face_path = self.faces_dir / face_filename
                cv2.imwrite(str(face_path), cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                
                face_data = {
                    'face_id': f"{photo_id}_face_{i}",
                    'face_path': str(face_path),
                    'bbox': face_info['bbox'],
                    'confidence': face_info['confidence'],
                    'landmarks': face_info['landmarks'],
                    'area': face_info['area']
                }
                
                processed_faces.append(face_data)
                
            except Exception as e:
                self.logger.error(f"Error extracting face {i} from {photo_id}: {str(e)}")
                continue
        
        return {
            'photo_id': photo_id,
            'original_path': image_path,
            'status': 'success' if processed_faces else 'no_faces',
            'faces': processed_faces,
            'total_faces': len(processed_faces),
            'image_shape': image.shape
        }
    
    def process_images_batch(self, photos_metadata: List[Dict[str, Any]], max_workers: int = 4) -> List[Dict[str, Any]]:
        """Process multiple images in parallel."""
        results = []
        
        with ProgressTracker(len(photos_metadata), "Processing images") as progress:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_photo = {
                    executor.submit(self.process_single_image, photo): photo
                    for photo in photos_metadata
                }
                
                for future in as_completed(future_to_photo):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        photo = future_to_photo[future]
                        self.logger.error(f"Error processing photo {photo['id']}: {str(e)}")
                        results.append({
                            'photo_id': photo['id'],
                            'status': 'processing_error',
                            'faces': []
                        })
                    
                    progress.update(1)
        
        return results
    
    def create_annotations(self, processing_results: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create annotations DataFrame from processing results."""
        annotations = []
        
        for result in processing_results:
            if result['status'] == 'success':
                for face in result['faces']:
                    annotation = {
                        'photo_id': result['photo_id'],
                        'face_id': face['face_id'],
                        'face_path': face['face_path'],
                        'bbox_x': face['bbox'][0],
                        'bbox_y': face['bbox'][1],
                        'bbox_w': face['bbox'][2],
                        'bbox_h': face['bbox'][3],
                        'detection_confidence': face['confidence'],
                        'face_area': face['area'],
                        'status': 'detected'
                    }
                    annotations.append(annotation)
        
        df = pd.DataFrame(annotations)
        return df
    
    def run_preprocessing(self, metadata_file: str = "data/raw/flickr30k_metadata.json") -> pd.DataFrame:
        """Main method to run image preprocessing."""
        self.logger.info("Starting image preprocessing...")
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            photos_metadata = json.load(f)
        
        # Filter only successfully downloaded photos
        valid_photos = [
            photo for photo in photos_metadata 
            if photo.get('download_status') == 'success'
        ]
        
        self.logger.info(f"Processing {len(valid_photos)} valid photos")
        
        # Process images
        processing_results = self.process_images_batch(
            valid_photos,
            max_workers=self.config['processing']['max_workers']
        )
        
        # Create annotations
        annotations_df = self.create_annotations(processing_results)
        
        # Save results
        annotations_file = self.annotations_dir / "face_annotations.csv"
        annotations_df.to_csv(annotations_file, index=False)
        
        processing_summary_file = self.processed_dir / "processing_results.json"
        with open(processing_summary_file, 'w') as f:
            json.dump(processing_results, f, indent=2, default=str)
        
        # Create summary statistics
        summary = {
            'total_images_processed': len(processing_results),
            'images_with_faces': len([r for r in processing_results if r['status'] == 'success']),
            'total_faces_detected': len(annotations_df),
            'average_faces_per_image': len(annotations_df) / len(valid_photos) if valid_photos else 0,
            'processing_success_rate': len([r for r in processing_results if r['status'] == 'success']) / len(processing_results) if processing_results else 0
        }
        
        save_results(summary, "preprocessing_summary", "json")
        
        self.logger.info(f"Preprocessing completed! Detected {len(annotations_df)} faces from {len(valid_photos)} images")
        return annotations_df

def main():
    """Main function to run preprocessing."""
    try:
        preprocessor = ImagePreprocessor()
        annotations = preprocessor.run_preprocessing()
        print(f"Preprocessing completed. Found {len(annotations)} faces.")
        return annotations
    except Exception as e:
        logging.error(f"Preprocessing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()