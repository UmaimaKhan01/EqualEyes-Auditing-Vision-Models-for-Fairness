import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import json
from scipy import stats
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from ..utils import load_config, save_results

class BiasAnalyzer:
    """Analyze gender bias in the dataset and model predictions."""
    
    def __init__(self):
        self.config = load_config()
        self.results_dir = Path("results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_analysis_data(self, results_file: str = "results/gender_analysis_filtered.csv") -> pd.DataFrame:
        """Load gender analysis results."""
        try:
            df = pd.read_csv(results_file)
            self.logger.info(f"Loaded {len(df)} analysis results")
            return df
        except FileNotFoundError:
            self.logger.error(f"Results file not found: {results_file}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading analysis data: {str(e)}")
            raise
    
    def calculate_representation_ratio(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate gender representation ratios."""
        gender_counts = df['predicted_gender'].value_counts()
        total = len(df)
        
        if total == 0:
            return {'error': 'No data available'}
        
        representation = {
            'total_faces': total,
            'male_count': int(gender_counts.get('male', 0)),
            'female_count': int(gender_counts.get('female', 0)),
            'male_percentage': gender_counts.get('male', 0) / total * 100,
            'female_percentage': gender_counts.get('female', 0) / total * 100,
            'representation_ratio': gender_counts.get('male', 0) / max(gender_counts.get('female', 1), 1),
            'is_balanced': abs(gender_counts.get('male', 0) - gender_counts.get('female', 0)) / total < 0.1
        }
        
        return representation
    
    def analyze_confidence_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze confidence score distributions by gender."""
        confidence_analysis = {}
        
        for gender in ['male', 'female']:
            gender_data = df[df['predicted_gender'] == gender]['gender_confidence']
            
            if len(gender_data) > 0:
                confidence_analysis[gender] = {
                    'count': len(gender_data),
                    'mean': float(gender_data.mean()),
                    'median': float(gender_data.median()),
                    'std': float(gender_data.std()),
                    'min': float(gender_data.min()),
                    'max': float(gender_data.max()),
                    'q25': float(gender_data.quantile(0.25)),
                    'q75': float(gender_data.quantile(0.75))
                }
            else:
                confidence_analysis[gender] = {'count': 0}
        
        # Statistical test for confidence difference
        male_conf = df[df['predicted_gender'] == 'male']['gender_confidence']
        female_conf = df[df['predicted_gender'] == 'female']['gender_confidence']
        
        if len(male_conf) > 0 and len(female_conf) > 0:
            try:
                t_stat, p_value = stats.ttest_ind(male_conf, female_conf)
                confidence_analysis['statistical_test'] = {
                    'test': 'independent_t_test',
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant_difference': p_value < 0.05,
                    'interpretation': 'Significant difference in confidence' if p_value < 0.05 else 'No significant difference'
                }
            except Exception as e:
                confidence_analysis['statistical_test'] = {'error': str(e)}
        
        return confidence_analysis
    
    def analyze_age_gender_intersection(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze intersection of age and gender predictions."""
        if 'predicted_age_range' not in df.columns or df['predicted_age_range'].isna().all():
            return {'status': 'age_data_unavailable'}
        
        # Create cross-tabulation
        crosstab = pd.crosstab(df['predicted_age_range'], df['predicted_gender'], margins=True)
        
        # Convert to dictionary with percentages
        intersection_analysis = {
            'crosstab_counts': crosstab.to_dict(),
            'crosstab_percentages': (crosstab / crosstab.loc['All', 'All'] * 100).to_dict()
        }
        
        # Analyze potential bias patterns
        bias_patterns = []
        
        for age_range in crosstab.index[:-1]:  # Exclude 'All' row
            row_data = crosstab.loc[age_range]
            total_in_age = row_data['All']
            
            if total_in_age > 10:  # Only analyze age groups with sufficient data
                male_pct = row_data.get('male', 0) / total_in_age * 100
                female_pct = row_data.get('female', 0) / total_in_age * 100
                
                if abs(male_pct - female_pct) > 20:  # More than 20% difference
                    bias_patterns.append({
                        'age_range': age_range,
                        'male_percentage': male_pct,
                        'female_percentage': female_pct,
                        'bias_direction': 'male' if male_pct > female_pct else 'female',
                        'severity': 'high' if abs(male_pct - female_pct) > 40 else 'moderate'
                    })
        
        intersection_analysis['bias_patterns'] = bias_patterns
        intersection_analysis['has_age_gender_bias'] = len(bias_patterns) > 0
        
        return intersection_analysis
    
    def calculate_demographic_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate overall demographic distribution and balance."""
        distribution = {}
        
        # Gender distribution
        distribution['gender'] = df['predicted_gender'].value_counts().to_dict()
        
        # Age distribution (if available)
        if 'predicted_age_range' in df.columns and not df['predicted_age_range'].isna().all():
            distribution['age'] = df['predicted_age_range'].value_counts().to_dict()
        
        # Face area distribution (as proxy for image prominence)
        face_areas = df['face_area']
        distribution['face_area_stats'] = {
            'mean': float(face_areas.mean()),
            'median': float(face_areas.median()),
            'std': float(face_areas.std()),
            'quartiles': [float(face_areas.quantile(q)) for q in [0.25, 0.5, 0.75]]
        }
        
        # Detection confidence distribution
        detection_conf = df['detection_confidence']
        distribution['detection_confidence_stats'] = {
            'mean': float(detection_conf.mean()),
            'median': float(detection_conf.median()),
            'std': float(detection_conf.std()),
            'quartiles': [float(detection_conf.quantile(q)) for q in [0.25, 0.5, 0.75]]
        }
        
        return distribution
    
    def analyze_model_bias_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze potential bias patterns in model predictions."""
        bias_patterns = {}
        
        # 1. Confidence bias by gender
        male_data = df[df['predicted_gender'] == 'male']
        female_data = df[df['predicted_gender'] == 'female']
        
        if len(male_data) > 0 and len(female_data) > 0:
            bias_patterns['confidence_bias'] = {
                'male_avg_confidence': float(male_data['gender_confidence'].mean()),
                'female_avg_confidence': float(female_data['gender_confidence'].mean()),
                'confidence_gap': float(male_data['gender_confidence'].mean() - female_data['gender_confidence'].mean()),
                'potential_bias': abs(male_data['gender_confidence'].mean() - female_data['gender_confidence'].mean()) > 0.05
            }
        
        # 2. Face size bias
        if len(male_data) > 0 and len(female_data) > 0:
            bias_patterns['face_size_bias'] = {
                'male_avg_face_area': float(male_data['face_area'].mean()),
                'female_avg_face_area': float(female_data['face_area'].mean()),
                'face_area_gap': float(male_data['face_area'].mean() - female_data['face_area'].mean()),
                'potential_bias': abs(male_data['face_area'].mean() - female_data['face_area'].mean()) > 1000
            }
        
        # 3. Detection confidence bias
        if len(male_data) > 0 and len(female_data) > 0:
            bias_patterns['detection_confidence_bias'] = {
                'male_avg_detection_conf': float(male_data['detection_confidence'].mean()),
                'female_avg_detection_conf': float(female_data['detection_confidence'].mean()),
                'detection_conf_gap': float(male_data['detection_confidence'].mean() - female_data['detection_confidence'].mean()),
                'potential_bias': abs(male_data['detection_confidence'].mean() - female_data['detection_confidence'].mean()) > 0.05
            }
        
        return bias_patterns
    
    def generate_bias_score(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall bias score and recommendations."""
        bias_indicators = []
        bias_score = 0
        max_score = 100
        
        # 1. Representation bias (40 points)
        representation = analysis_results.get('representation_ratio', {})
        if not representation.get('is_balanced', True):
            representation_penalty = min(40, abs(representation.get('male_percentage', 50) - 50) * 0.8)
            bias_score += representation_penalty
            bias_indicators.append({
                'type': 'representation_imbalance',
                'severity': 'high' if representation_penalty > 20 else 'moderate',
                'description': f"Gender representation is imbalanced: {representation.get('male_percentage', 0):.1f}% male, {representation.get('female_percentage', 0):.1f}% female"
            })
        
        # 2. Confidence bias (30 points)
        model_bias = analysis_results.get('model_bias_patterns', {})
        confidence_bias = model_bias.get('confidence_bias', {})
        if confidence_bias.get('potential_bias', False):
            confidence_penalty = min(30, abs(confidence_bias.get('confidence_gap', 0)) * 300)
            bias_score += confidence_penalty
            bias_indicators.append({
                'type': 'confidence_bias',
                'severity': 'high' if confidence_penalty > 15 else 'moderate',
                'description': f"Model shows different confidence levels for genders (gap: {confidence_bias.get('confidence_gap', 0):.3f})"
            })
        
        # 3. Age-gender intersection bias (20 points)
        age_gender = analysis_results.get('age_gender_intersection', {})
        if age_gender.get('has_age_gender_bias', False):
            age_bias_penalty = min(20, len(age_gender.get('bias_patterns', [])) * 10)
            bias_score += age_bias_penalty
            bias_indicators.append({
                'type': 'age_gender_intersection_bias',
                'severity': 'moderate',
                'description': f"Found {len(age_gender.get('bias_patterns', []))} age-gender bias patterns"
            })
        
        # 4. Face size bias (10 points)
        face_size_bias = model_bias.get('face_size_bias', {})
        if face_size_bias.get('potential_bias', False):
            face_size_penalty = min(10, abs(face_size_bias.get('face_area_gap', 0)) / 1000)
            bias_score += face_size_penalty
            bias_indicators.append({
                'type': 'face_size_bias',
                'severity': 'low',
                'description': f"Different average face sizes detected for genders"
            })
        
        # Generate recommendations
        recommendations = []
        
        if bias_score > 50:
            recommendations.extend([
                "Consider collecting more balanced training data",
                "Implement bias mitigation techniques in model training",
                "Use stratified sampling for evaluation metrics"
            ])
        elif bias_score > 25:
            recommendations.extend([
                "Monitor model performance across demographic groups",
                "Consider data augmentation for underrepresented groups"
            ])
        else:
            recommendations.append("Continue monitoring for potential bias in future datasets")
        
        bias_assessment = {
            'overall_bias_score': bias_score,
            'max_possible_score': max_score,
            'bias_percentage': bias_score / max_score * 100,
            'bias_level': (
                'high' if bias_score > 50 else
                'moderate' if bias_score > 25 else
                'low'
            ),
            'bias_indicators': bias_indicators,
            'recommendations': recommendations
        }
        
        return bias_assessment
    
    def run_bias_analysis(self, results_file: str = "results/gender_analysis_filtered.csv") -> Dict[str, Any]:
        """Main method to run comprehensive bias analysis."""
        self.logger.info("Starting bias analysis...")
        
        # Load data
        df = self.load_analysis_data(results_file)
        
        if len(df) == 0:
            self.logger.warning("No data available for bias analysis")
            return {'error': 'no_data_available'}
        
        # Run all analyses
        analysis_results = {}
        
        # 1. Representation analysis
        self.logger.info("Analyzing representation ratios...")
        analysis_results['representation_ratio'] = self.calculate_representation_ratio(df)
        
        # 2. Confidence distribution analysis
        self.logger.info("Analyzing confidence distributions...")
        analysis_results['confidence_distribution'] = self.analyze_confidence_distribution(df)
        
        # 3. Age-gender intersection analysis
        self.logger.info("Analyzing age-gender intersections...")
        analysis_results['age_gender_intersection'] = self.analyze_age_gender_intersection(df)
        
        # 4. Demographic distribution
        self.logger.info("Calculating demographic distributions...")
        analysis_results['demographic_distribution'] = self.calculate_demographic_distribution(df)
        
        # 5. Model bias patterns
        self.logger.info("Analyzing model bias patterns...")
        analysis_results['model_bias_patterns'] = self.analyze_model_bias_patterns(df)
        
        # 6. Generate overall bias assessment
        self.logger.info("Generating bias score and recommendations...")
        analysis_results['bias_assessment'] = self.generate_bias_score(analysis_results)
        
        # Add metadata
        analysis_results['metadata'] = {
            'analysis_date': pd.Timestamp.now().isoformat(),
            'total_faces_analyzed': len(df),
            'dataset_source': 'flickr',
            'confidence_threshold': self.config['models']['gender_detection']['confidence_threshold']
        }
        
        # Save results
        save_results(analysis_results, "bias_analysis_complete", "json")
        
        # Save summary
        summary = {
            'bias_level': analysis_results['bias_assessment']['bias_level'],
            'bias_score': analysis_results['bias_assessment']['bias_percentage'],
            'key_findings': [indicator['description'] for indicator in analysis_results['bias_assessment']['bias_indicators']],
            'recommendations': analysis_results['bias_assessment']['recommendations']
        }
        
        save_results(summary, "bias_analysis_summary", "json")
        
        self.logger.info("Bias analysis completed!")
        return analysis_results

def main():
    """Main function to run bias analysis."""
    try:
        analyzer = BiasAnalyzer()
        results = analyzer.run_bias_analysis()
        
        if 'error' not in results:
            bias_level = results['bias_assessment']['bias_level']
            bias_score = results['bias_assessment']['bias_percentage']
            print(f"Bias analysis completed. Bias level: {bias_level} ({bias_score:.1f}%)")
        
        return results
    except Exception as e:
        logging.error(f"Bias analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()