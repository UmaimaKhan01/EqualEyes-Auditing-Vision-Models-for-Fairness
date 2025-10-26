import cv2
import numpy as np
from collections import deque
import time
from typing import Dict, List, Any
import threading
import queue
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.gender_detector import MultiModalBiasDetector

class RealTimeBiasMonitor:
    """Real-time bias monitoring for video streams and live content."""
    
    def __init__(self, bias_detector):
        self.bias_detector = bias_detector
        self.frame_queue = queue.Queue(maxsize=100)
        self.results_buffer = deque(maxlen=1000)  # Store last 1000 analyses
        self.monitoring = False
        self.bias_threshold = 0.6
        
    def monitor_video_stream(self, source=0, output_path=None):
        """Monitor bias in real-time video stream."""
        cap = cv2.VideoCapture(source)
        
        # Setup video writer if output path provided
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, 20.0, (640, 480))
        
        self.monitoring = True
        analysis_thread = threading.Thread(target=self._analysis_worker)
        analysis_thread.start()
        
        frame_count = 0
        
        while self.monitoring:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze every 30th frame to avoid overload
            if frame_count % 30 == 0:
                try:
                    self.frame_queue.put_nowait(frame.copy())
                except queue.Full:
                    pass  # Skip if queue is full
            
            # Draw bias overlay
            frame_with_overlay = self._draw_bias_overlay(frame)
            
            cv2.imshow('Bias Monitor', frame_with_overlay)
            
            if output_path:
                out.write(frame_with_overlay)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        self.monitoring = False
        analysis_thread.join()
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
    
    def _analysis_worker(self):
        """Background worker for bias analysis."""
        while self.monitoring:
            try:
                frame = self.frame_queue.get(timeout=1)
                
                # Detect faces and analyze bias
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                
                if len(faces) > 0:
                    # Analyze largest face
                    largest_face = max(faces, key=lambda x: x[2] * x[3])
                    x, y, w, h = largest_face
                    
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Convert to PIL Image for analysis
                    from PIL import Image
                    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                    
                    # Run bias analysis
                    results = self.bias_detector.analyze_intersectional_bias(pil_img)
                    
                    # Store results with timestamp
                    result_with_time = {
                        'timestamp': time.time(),
                        'face_count': len(faces),
                        'bias_results': results,
                        'alert_level': self._calculate_alert_level(results)
                    }
                    
                    self.results_buffer.append(result_with_time)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Analysis error: {e}")
    
    def _draw_bias_overlay(self, frame):
        """Draw bias information overlay on frame."""
        overlay = frame.copy()
        
        # Get recent bias data
        if self.results_buffer:
            recent_result = self.results_buffer[-1]
            
            # Draw bias score
            bias_score = recent_result.get('bias_results', {}).get('intersectional_score', 0)
            alert_level = recent_result.get('alert_level', 'LOW')
            
            # Color based on alert level
            color = {'LOW': (0, 255, 0), 'MEDIUM': (0, 255, 255), 'HIGH': (0, 0, 255)}.get(alert_level, (255, 255, 255))
            
            # Draw bias info
            cv2.putText(overlay, f"Bias Score: {bias_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(overlay, f"Alert: {alert_level}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw trend graph
            self._draw_bias_trend(overlay)
        
        return overlay
    
    def _draw_bias_trend(self, frame, start_pos=(400, 50), size=(200, 100)):
        """Draw real-time bias trend graph."""
        if len(self.results_buffer) < 2:
            return
        
        # Get last 50 bias scores
        recent_scores = [r.get('bias_results', {}).get('intersectional_score', 0) 
                        for r in list(self.results_buffer)[-50:]]
        
        if not recent_scores:
            return
        
        x, y = start_pos
        w, h = size
        
        # Draw graph background
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 1)
        
        # Draw trend line
        max_score = max(recent_scores) if recent_scores else 1
        for i in range(1, len(recent_scores)):
            x1 = x + int((i - 1) * w / len(recent_scores))
            y1 = y + h - int(recent_scores[i - 1] / max_score * h)
            x2 = x + int(i * w / len(recent_scores))
            y2 = y + h - int(recent_scores[i] / max_score * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    def _calculate_alert_level(self, results: Dict) -> str:
        """Calculate alert level based on bias results."""
        intersectional_score = results.get('intersectional_score', 0)
        
        if intersectional_score > 0.7:
            return 'HIGH'
        elif intersectional_score > 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        if not self.results_buffer:
            return {'error': 'No monitoring data available'}
        
        results = list(self.results_buffer)
        
        # Calculate statistics
        bias_scores = [r.get('bias_results', {}).get('intersectional_score', 0) for r in results]
        alert_levels = [r.get('alert_level', 'LOW') for r in results]
        
        report = {
            'monitoring_duration': results[-1]['timestamp'] - results[0]['timestamp'],
            'total_frames_analyzed': len(results),
            'average_bias_score': np.mean(bias_scores),
            'max_bias_score': max(bias_scores),
            'min_bias_score': min(bias_scores),
            'alert_distribution': {
                'HIGH': alert_levels.count('HIGH'),
                'MEDIUM': alert_levels.count('MEDIUM'),
                'LOW': alert_levels.count('LOW')
            },
            'bias_trend': bias_scores[-100:],  # Last 100 scores for trending
            'recommendations': self._generate_recommendations(bias_scores, alert_levels)
        }
        
        return report
    
    def _generate_recommendations(self, bias_scores: List[float], alert_levels: List[str]) -> List[str]:
        """Generate actionable recommendations based on monitoring data."""
        recommendations = []
        
        avg_bias = np.mean(bias_scores)
        high_alerts = alert_levels.count('HIGH')
        
        if avg_bias > 0.6:
            recommendations.append("High average bias detected. Review content diversity policies.")
        
        if high_alerts > len(alert_levels) * 0.2:
            recommendations.append("Frequent high-bias alerts. Implement real-time content filtering.")
        
        if len(set(bias_scores[-10:])) < 3:
            recommendations.append("Low bias variance detected. May indicate limited demographic representation.")
        
        return recommendations

# Usage example
def start_real_time_monitoring():
    """Start real-time bias monitoring session."""
    bias_detector = MultiModalBiasDetector()
    monitor = RealTimeBiasMonitor(bias_detector)
    
    print("Starting real-time bias monitoring...")
    print("Press 'q' to quit")
    
    # Try camera first, fallback to demo mode
    try:
        # Test if camera is available
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise Exception("No camera available")
        cap.release()
        
        # Monitor webcam if available
        monitor.monitor_video_stream(source=0, output_path="bias_monitoring_output.avi")
        
    except Exception as e:
        print(f"Camera not available ({e}), running demo mode...")
        
        # Create demo monitoring data
        demo_results = []
        for i in range(10):
            demo_result = {
                'timestamp': time.time() + i,
                'face_count': 1,
                'bias_results': {
                    'gender': {'prediction': 'female' if i % 2 else 'male', 'confidence': 0.8 + (i * 0.02)},
                    'intersectional_score': 0.3 + (i * 0.05)
                },
                'alert_level': 'LOW' if i < 7 else 'MEDIUM'
            }
            demo_results.append(demo_result)
        
        # Simulate monitoring session
        monitor.results_buffer.extend(demo_results)
        print(f"Demo monitoring completed with {len(demo_results)} simulated frames")
    
    # Generate final report
    report = monitor.generate_monitoring_report()
    
    with open("monitoring_report.json", "w") as f:
        import json
        json.dump(report, f, indent=2, default=str)
    
    print("Monitoring session completed. Report saved to monitoring_report.json")
    
    # Display report summary
    if 'error' not in report:
        print(f"\nMonitoring Summary:")
        print(f"- Duration: {report.get('monitoring_duration', 0):.1f} seconds")
        print(f"- Frames analyzed: {report.get('total_frames_analyzed', 0)}")
        print(f"- Average bias: {report.get('average_bias_score', 0):.2f}")
        print(f"- Alert distribution: {report.get('alert_distribution', {})}")

if __name__ == "__main__":
    start_real_time_monitoring()