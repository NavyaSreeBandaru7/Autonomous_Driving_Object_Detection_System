import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, Circle
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path

class AdvancedVisualizer:
    """
    Advanced visualization system for autonomous driving object detection results.
    
    Features:
    - Risk heatmaps
    - Detection analytics
    - Performance charts
    - Custom styling
    """
    
    def __init__(self, style='automotive'):
        """
        Initialize the visualizer with custom styling.
        
        Args:
            style (str): Visualization style ('automotive', 'technical', 'minimal')
        """
        self.style = style
        self.setup_style()
        
        # Define color schemes
        self.color_schemes = {
            'automotive': {
                'vehicle': '#2E8B57',      # Sea Green
                'person': '#DC143C',       # Crimson
                'traffic': '#4169E1',      # Royal Blue
                'obstacle': '#FF8C00',     # Dark Orange
                'background': '#1A1A1A',   # Dark Gray
                'text': '#FFFFFF'          # White
            },
            'technical': {
                'vehicle': '#00FF00',      # Bright Green
                'person': '#FF0000',       # Bright Red
                'traffic': '#0000FF',      # Bright Blue
                'obstacle': '#FFFF00',     # Bright Yellow
                'background': '#000000',   # Black
                'text': '#FFFFFF'          # White
            },
            'minimal': {
                'vehicle': '#404040',      # Dark Gray
                'person': '#606060',       # Medium Gray
                'traffic': '#808080',      # Light Gray
                'obstacle': '#A0A0A0',     # Lighter Gray
                'background': '#F5F5F5',   # Light Gray
                'text': '#000000'          # Black
            }
        }
        
        self.colors = self.color_schemes[style]
    
    def setup_style(self):
        """Setup matplotlib style based on selected theme"""
        if self.style == 'automotive':
            plt.style.use('dark_background')
            sns.set_palette("husl")
        elif self.style == 'technical':
            plt.style.use('default')
            sns.set_palette("bright")
        else:  # minimal
            plt.style.use('seaborn-v0_8-whitegrid')
            sns.set_palette("muted")
    
    def draw_enhanced_detections(self, image: np.ndarray, detections: List, 
                               class_names: Dict, risk_level: str = 'MEDIUM') -> np.ndarray:
        """
        Draw enhanced detection boxes with advanced styling.
        
        Args:
            image: Input image array
            detections: List of detection results
            class_names: Dictionary mapping class IDs to names
            risk_level: Current risk level
            
        Returns:
            Enhanced annotated image
        """
        annotated = image.copy()
        height, width = image.shape[:2]
        
        # Draw risk overlay
        overlay = self._create_risk_overlay(annotated, risk_level)
        annotated = cv2.addWeighted(annotated, 0.8, overlay, 0.2, 0)
        
        # Draw center driving lane
        self._draw_driving_lane(annotated)
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            class_name = class_names[int(cls)]
            category = self._categorize_detection(class_name)
            
            # Get color for category
            color_hex = self.colors.get(category, '#FFFFFF')
            color_bgr = self._hex_to_bgr(color_hex)
            
            # Draw main bounding box with gradient effect
            self._draw_gradient_box(annotated, (x1, y1), (x2, y2), color_bgr, conf)
            
            # Draw corner markers
            self._draw_corner_markers(annotated, (x1, y1), (x2, y2), color_bgr)
            
            # Draw confidence indicator
            self._draw_confidence_bar(annotated, (x1, y1), conf, color_bgr)
            
            # Draw label with enhanced styling
            self._draw_enhanced_label(annotated, (x1, y1), class_name, conf, color_bgr)
            
            # Draw risk indicator for people
            if category == 'person':
                self._draw_person_risk_indicator(annotated, (x1, y1, x2, y2))
        
        # Draw HUD overlay
        self._draw_hud_overlay(annotated, detections, risk_level)
        
        return annotated
    
    def _create_risk_overlay(self, image: np.ndarray, risk_level: str) -> np.ndarray:
        """Create a subtle risk level overlay"""
        overlay = np.zeros_like(image)
        height, width = image.shape[:2]
        
        risk_colors = {
            'LOW': (0, 255, 0),
            'MEDIUM': (0, 255, 255),
            'HIGH': (0, 165, 255),
            'CRITICAL': (0, 0, 255)
        }
        
        color = risk_colors.get(risk_level, (0, 255, 255))
        
        # Create subtle gradient from top
        for i in range(min(50, height)):
            alpha = (50 - i) / 50 * 0.1
            overlay[i, :] = [c * alpha for c in color]
        
        return overlay
    
    def _draw_driving_lane(self, image: np.ndarray):
        """Draw driving lane indicators"""
        height, width = image.shape[:2]
        center_x = width // 2
        lane_width = width // 3
        
        # Draw lane boundaries
        left_x = center_x - lane_width // 2
        right_x = center_x + lane_width // 2
        
        # Dashed lines for lane boundaries
        dash_length = 20
        gap_length = 10
        
        for y in range(0, height, dash_length + gap_length):
            cv2.line(image, (left_x, y), (left_x, min(y + dash_length, height)), 
                    (100, 100, 100), 2)
            cv2.line(image, (right_x, y), (right_x, min(y + dash_length, height)), 
                    (100, 100, 100), 2)
    
    def _draw_gradient_box(self, image: np.ndarray, pt1: Tuple[int, int], 
                          pt2: Tuple[int, int], color: Tuple[int, int, int], confidence: float):
        """Draw gradient bounding box"""
        x1, y1 = pt1
        x2, y2 = pt2
        
        # Main box
        thickness = max(2, int(confidence * 4))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Inner highlight
        highlight_color = tuple(min(255, c + 50) for c in color)
        cv2.rectangle(image, (x1 + thickness, y1 + thickness), 
                     (x2 - thickness, y2 - thickness), highlight_color, 1)
    
    def _draw_corner_markers(self, image: np.ndarray, pt1: Tuple[int, int], 
                           pt2: Tuple[int, int], color: Tuple[int, int, int]):
        """Draw corner markers for enhanced box appearance"""
        x1, y1 = pt1
        x2, y2 = pt2
        corner_length = 15
        
        # Top-left corner
        cv2.line(image, (x1, y1), (x1 + corner_length, y1), color, 3)
        cv2.line(image, (x1, y1), (x1, y1 + corner_length), color, 3)
        
        # Top-right corner
        cv2.line(image, (x2, y1), (x2 - corner_length, y1), color, 3)
        cv2.line(image, (x2, y1), (x2, y1 + corner_length), color, 3)
        
        # Bottom-left corner
        cv2.line(image, (x1, y2), (x1 + corner_length, y2), color, 3)
        cv2.line(image, (x1, y2), (x1, y2 - corner_length), color, 3)
        
        # Bottom-right corner
        cv2.line(image, (x2, y2), (x2 - corner_length, y2), color, 3)
        cv2.line(image, (x2, y2), (x2, y2 - corner_length), color, 3)
    
    def _draw_confidence_bar(self, image: np.ndarray, pt: Tuple[int, int], 
                           confidence: float, color: Tuple[int, int, int]):
        """Draw confidence level bar"""
        x, y = pt
        bar_width = 60
        bar_height = 6
        
        # Background bar
        cv2.rectangle(image, (x, y - 25), (x + bar_width, y - 25 + bar_height), 
                     (50, 50, 50), -1)
        
        # Confidence fill
        fill_width = int(bar_width * confidence)
        cv2.rectangle(image, (x, y - 25), (x + fill_width, y - 25 + bar_height), 
                     color, -1)
        
        # Border
        cv2.rectangle(image, (x, y - 25), (x + bar_width, y - 25 + bar_height), 
                     (255, 255, 255), 1)
    
    def _draw_enhanced_label(self, image: np.ndarray, pt: Tuple[int, int], 
                           class_name: str, confidence: float, color: Tuple[int, int, int]):
        """Draw enhanced label with background"""
        x, y = pt
        label = f"{class_name}: {confidence:.2f}"
        
        # Get text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (label_width, label_height), _ = cv2.getTextSize(label, font, font_scale, thickness)
        
        # Draw background with rounded corners effect
        padding = 8
        bg_color = tuple(int(c * 0.8) for c in color)  # Darker version of color
        
        # Main background
        cv2.rectangle(image, (x, y - label_height - padding * 2), 
                     (x + label_width + padding * 2, y), bg_color, -1)
        
        # Border
        cv2.rectangle(image, (x, y - label_height - padding * 2), 
                     (x + label_width + padding * 2, y), color, 2)
        
        # Text
        cv2.putText(image, label, (x + padding, y - padding), 
                   font, font_scale, (255, 255, 255), thickness)
    
    def _draw_person_risk_indicator(self, image: np.ndarray, bbox: Tuple[int, int, int, int]):
        """Draw special risk indicator for people"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Pulsing circle effect
        radius = 30
        cv2.circle(image, (center_x, center_y), radius, (0, 0, 255), 3)
        cv2.circle(image, (center_x, center_y), radius - 10, (0, 0, 255), 1)
        
        # Warning triangle
        triangle_size = 15
        triangle_pts = np.array([
            [center_x, center_y - triangle_size],
            [center_x - triangle_size, center_y + triangle_size],
            [center_x + triangle_size, center_y + triangle_size]
        ], np.int32)
        
        cv2.fillPoly(image, [triangle_pts], (0, 255, 255))
        cv2.putText(image, "!", (center_x - 5, center_y + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    def _draw_hud_overlay(self, image: np.ndarray, detections: List, risk_level: str):
        """Draw heads-up display overlay"""
        height, width = image.shape[:2]
        
        # Status panel background
        panel_height = 120
        panel_width = 250
        cv2.rectangle(image, (10, 10), (panel_width, panel_height), 
                     (0, 0, 0), -1)
        cv2.rectangle(image, (10, 10), (panel_width, panel_height), 
                     (100, 100, 100), 2)
        
        # Risk level
        risk_colors = {
            'LOW': (0, 255, 0),
            'MEDIUM': (0, 255, 255),
            'HIGH': (0, 165, 255),
            'CRITICAL': (0, 0, 255)
        }
        
        risk_color = risk_colors.get(risk_level, (0, 255, 255))
        cv2.putText(image, f"RISK: {risk_level}", (20, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, risk_color, 2)
        
        # Object count
        cv2.putText(image, f"OBJECTS: {len(detections)}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # System status
        cv2.putText(image, "SYSTEM: ACTIVE", (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Mini radar display
        self._draw_mini_radar(image, detections, (width - 150, 20))
    
    def _draw_mini_radar(self, image: np.ndarray, detections: List, center: Tuple[int, int]):
        """Draw mini radar display showing object positions"""
        cx, cy = center
        radar_size = 60
        
        # Radar background
        cv2.circle(image, (cx, cy), radar_size, (0, 50, 0), -1)
        cv2.circle(image, (cx, cy), radar_size, (0, 255, 0), 2)
        cv2.circle(image, (cx, cy), radar_size // 2, (0, 255, 0), 1)
        
        # Radar sweep line (static for simplicity)
        cv2.line(image, (cx, cy), (cx + radar_size, cy), (0, 255, 0), 1)
        
        # Plot objects
        height, width = image.shape[:2]
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            
            # Convert to radar coordinates
            obj_x = (x1 + x2) / 2 / width  # Normalize to 0-1
            obj_y = (y1 + y2) / 2 / height
            
            # Map to radar
            radar_x = int(cx + (obj_x - 0.5) * radar_size * 1.5)
            radar_y = int(cy + (obj_y - 0.5) * radar_size * 1.5)
            
            # Check if within radar bounds
            if ((radar_x - cx) ** 2 + (radar_y - cy) ** 2) <= radar_size ** 2:
                cv2.circle(image, (radar_x, radar_y), 3, (255, 255, 0), -1)
    
    def create_analytics_dashboard(self, detection_history: List[Dict], 
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create comprehensive analytics dashboard.
        
        Args:
            detection_history: List of detection results over time
            save_path: Optional path to save the dashboard
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('🚗 Autonomous Driving Detection Analytics', fontsize=16, fontweight='bold')
        
        # Extract data for analysis
        timestamps = [d.get('timestamp', 0) for d in detection_history]
        object_counts = [len(d.get('detections', [])) for d in detection_history]
        risk_levels = [d.get('risk_level', 'LOW') for d in detection_history]
        processing_times = [d.get('processing_time', 0) for d in detection_history]
        
        # 1. Object Detection Over Time
        axes[0, 0].plot(timestamps, object_counts, 'b-', linewidth=2, marker='o', markersize=4)
        axes[0, 0].set_title('Objects Detected Over Time')
        axes[0, 0].set_xlabel('Time')
        axes[0, 0].set_ylabel('Number of Objects')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Risk Level Distribution
        risk_counts = {level: risk_levels.count(level) for level in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']}
        colors = ['green', 'yellow', 'orange', 'red']
        axes[0, 1].pie(risk_counts.values(), labels=risk_counts.keys(), colors=colors, autopct='%1.1f%%')
        axes[0, 1].set_title('Risk Level Distribution')
        
        # 3. Processing Performance
        axes[0, 2].hist(processing_times, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
        axes[0, 2].set_title('Processing Time Distribution')
        axes[0, 2].set_xlabel('Processing Time (ms)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].axvline(np.mean(processing_times), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(processing_times):.1f}ms')
        axes[0, 2].legend()
        
        # 4. Category Detection Heatmap
        categories = ['vehicle', 'person', 'traffic', 'obstacle']
        category_data = []
        for category in categories:
            counts = []
            for detection_result in detection_history:
                count = sum(1 for d in detection_result.get('detections', []) 
                           if d.get('category') == category)
                counts.append(count)
            category_data.append(counts)
        
        im = axes[1, 0].imshow(category_data, cmap='YlOrRd', aspect='auto')
        axes[1, 0].set_title('Detection Heatmap by Category')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Object Categories')
        axes[1, 0].set_yticks(range(len(categories)))
        axes[1, 0].set_yticklabels(categories)
        plt.colorbar(im, ax=axes[1, 0])
        
        # 5. Confidence Score Analysis
        all_confidences = []
        for detection_result in detection_history:
            confidences = [d.get('confidence', 0) for d in detection_result.get('detections', [])]
            all_confidences.extend(confidences)
        
        if all_confidences:
            axes[1, 1].boxplot(all_confidences, patch_artist=True, 
                             boxprops=dict(facecolor='lightblue'))
            axes[1, 1].set_title('Confidence Score Distribution')
            axes[1, 1].set_ylabel('Confidence Score')
            axes[1, 1].set_xticklabels(['All Detections'])
        
        # 6. System Performance Metrics
        fps_data = [1000 / pt if pt > 0 else 0 for pt in processing_times]
        
        performance_metrics = {
            'Avg FPS': np.mean(fps_data),
            'Avg Objects': np.mean(object_counts),
            'Peak Objects': max(object_counts) if object_counts else 0,
            'Avg Processing (ms)': np.mean(processing_times)
        }
        
        metric_names = list(performance_metrics.keys())
        metric_values = list(performance_metrics.values())
        
        bars = axes[1, 2].bar(metric_names, metric_values, color=['blue', 'green', 'orange', 'red'])
        axes[1, 2].set_title('Performance Summary')
        axes[1, 2].set_ylabel('Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1, 2].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Analytics dashboard saved to: {save_path}")
        
        return fig
    
    def create_performance_report(self, detection_data: Dict, save_path: Optional[str] = None) -> str:
        """
        Generate detailed performance report.
        
        Args:
            detection_data: Dictionary containing detection statistics
            save_path: Optional path to save the report
            
        Returns:
            Report text content
        """
        report_lines = [
            "🚗 AUTONOMOUS DRIVING DETECTION PERFORMANCE REPORT",
            "=" * 60,
            "",
            f"📊 DETECTION SUMMARY",
            f"Total Frames Processed: {detection_data.get('total_frames', 0)}",
            f"Average FPS: {detection_data.get('average_fps', 0):.2f}",
            f"Total Objects Detected: {sum(detection_data.get('category_counts', {}).values())}",
            "",
            f"🎯 CATEGORY BREAKDOWN",
        ]
        
        for category, count in detection_data.get('category_counts', {}).items():
            percentage = (count / sum(detection_data.get('category_counts', {}).values())) * 100
            report_lines.append(f"  {category.capitalize()}: {count} ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            f"⚡ PERFORMANCE METRICS",
            f"Average Processing Time: {detection_data.get('avg_processing_time', 0):.1f}ms",
            f"Peak Processing Time: {detection_data.get('peak_processing_time', 0):.1f}ms",
            f"Memory Usage: {detection_data.get('memory_usage', 'N/A')}",
            "",
            f"🚨 RISK ASSESSMENT",
            f"Risk Level Distribution:",
        ])
        
        for risk, count in detection_data.get('risk_distribution', {}).items():
            report_lines.append(f"  {risk}: {count} occurrences")
        
        report_lines.extend([
            "",
            f"🔧 SYSTEM CONFIGURATION",
            f"Model: {detection_data.get('model_name', 'YOLOv5s')}",
            f"Confidence Threshold: {detection_data.get('confidence_threshold', 0.5)}",
            f"Device: {detection_data.get('device', 'CPU')}",
            "",
            f"📈 RECOMMENDATIONS",
            self._generate_recommendations(detection_data),
            "",
            "=" * 60,
            f"Report generated at: {detection_data.get('timestamp', 'N/A')}"
        ])
        
        report_text = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            print(f"📄 Performance report saved to: {save_path}")
        
        return report_text
    
    def _generate_recommendations(self, data: Dict) -> str:
        """Generate performance recommendations based on data"""
        recommendations = []
        
        avg_fps = data.get('average_fps', 0)
        if avg_fps < 15:
            recommendations.append("• Consider using a smaller model (YOLOv5s) for better performance")
            recommendations.append("• Enable GPU acceleration if available")
        
        avg_processing = data.get('avg_processing_time', 0)
        if avg_processing > 100:
            recommendations.append("• Reduce input image resolution for faster processing")
            recommendations.append("• Increase confidence threshold to reduce false positives")
        
        person_count = data.get('category_counts', {}).get('person', 0)
        total_objects = sum(data.get('category_counts', {}).values())
        
        if person_count / total_objects > 0.3 if total_objects > 0 else False:
            recommendations.append("• High pedestrian activity detected - consider enhanced safety protocols")
        
        if not recommendations:
            recommendations.append("• System performance is optimal")
            recommendations.append("• Consider fine-tuning for specific use cases")
        
        return "\n".join(recommendations)
    
    def _categorize_detection(self, class_name: str) -> str:
        """Categorize detection into autonomous driving categories"""
        autonomous_classes = {
            'vehicle': ['car', 'truck', 'bus', 'motorcycle', 'bicycle'],
            'person': ['person'],
            'traffic': ['traffic light', 'stop sign'],
            'obstacle': ['chair', 'bench', 'potted plant']
        }
        
        for category, classes in autonomous_classes.items():
            if class_name.lower() in classes:
                return category
        return 'other'
    
    def _hex_to_bgr(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to BGR tuple"""
        hex_color = hex_color.lstrip('#')
        rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return rgb[::-1]  # Convert RGB to BGR

# Convenience functions for easy use
def visualize_detections(image: np.ndarray, detections: List, class_names: Dict, 
                        style: str = 'automotive') -> np.ndarray:
    """
    Quick function to visualize detections with enhanced styling.
    
    Args:
        image: Input image
        detections: Detection results
        class_names: Class name mapping
        style: Visualization style
        
    Returns:
        Annotated image
    """
    visualizer = AdvancedVisualizer(style=style)
    return visualizer.draw_enhanced_detections(image, detections, class_names)

def create_dashboard(detection_history: List[Dict], save_path: str = 'analytics_dashboard.png'):
    """
    Quick function to create analytics dashboard.
    
    Args:
        detection_history: Historical detection data
        save_path: Path to save dashboard
    """
    visualizer = AdvancedVisualizer()
    fig = visualizer.create_analytics_dashboard(detection_history, save_path)
    return fig

def generate_report(detection_data: Dict, save_path: str = 'performance_report.txt') -> str:
    """
    Quick function to generate performance report.
    
    Args:
        detection_data: Detection statistics
        save_path: Path to save report
        
    Returns:
        Report text
    """
    visualizer = AdvancedVisualizer()
    return visualizer.create_performance_report(detection_data, save_path)
