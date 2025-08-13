import unittest
import numpy as np
import cv2
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
import sys
import pytest
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from main import AutonomousVisionSystem
from utils.visualization import AdvancedVisualizer, visualize_detections

class TestAutonomousVisionSystem(unittest.TestCase):
    """Test cases for the main detection system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.mock_detections = np.array([
            [100, 100, 200, 200, 0.8, 0],  # car
            [300, 150, 350, 250, 0.9, 1],  # person
            [50, 50, 100, 100, 0.7, 2],    # bicycle
        ])
        
    @patch('torch.hub.load')
    def test_system_initialization(self, mock_torch_load):
        """Test system initialization"""
        # Mock the model
        mock_model = MagicMock()
        mock_model.names = {0: 'car', 1: 'person', 2: 'bicycle'}
        mock_torch_load.return_value = mock_model
        
        # Initialize system
        system = AutonomousVisionSystem(confidence_threshold=0.6)
        
        # Assertions
        self.assertEqual(system.confidence_threshold, 0.6)
        self.assertIsNotNone(system.model)
        self.assertIn('vehicle', system.autonomous_classes)
        self.assertIn('person', system.autonomous_classes)
        
    def test_categorize_detection(self):
        """Test object categorization"""
        with patch('torch.hub.load'):
            system = AutonomousVisionSystem()
            
            # Test vehicle categorization
            self.assertEqual(system.categorize_detection('car'), 'vehicle')
            self.assertEqual(system.categorize_detection('truck'), 'vehicle')
            self.assertEqual(system.categorize_detection('motorcycle'), 'vehicle')
            
            # Test person categorization
            self.assertEqual(system.categorize_detection('person'), 'person')
            
            # Test traffic categorization
            self.assertEqual(system.categorize_detection('traffic light'), 'traffic')
            self.assertEqual(system.categorize_detection('stop sign'), 'traffic')
            
            # Test unknown object
            self.assertEqual(system.categorize_detection('unknown'), 'other')
    
    def test_risk_level_calculation(self):
        """Test risk level calculation logic"""
        with patch('torch.hub.load'):
            system = AutonomousVisionSystem()
            system.model = MagicMock()
            system.model.names = {0: 'car', 1: 'person', 2: 'bicycle'}
            
            frame_shape = (480, 640, 3)
            
            # Test low risk scenario (no objects in center)
            low_risk_detections = np.array([
                [50, 100, 100, 200, 0.8, 0],  # car on left
            ])
            risk = system.calculate_risk_level(low_risk_detections, frame_shape)
            self.assertEqual(risk, 'LOW')
            
            # Test high risk scenario (person in center)
            high_risk_detections = np.array([
                [300, 150, 350, 250, 0.9, 1],  # person in center
            ])
            risk = system.calculate_risk_level(high_risk_detections, frame_shape)
            self.assertIn(risk, ['MEDIUM', 'HIGH', 'CRITICAL'])
    
    @patch('torch.hub.load')
    def test_process_frame(self, mock_torch_load):
        """Test frame processing functionality"""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.names = {0: 'car', 1: 'person', 2: 'bicycle'}
        mock_results = MagicMock()
        mock_results.xyxy = [self.mock_detections]
        mock_model.return_value = mock_results
        mock_torch_load.return_value = mock_model
        
        # Initialize system and process frame
        system = AutonomousVisionSystem()
        annotated_frame, detections, risk_level = system.process_frame(self.test_image)
        
        # Assertions
        self.assertEqual(annotated_frame.shape, self.test_image.shape)
        self.assertIsInstance(detections, np.ndarray)
        self.assertIn(risk_level, ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
        self.assertGreater(len(system.processing_times), 0)
    
    def test_image_processing(self):
        """Test image file processing"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            # Create and save test image
            cv2.imwrite(temp_file.name, self.test_image)
            
            with patch('torch.hub.load'):
                system = AutonomousVisionSystem()
                
                # Mock the process_frame method
                system.process_frame = MagicMock(return_value=(
                    self.test_image, self.mock_detections, 'MEDIUM'
                ))
                
                # Test image processing
                result_image, detections, risk = system.process_image(temp_file.name)
                
                # Assertions
                self.assertIsInstance(result_image, np.ndarray)
                self.assertEqual(risk, 'MEDIUM')
            
            # Cleanup
            os.unlink(temp_file.name)
    
    def test_invalid_image_handling(self):
        """Test handling of invalid image files"""
        with patch('torch.hub.load'):
            system = AutonomousVisionSystem()
            
            # Test with non-existent file
            with self.assertRaises(ValueError):
                system.process_image('non_existent_file.jpg')
    
    def test_report_generation(self):
        """Test performance report generation"""
        with patch('torch.hub.load'):
            system = AutonomousVisionSystem()
            
            # Add some dummy processing times
            system.processing_times = [0.03, 0.04, 0.035, 0.042]
            system.detection_stats = {'vehicle': 10, 'person': 5, 'traffic': 2}
            
            report = system.generate_report()
            
            # Assertions
            self.assertIsInstance(report, dict)
            self.assertIn('total_frames_processed', report)
            self.assertIn('average_fps', report)
            self.assertIn('detection_statistics', report)
            
            # Check if report file was created
            self.assertTrue(os.path.exists('detection_report.json'))
            
            # Cleanup
            if os.path.exists('detection_report.json'):
                os.remove('detection_report.json')

class TestAdvancedVisualizer(unittest.TestCase):
    """Test cases for the visualization system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.test_detections = [
            [100, 100, 200, 200, 0.8, 0],
            [300, 150, 350, 250, 0.9, 1],
        ]
        self.test_class_names = {0: 'car', 1: 'person'}
    
    def test_visualizer_initialization(self):
        """Test visualizer initialization with different styles"""
        # Test automotive style
        viz_auto = AdvancedVisualizer(style='automotive')
        self.assertEqual(viz_auto.style, 'automotive')
        self.assertIn('vehicle', viz_auto.colors)
        
        # Test technical style
        viz_tech = AdvancedVisualizer(style='technical')
        self.assertEqual(viz_tech.style, 'technical')
        
        # Test minimal style
        viz_min = AdvancedVisualizer(style='minimal')
        self.assertEqual(viz_min.style, 'minimal')
    
    def test_detection_visualization(self):
        """Test detection visualization functionality"""
        visualizer = AdvancedVisualizer()
        
        annotated_image = visualizer.draw_enhanced_detections(
            self.test_image, self.test_detections, self.test_class_names, 'MEDIUM'
        )
        
        # Assertions
        self.assertEqual(annotated_image.shape, self.test_image.shape)
        self.assertIsInstance(annotated_image, np.ndarray)
        
        # Check that image was modified (not identical to original)
        self.assertFalse(np.array_equal(annotated_image, self.test_image))
    
    def test_color_conversion(self):
        """Test hex to BGR color conversion"""
        visualizer = AdvancedVisualizer()
        
        # Test red color conversion
        bgr_color = visualizer._hex_to_bgr('#FF0000')
        self.assertEqual(bgr_color, (0, 0, 255))  # BGR format
        
        # Test green color conversion
        bgr_color = visualizer._hex_to_bgr('#00FF00')
        self.assertEqual(
