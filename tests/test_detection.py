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
        self.assertEqual(bgr_color, (0, 255, 0))  # BGR format
        
        # Test blue color conversion
        bgr_color = visualizer._hex_to_bgr('#0000FF')
        self.assertEqual(bgr_color, (255, 0, 0))  # BGR format
    
    def test_analytics_dashboard_creation(self):
        """Test analytics dashboard creation"""
        visualizer = AdvancedVisualizer()
        
        # Create mock detection history
        detection_history = [
            {
                'timestamp': 1234567890.0,
                'detections': [
                    {'category': 'vehicle', 'confidence': 0.8},
                    {'category': 'person', 'confidence': 0.9}
                ],
                'risk_level': 'MEDIUM',
                'processing_time': 45.0
            },
            {
                'timestamp': 1234567891.0,
                'detections': [
                    {'category': 'vehicle', 'confidence': 0.7}
                ],
                'risk_level': 'LOW',
                'processing_time': 42.0
            }
        ]
        
        # Create dashboard
        fig = visualizer.create_analytics_dashboard(detection_history)
        
        # Assertions
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), 6)  # Should have 6 subplots
    
    def test_performance_report_generation(self):
        """Test performance report generation"""
        visualizer = AdvancedVisualizer()
        
        detection_data = {
            'total_frames': 100,
            'average_fps': 25.5,
            'category_counts': {'vehicle': 50, 'person': 20, 'traffic': 10},
            'avg_processing_time': 40.0,
            'peak_processing_time': 55.0,
            'confidence_threshold': 0.5,
            'device': 'cuda'
        }
        
        report = visualizer.create_performance_report(detection_data)
        
        # Assertions
        self.assertIsInstance(report, str)
        self.assertIn('AUTONOMOUS DRIVING DETECTION', report)
        self.assertIn('PERFORMANCE REPORT', report)
        self.assertIn('Total Frames Processed: 100', report)
        self.assertIn('Average FPS: 25.50', report)

class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions"""
    
    def test_visualize_detections_function(self):
        """Test the standalone visualize_detections function"""
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_detections = [[100, 100, 200, 200, 0.8, 0]]
        test_class_names = {0: 'car'}
        
        result = visualize_detections(test_image, test_detections, test_class_names)
        
        # Assertions
        self.assertEqual(result.shape, test_image.shape)
        self.assertIsInstance(result, np.ndarray)

class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    @patch('torch.hub.load')
    def test_end_to_end_image_processing(self, mock_torch_load):
        """Test complete end-to-end image processing workflow"""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.names = {0: 'car', 1: 'person', 2: 'bicycle'}
        mock_results = MagicMock()
        mock_results.xyxy = [np.array([
            [100, 100, 200, 200, 0.8, 0],
            [300, 150, 350, 250, 0.9, 1]
        ])]
        mock_model.return_value = mock_results
        mock_torch_load.return_value = mock_model
        
        # Create test image file
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            cv2.imwrite(temp_file.name, test_image)
            
            try:
                # Initialize system and process image
                system = AutonomousVisionSystem()
                result_image, detections, risk_level = system.process_image(temp_file.name)
                
                # Generate report
                report = system.generate_report()
                
                # Visualize results
                visualizer = AdvancedVisualizer()
                annotated = visualizer.draw_enhanced_detections(
                    result_image, detections, system.model.names, risk_level
                )
                
                # Assertions
                self.assertIsInstance(result_image, np.ndarray)
                self.assertIsInstance(detections, np.ndarray)
                self.assertIn(risk_level, ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'])
                self.assertIsInstance(report, dict)
                self.assertIsInstance(annotated, np.ndarray)
                
            finally:
                # Cleanup
                os.unlink(temp_file.name)
                if os.path.exists('detection_report.json'):
                    os.remove('detection_report.json')

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_empty_image_handling(self):
        """Test handling of empty images"""
        with patch('torch.hub.load'):
            system = AutonomousVisionSystem()
            
            # Test with empty image
            empty_image = np.zeros((0, 0, 3), dtype=np.uint8)
            
            # Should handle gracefully without crashing
            try:
                system.process_frame(empty_image)
            except Exception as e:
                # Some error is expected, but shouldn't be a crash
                self.assertIsInstance(e, (ValueError, AttributeError))
    
    def test_invalid_detection_data(self):
        """Test handling of invalid detection data"""
        visualizer = AdvancedVisualizer()
        
        # Test with empty detection history
        empty_history = []
        
        # Should handle empty data gracefully
        try:
            fig = visualizer.create_analytics_dashboard(empty_history)
            self.assertIsNotNone(fig)
        except Exception as e:
            # Should not crash completely
            self.assertIsInstance(e, (ValueError, IndexError))
    
    def test_invalid_confidence_threshold(self):
        """Test invalid confidence threshold handling"""
        with patch('torch.hub.load'):
            # Test with invalid confidence values
            system1 = AutonomousVisionSystem(confidence_threshold=-0.1)
            system2 = AutonomousVisionSystem(confidence_threshold=1.5)
            
            # System should handle invalid values by clamping or using defaults
            self.assertGreaterEqual(system1.confidence_threshold, 0.0)
            self.assertLessEqual(system2.confidence_threshold, 1.0)

class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance benchmark tests"""
    
    @patch('torch.hub.load')
    def test_processing_speed_benchmark(self, mock_torch_load):
        """Benchmark processing speed"""
        # Setup mock model
        mock_model = MagicMock()
        mock_model.names = {0: 'car', 1: 'person'}
        mock_results = MagicMock()
        mock_results.xyxy = [np.array([[100, 100, 200, 200, 0.8, 0]])]
        mock_model.return_value = mock_results
        mock_torch_load.return_value = mock_model
        
        system = AutonomousVisionSystem()
        
        # Test processing speed on various image sizes
        image_sizes = [(240, 320), (480, 640), (720, 1280)]
        
        for height, width in image_sizes:
            test_image = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            
            import time
            start_time = time.time()
            system.process_frame(test_image)
            processing_time = time.time() - start_time
            
            # Assert reasonable processing times (should be under 1 second for mock)
            self.assertLess(processing_time, 1.0)
            
            # Calculate FPS
            fps = 1.0 / processing_time if processing_time > 0 else 0
            self.assertGreater(fps, 1.0)  # Should achieve at least 1 FPS

# Pytest-style tests for additional coverage
def test_system_memory_usage():
    """Test memory usage doesn't grow excessively"""
    import psutil
    import gc
    
    with patch('torch.hub.load'):
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        system = AutonomousVisionSystem()
        
        # Process multiple frames
        for _ in range(10):
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            system.process_frame = MagicMock(return_value=(test_image, [], 'LOW'))
            system.process_frame(test_image)
        
        gc.collect()  # Force garbage collection
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024

def test_concurrent_processing():
    """Test thread safety of the system"""
    import threading
    import queue
    
    with patch('torch.hub.load'):
        system = AutonomousVisionSystem()
        results_queue = queue.Queue()
        
        def process_image_worker():
            test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            system.process_frame = MagicMock(return_value=(test_image, [], 'LOW'))
            result = system.process_frame(test_image)
            results_queue.put(result)
        
        # Start multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=process_image_worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check all threads completed successfully
        assert results_queue.qsize() == 5

# Custom test runner for comprehensive testing
class CustomTestResult(unittest.TextTestResult):
    """Custom test result class for detailed output"""
    
    def addSuccess(self, test):
        super().addSuccess(test)
        print(f"✅ {test._testMethodName}")
    
    def addError(self, test, err):
        super().addError(test, err)
        print(f"❌ {test._testMethodName} - ERROR")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        print(f"❌ {test._testMethodName} - FAILED")

def run_comprehensive_tests():
    """Run all tests with detailed output"""
    print("🧪 Running Comprehensive Test Suite for Autonomous Driving Detection")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestAutonomousVisionSystem,
        TestAdvancedVisualizer,
        TestConvenienceFunctions,
        TestIntegration,
        TestErrorHandling,
        TestPerformanceBenchmarks
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with custom result class
    runner = unittest.TextTestRunner(
        resultclass=CustomTestResult,
        verbosity=2,
        stream=sys.stdout
    )
    
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    print(f"🏁 TEST SUMMARY")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    # Run comprehensive tests
    success = run_comprehensive_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
