"""
Flask REST API for Autonomous Driving Object Detection
=====================================================
Production-ready API server for object detection services
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
import os
import time
import uuid
from PIL import Image
import tempfile
import json
from main import AutonomousVisionSystem

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Global variables
vision_system = None
upload_folder = "temp_uploads"
results_folder = "temp_results"

# Create necessary directories
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(results_folder, exist_ok=True)

def initialize_system():
    """Initialize the vision system"""
    global vision_system
    if vision_system is None:
        print("🔄 Initializing Autonomous Vision System...")
        vision_system = AutonomousVisionSystem()
        print("✅ System ready!")

def encode_image_to_base64(image_array):
    """Convert image array to base64 string"""
    _, buffer = cv2.imencode('.jpg', image_array)
    img_str = base64.b64encode(buffer).decode('utf-8')
    return img_str

def decode_base64_to_image(base64_string):
    """Convert base64 string to image array"""
    img_data = base64.b64decode(base64_string)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route('/', methods=['GET'])
def home():
    """API documentation endpoint"""
    return jsonify({
        "service": "Autonomous Driving Object Detection API",
        "version": "1.0.0",
        "author": "Your Name",
        "description": "Real-time object detection for autonomous vehicles",
        "endpoints": {
            "/health": "GET - System health check",
            "/detect": "POST - Detect objects in image",
            "/detect/batch": "POST - Batch process multiple images",
            "/stats": "GET - Get system statistics",
            "/config": "GET/POST - View/update configuration"
        },
        "usage": {
            "image_formats": ["jpg", "jpeg", "png", "bmp"],
            "max_file_size": "10MB",
            "supported_inputs": ["base64", "file_upload", "url"]
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """System health check endpoint"""
    try:
        initialize_system()
        
        # Test detection on a small dummy image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        start_time = time.time()
        _, _, _ = vision_system.process_frame(test_image)
        response_time = time.time() - start_time
        
        return jsonify({
            "status": "healthy",
            "timestamp": time.time(),
            "response_time_ms": round(response_time * 1000, 2),
            "system_info": {
                "device": str(vision_system.device),
                "model_loaded": vision_system.model is not None,
                "confidence_threshold": vision_system.confidence_threshold
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Main object detection endpoint"""
    try:
        initialize_system()
        
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Handle different input types
        image = None
        
        if 'image_base64' in data:
            # Base64 encoded image
            try:
                image = decode_base64_to_image(data['image_base64'])
            except Exception as e:
                return jsonify({"error": f"Invalid base64 image: {str(e)}"}), 400
                
        elif 'image_url' in data:
            # Image URL (for future implementation)
            return jsonify({"error": "URL input not yet implemented"}), 400
            
        else:
            return jsonify({"error": "No valid image input found. Use 'image_base64' or 'image_url'"}), 400
        
        if image is None:
            return jsonify({"error": "Failed to load image"}), 400
        
        # Optional parameters
        confidence = data.get('confidence', 0.5)
        include_image = data.get('include_processed_image', False)
        
        # Update confidence threshold
        vision_system.confidence_threshold = confidence
        vision_system.model.conf = confidence
        
        # Process the image
        start_time = time.time()
        annotated_frame, detections, risk_level = vision_system.process_frame(image)
        processing_time = time.time() - start_time
        
        # Prepare detection results
        detection_results = []
        category_counts = {}
        
        for detection in detections:
            x1, y1, x2, y2, conf, cls = detection
            class_name = vision_system.model.names[int(cls)]
            category = vision_system.categorize_detection(class_name)
            
            detection_info = {
                "class": class_name,
                "category": category,
                "confidence": float(conf),
                "bbox": {
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2)
                },
                "center": {
                    "x": int((x1 + x2) / 2),
                    "y": int((y1 + y2) / 2)
                }
            }
            
            detection_results.append(detection_info)
            category_counts[category] = category_counts.get(category, 0) + 1
        
        # Prepare response
        response = {
            "success": True,
            "timestamp": time.time(),
            "processing_time_ms": round(processing_time * 1000, 2),
            "image_info": {
                "width": image.shape[1],
                "height": image.shape[0],
                "channels": image.shape[2] if len(image.shape) > 2 else 1
            },
            "detection_summary": {
                "total_objects": len(detections),
                "risk_level": risk_level,
                "category_counts": category_counts,
                "confidence_threshold": confidence
            },
            "detections": detection_results
        }
        
        # Include processed image if requested
        if include_image:
            processed_image_b64 = encode_image_to_base64(annotated_frame)
            response["processed_image_base64"] = processed_image_b64
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": time.time()
        }), 500

@app.route('/detect/file', methods=['POST'])
def detect_objects_file():
    """File upload endpoint for object detection"""
    try:
        initialize_system()
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Optional parameters from form data
        confidence = float(request.form.get('confidence', 0.5))
        include_image = request.form.get('include_processed_image', 'false').lower() == 'true'
        
        # Save uploaded file temporarily
        file_id = str(uuid.uuid4())
        temp_path = os.path.join(upload_folder, f"{file_id}_{file.filename}")
        file.save(temp_path)
        
        try:
            # Read and process image
            image = cv2.imread(temp_path)
            if image is None:
                return jsonify({"error": "Invalid image file"}), 400
            
            # Update confidence threshold
            vision_system.confidence_threshold = confidence
            vision_system.model.conf = confidence
            
            # Process the image
            start_time = time.time()
            annotated_frame, detections, risk_level = vision_system.process_frame(image)
            processing_time = time.time() - start_time
            
            # Save processed image
            result_path = os.path.join(results_folder, f"result_{file_id}.jpg")
            cv2.imwrite(result_path, annotated_frame)
            
            # Prepare detection results (same as above)
            detection_results = []
            category_counts = {}
            
            for detection in detections:
                x1, y1, x2, y2, conf, cls = detection
                class_name = vision_system.model.names[int(cls)]
                category = vision_system.categorize_detection(class_name)
                
                detection_info = {
                    "class": class_name,
                    "category": category,
                    "confidence": float(conf),
                    "bbox": {
                        "x1": int(x1), "y1": int(y1),
                        "x2": int(x2), "y2": int(y2)
                    }
                }
                
                detection_results.append(detection_info)
                category_counts[category] = category_counts.get(category, 0) + 1
            
            response = {
                "success": True,
                "file_id": file_id,
                "original_filename": file.filename,
                "processing_time_ms": round(processing_time * 1000, 2),
                "result_image_url": f"/results/{file_id}",
                "detection_summary": {
                    "total_objects": len(detections),
                    "risk_level": risk_level,
                    "category_counts": category_counts
                },
                "detections": detection_results
            }
            
            return jsonify(response), 200
            
        finally:
            # Clean up uploaded file
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/results/<file_id>', methods=['GET'])
def get_result_image(file_id):
    """Serve processed result images"""
    try:
        result_path = os.path.join(results_folder, f"result_{file_id}.jpg")
        if not os.path.exists(result_path):
            return jsonify({"error": "Result image not found"}), 404
        
        return send_file(result_path, mimetype='image/jpeg')
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def get_statistics():
    """Get system performance statistics"""
    try:
        initialize_system()
        
        stats = {
            "system_status": "operational",
            "model_info": {
                "device": str(vision_system.device),
                "confidence_threshold": vision_system.confidence_threshold
            },
            "performance": {
                "total_requests": len(vision_system.processing_times) if vision_system.processing_times else 0,
                "average_processing_time_ms": round(
                    sum(vision_system.processing_times) / len(vision_system.processing_times) * 1000, 2
                ) if vision_system.processing_times else 0,
                "detection_counts": dict(vision_system.detection_stats)
            },
            "server_info": {
                "temp_files": len(os.listdir(results_folder)),
                "uptime": time.time()  # Simplified uptime
            }
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/config', methods=['GET', 'POST'])
def handle_config():
    """Get or update system configuration"""
    try:
        initialize_system()
        
        if request.method == 'GET':
            config = {
                "confidence_threshold": vision_system.confidence_threshold,
                "device": str(vision_system.device),
                "model_classes": list(vision_system.model.names.values()) if vision_system.model else [],
                "autonomous_categories": vision_system.autonomous_classes
            }
            return jsonify(config), 200
            
        elif request.method == 'POST':
            data = request.get_json()
            
            if 'confidence_threshold' in data:
                new_confidence = float(data['confidence_threshold'])
                if 0.1 <= new_confidence <= 1.0:
                    vision_system.confidence_threshold = new_confidence
                    vision_system.model.conf = new_confidence
                else:
                    return jsonify({"error": "Confidence must be between 0.1 and 1.0"}), 400
            
            return jsonify({
                "success": True,
                "message": "Configuration updated",
                "current_config": {
                    "confidence_threshold": vision_system.confidence_threshold
                }
            }), 200
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/health", "/detect", "/detect/file", "/stats", "/config"]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "message": "Please check the server logs for more details"
    }), 500

if __name__ == '__main__':
    print("🚗 Starting Autonomous Driving Detection API Server...")
    print("📋 Available endpoints:")
    print("   • GET  /          - API documentation")
    print("   • GET  /health    - Health check")
    print("   • POST /detect    - Object detection (JSON)")
    print("   • POST /detect/file - Object detection (file upload)")
    print("   • GET  /stats     - Performance statistics")
    print("   • GET/POST /config - Configuration management")
    print("\n🔗 API will be available at: http://localhost:5000")
    print("📚 Documentation: http://localhost:5000")
    
    # Initialize system on startup
    initialize_system()
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,  # Set to False for production
        threaded=True
    )
