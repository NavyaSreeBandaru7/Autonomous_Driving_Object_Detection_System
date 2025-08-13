import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import io
import time
import json
from main import AutonomousVisionSystem

# Page configuration
st.set_page_config(
    page_title="🚗 Autonomous Driving Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-indicator {
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    .risk-low { background-color: #d4edda; color: #155724; }
    .risk-medium { background-color: #fff3cd; color: #856404; }
    .risk-high { background-color: #f8d7da; color: #721c24; }
    .risk-critical { background-color: #721c24; color: white; }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_detection_system():
    """Load and cache the detection system"""
    with st.spinner("🔄 Loading AI model... This may take a moment on first run"):
        return AutonomousVisionSystem()

def display_risk_indicator(risk_level):
    """Display risk level with appropriate styling"""
    risk_classes = {
        'LOW': 'risk-low',
        'MEDIUM': 'risk-medium', 
        'HIGH': 'risk-high',
        'CRITICAL': 'risk-critical'
    }
    
    risk_class = risk_classes.get(risk_level, 'risk-low')
    st.markdown(f"""
    <div class="risk-indicator {risk_class}">
        🚨 RISK LEVEL: {risk_level}
    </div>
    """, unsafe_allow_html=True)

def process_uploaded_image(vision_system, uploaded_file, confidence):
    """Process an uploaded image file"""
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    
    # Convert RGB to BGR for OpenCV
    if len(image_array.shape) == 3:
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    
    # Update confidence threshold
    vision_system.confidence_threshold = confidence
    vision_system.model.conf = confidence
    
    # Process the image
    with st.spinner("🔍 Analyzing image for objects..."):
        start_time = time.time()
        annotated_frame, detections, risk_level = vision_system.process_frame(image_array)
        processing_time = time.time() - start_time
    
    # Convert back to RGB for display
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    
    return annotated_frame, detections, risk_level, processing_time

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚗 Autonomous Driving Object Detection</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This interactive demo showcases real-time object detection for autonomous vehicles.
    Upload an image to see how our AI system identifies vehicles, pedestrians, traffic signs, and potential obstacles.
    """)
    
    # Load the detection system
    vision_system = load_detection_system()
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎛️ Detection Settings")
        
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Lower values detect more objects but may include false positives"
        )
        
        st.header("📊 Detection Categories")
        st.markdown("""
        - 🟢 **Vehicles**: Cars, trucks, motorcycles, bicycles
        - 🔴 **People**: Pedestrians and cyclists
        - 🔵 **Traffic**: Traffic lights and stop signs  
        - 🟡 **Obstacles**: Road obstacles and debris
        """)
        
        st.header("🎯 Risk Assessment")
        st.markdown("""
        Risk levels are calculated based on:
        - Object types (pedestrians = highest risk)
        - Position in driving path
        - Number of detected objects
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload a driving scene image for object detection"
        )
        
        # Sample images section
        st.subheader("🖼️ Try Sample Images")
        sample_options = {
            "Urban Street Scene": "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=800",
            "Highway Traffic": "https://images.unsplash.com/photo-1558618047-3c8c76ca7d13?w=800",
            "Pedestrian Crossing": "https://images.unsplash.com/photo-1573348722427-f1d6819fdf98?w=800"
        }
        
        selected_sample = st.selectbox("Select a sample image:", list(sample_options.keys()))
        
        if st.button("🔄 Load Sample Image"):
            st.info(f"Loading {selected_sample}...")
            # You would implement sample image loading here
    
    with col2:
        st.header("🔍 Detection Results")
        
        if uploaded_file is not None:
            # Process the uploaded image
            try:
                annotated_image, detections, risk_level, processing_time = process_uploaded_image(
                    vision_system, uploaded_file, confidence
                )
                
                # Display results
                st.image(annotated_image, caption="Detected Objects", use_column_width=True)
                
                # Risk level indicator
                display_risk_indicator(risk_level)
                
                # Performance metrics
                col_metric1, col_metric2, col_metric3 = st.columns(3)
                
                with col_metric1:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h4>⚡ Processing Time</h4>
                        <h2>{processing_time*1000:.1f}ms</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_metric2:
                    st.markdown(f"""
                    <div class="metric-box">
                        <h4>🎯 Objects Found</h4>
                        <h2>{len(detections)}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_metric3:
                    fps = 1.0 / processing_time if processing_time > 0 else 0
                    st.markdown(f"""
                    <div class="metric-box">
                        <h4>📊 Estimated FPS</h4>
                        <h2>{fps:.1f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed detection breakdown
                if len(detections) > 0:
                    st.subheader("📋 Detection Details")
                    
                    detection_data = []
                    for i, detection in enumerate(detections):
                        x1, y1, x2, y2, conf, cls = detection
                        class_name = vision_system.model.names[int(cls)]
                        category = vision_system.categorize_detection(class_name)
                        
                        detection_data.append({
                            "Object": class_name.title(),
                            "Category": category.title(),
                            "Confidence": f"{conf:.2f}",
                            "Position": f"({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})"
                        })
                    
                    st.table(detection_data)
                    
                    # Download processed image
                    img_buffer = io.BytesIO()
                    Image.fromarray(annotated_image).save(img_buffer, format='PNG')
                    
                    st.download_button(
                        label="📥 Download Processed Image",
                        data=img_buffer.getvalue(),
                        file_name=f"detected_{uploaded_file.name}",
                        mime="image/png"
                    )
                
            except Exception as e:
                st.error(f"❌ Error processing image: {str(e)}")
                st.info("Please try uploading a different image or adjusting the confidence threshold.")
        
        else:
            st.info("👆 Upload an image to see the object detection in action!")
            
            # Show example detection
            st.subheader("🎬 Example Detection")
            st.markdown("""
            Here's what you can expect to see:
            - **Green boxes**: Vehicles (cars, trucks, motorcycles)
            - **Red boxes**: People (highest priority for safety)
            - **Blue boxes**: Traffic elements (lights, signs)
            - **Yellow boxes**: Obstacles and other objects
            """)
    
    # Additional features section
    st.markdown("---")
    
    col_feature1, col_feature2, col_feature3 = st.columns(3)
    
    with col_feature1:
        st.header("🎥 Video Processing")
        st.markdown("""
        **Coming Soon**: Upload driving videos for frame-by-frame analysis
        - Real-time risk assessment
        - Object tracking across frames
        - Performance analytics
        """)
        
        if st.button("🔜 Enable Video Mode"):
            st.info("Video processing feature will be available in the next update!")
    
    with col_feature2:
        st.header("📊 Analytics Dashboard")
        st.markdown("""
        **Performance Insights**:
        - Detection accuracy metrics
        - Processing speed analysis
        - Risk level distribution
        """)
        
        if st.button("📈 View Analytics"):
            # Generate sample analytics
            st.subheader("📊 Detection Statistics")
            
            sample_stats = {
                "Vehicles Detected": 156,
                "Pedestrians Detected": 43,
                "Average Confidence": 0.78,
                "Average Processing Time": "45ms"
            }
            
            for metric, value in sample_stats.items():
                st.metric(metric, value)
    
    with col_feature3:
        st.header("⚙️ Advanced Settings")
        st.markdown("""
        **Customization Options**:
        - Model selection (YOLOv5s/m/l/x)
        - Custom object classes
        - Export configurations
        """)
        
        with st.expander("🔧 Advanced Configuration"):
            model_variant = st.selectbox(
                "Model Variant",
                ["yolov5s (Fast)", "yolov5m (Balanced)", "yolov5l (Accurate)"],
                help="Larger models are more accurate but slower"
            )
            
            enable_tracking = st.checkbox(
                "Enable Object Tracking",
                help="Track objects across multiple frames (video only)"
            )
            
            export_format = st.selectbox(
                "Export Format",
                ["PNG", "JPG", "PDF Report"],
                help="Choose output format for processed images"
            )
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        🚗 Built with ❤️ for Autonomous Driving Research<br>
        Powered by YOLOv5 • OpenCV • PyTorch • Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
