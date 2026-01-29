import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Thyroid Cancer Analysis Tool",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# Add the current directory to path (to ensure all imports work correctly)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import main function from test.py for thyroid risk prediction
from thyroid_risk import main as thyroid_risk_main

def predict_image(model_path, uploaded_image, conf_threshold=0.25):
    """
    Perform prediction on an uploaded image using the trained YOLO model.
    
    Args:
        model_path (str): Path to the trained model weights
        uploaded_image: Uploaded image file from Streamlit
        conf_threshold (float): Confidence threshold for detections
        
    Returns:
        results: YOLO prediction results
        img_with_boxes: Image with bounding boxes drawn
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        st.error("The ultralytics package is not installed. Please install it using: pip install ultralytics")
        return None, None
    
    # Load the model
    try:
        model = YOLO(model_path)
    except Exception as e:
        st.error(f"Error loading the model: {str(e)}")
        return None, None
    
    # Convert the uploaded file to an image
    image_bytes = uploaded_image.getvalue()
    
    # Save temporary image
    temp_img_path = os.path.join(current_dir, "temp_image.jpg")
    with open(temp_img_path, "wb") as f:
        f.write(image_bytes)
    
    # Perform prediction
    try:
        results = model.predict(temp_img_path, conf=conf_threshold)
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        os.remove(temp_img_path)
        return None, None
    
    # Read and process image
    img = cv2.imread(temp_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Clean up temporary file
    os.remove(temp_img_path)
    
    # Create a figure for drawing
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img)
    
    # Extract detection information
    detection_results = []
    if len(results) > 0 and hasattr(results[0], 'boxes'):
        boxes = results[0].boxes
        for box in boxes:
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Get class and confidence
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            # Get class name
            class_name = 'TC' if cls == 0 else 'Normal'
            
            # Draw bounding box
            rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            
            # Add label
            ax.text(
                x1, y1-10, 
                f"{class_name}: {conf:.2f}", 
                color='white', 
                fontsize=12,
                bbox=dict(facecolor='red', alpha=0.5)
            )
            
            # Add to results
            detection_results.append({
                'class': class_name,
                'confidence': conf,
                'coordinates': (x1, y1, x2, y2)
            })
    
    ax.axis('off')
    ax.set_title('Prediction Results')
    
    # Convert the matplotlib figure to an image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img_with_boxes = Image.open(buf)
    plt.close(fig)
    
    return detection_results, img_with_boxes

def image_prediction_page():
    """Page for thyroid image prediction using YOLO model"""
    st.markdown("<h1 style='text-align: center; color: #0f4c81;'>Thyroid Ultrasound Image Analysis</h1>", unsafe_allow_html=True)
    
    st.write("""
    This tool analyzes thyroid ultrasound images to detect potential thyroid abnormalities using a trained YOLO model.
    Upload a thyroid ultrasound image to get started.
    """)
    
    # Model path
    model_path = os.path.join(current_dir, "best.pt")
    
    # Check if model exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}. Please ensure the YOLO model file 'best.pt' is in the thyroid directory.")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a thyroid ultrasound image", type=["jpg", "jpeg", "png"])
    
    # Confidence threshold
    conf_threshold = st.slider(
        "Detection Confidence Threshold", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.25, 
        step=0.05,
        help="Minimum confidence required for a detection to be shown"
    )
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Predict button
        if st.button("Analyze Image"):
            with st.spinner("Analyzing image..."):
                # Perform prediction
                detection_results, img_with_boxes = predict_image(model_path, uploaded_file, conf_threshold)
                
                if detection_results is not None and img_with_boxes is not None:
                    # Display image with detections
                    st.image(img_with_boxes, caption="Detection Results", use_column_width=True)
                    
                    # Display detection details in a table
                    if detection_results:
                        st.subheader("Detection Details")
                        detection_df = {
                            "Class": [d['class'] for d in detection_results],
                            "Confidence": [f"{d['confidence']:.2%}" for d in detection_results]
                        }
                        st.dataframe(detection_df)
                        
                        # Show summary
                        tc_detections = sum(1 for d in detection_results if d['class'] == 'TC')
                        if tc_detections > 0:
                            st.warning(f"⚠️ Potential thyroid cancer detected with {tc_detections} region(s) of concern.")
                        else:
                            st.success("✅ No potential thyroid cancer regions detected.")
                    else:
                        st.info("No detections found in the image at the current confidence threshold.")
                        
                    # Information about the model
                    with st.expander("About the Detection Model"):
                        st.markdown("""
                        This tool uses a YOLOv8 model trained on thyroid ultrasound images. The model identifies:
                        - **TC**: Potential thyroid cancer regions
                        - **Normal**: Normal thyroid tissue
                        
                        The confidence score indicates how certain the model is about each detection.
                        """)
                else:
                    st.error("An error occurred during image analysis. Please try another image or adjust the confidence threshold.")
        
    # Add information about appropriate images
    with st.expander("Image Requirements"):
        st.markdown("""
        For best results:
        - Use clear ultrasound images of thyroid
        - Images should be in JPG, JPEG, or PNG format
        - Ensure the image shows the thyroid region clearly
        - Avoid images with excessive artifacts or noise
        """)

def setup_app():
    """Configure app settings and page structure"""
    # Add custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #0f4c81;
            text-align: center;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #0f4c81;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0 0;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0f4c81;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    setup_app()
    
    # Create tabs for different functionalities
    tab1, tab2 = st.tabs(["Thyroid Cancer Risk Assessment", "Thyroid Ultrasound Analysis"])
    
    with tab1:
        # Run the thyroid risk assessment application
        thyroid_risk_main()
        
    with tab2:
        # Run the image prediction application
        image_prediction_page()

if __name__ == "__main__":
    main() 