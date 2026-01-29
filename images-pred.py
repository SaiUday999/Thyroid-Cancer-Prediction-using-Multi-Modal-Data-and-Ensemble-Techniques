def predict_image(model_path, image_path, conf_threshold=0.25):
    """
    Perform prediction on a single image using the trained YOLO model.
    
    Args:
        model_path (str): Path to the trained model weights
        image_path (str): Path to the image for prediction
        conf_threshold (float): Confidence threshold for detections
        
    Returns:
        results: YOLO prediction results
    """
    from ultralytics import YOLO
    import cv2
    from matplotlib import pyplot as plt
    import numpy as np
    
    # Load the model
    model = YOLO(model_path)
    
    # Perform prediction
    results = model.predict(image_path, conf=conf_threshold)
    
    # Display the results
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(10, 10))
    
    # Plot original image
    plt.imshow(img)
    
    # Extract detection information
    if len(results) > 0:
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
            plt.gca().add_patch(rect)
            
            # Add label
            plt.text(
                x1, y1-10, 
                f"{class_name}: {conf:.2f}", 
                color='white', 
                fontsize=12,
                bbox=dict(facecolor='red', alpha=0.5)
            )
    
    plt.axis('off')
    plt.title('Prediction Results')
    plt.show()
    
    return results

# Example usage:
if __name__ == "__main__":
    # Path to your trained model
    MODEL_PATH = "thyroid/best.pt"
    
    # Single image prediction
    IMAGE_PATH = ""
    results = predict_image(MODEL_PATH, IMAGE_PATH, conf_threshold=0.25)
    