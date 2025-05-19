import cv2
import numpy as np

def detect_arrow_direction(roi):
    """Detect arrow direction in a blue box."""
    # Convert to grayscale and threshold to get the arrow shape
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    # Find contours of the arrow
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"
    
    # Find the largest contour (should be the arrow)
    arrow_contour = max(contours, key=cv2.contourArea)
    
    # Get the extreme points of the contour
    leftmost = tuple(arrow_contour[arrow_contour[:, :, 0].argmin()][0])
    rightmost = tuple(arrow_contour[arrow_contour[:, :, 0].argmax()][0])
    topmost = tuple(arrow_contour[arrow_contour[:, :, 1].argmin()][0])
    bottommost = tuple(arrow_contour[arrow_contour[:, :, 1].argmax()][0])
    
    # Get the center of the contour
    M = cv2.moments(arrow_contour)
    if M["m00"] == 0:
        return "unknown"
    
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Calculate distances from center to extreme points
    distances = {
        "up": cy - topmost[1],
        "down": bottommost[1] - cy,
        "left": cx - leftmost[0],
        "right": rightmost[0] - cx
    }
    
    # The direction with maximum distance is likely the arrow direction
    direction = max(distances, key=distances.get)
    return direction

def count_dots(roi):
    """Count the number of dots in a small box."""
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    
    # Find contours of dots
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours to count only the dots (small circular contours)
    dot_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if 5 < area < 100:  # Adjust these thresholds based on your image
            dot_count += 1
    
    return dot_count

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Define the expected regions for blue boxes and dot boxes
    # These coordinates need to be adjusted based on your specific image
    blue_boxes = [
        (10, 10, 100, 100),    # (x, y, width, height) for first blue box
        (120, 10, 100, 100),   # second blue box
        (230, 10, 100, 100),   # third blue box
        (340, 10, 100, 100)    # fourth blue box
    ]
    
    dot_boxes = [
        (10, 150, 100, 50),    # (x, y, width, height) for first dot box
        (120, 150, 100, 50),   # second dot box
        (230, 150, 100, 50),   # third dot box
        (340, 150, 100, 50)    # fourth dot box
    ]
    
    results = []
    
    # Process each blue box and its corresponding dot box
    for i, (blue_box, dot_box) in enumerate(zip(blue_boxes, dot_boxes)):
        # Extract blue box ROI
        x, y, w, h = blue_box
        blue_roi = image[y:y+h, x:x+w]
        
        # Detect arrow direction
        direction = detect_arrow_direction(blue_roi)
        
        # Extract dot box ROI
        x, y, w, h = dot_box
        dot_roi = image[y:y+h, x:x+w]
        
        # Count dots
        dot_count = count_dots(dot_roi)
        
        # Calculate the action
        if dot_count > 0:
            action = f"{direction} * {dot_count} (repeat {direction} {dot_count} times)"
        else:
            action = direction
        
        results.append(action)
        print(f"Box {i+1}: Arrow direction = {direction}, Dots = {dot_count}, Action = {action}")
    
    # For visualization, you can draw the results on the image
    visualize_results(image, blue_boxes, dot_boxes, results)
    
    return results

def visualize_results(image, blue_boxes, dot_boxes, results):
    """Visualize the detection results on the image."""
    output = image.copy()
    
    # Draw blue boxes and their detected directions
    for i, (box, result) in enumerate(zip(blue_boxes, results)):
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(output, result.split()[0], (x, y-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Draw dot boxes and the dot counts
    for i, box in enumerate(dot_boxes):
        x, y, w, h = box
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow("Detection Results", output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Replace with your image path
    image_path = "an.png"
    results = process_image(image_path)
    
    # Print final sequence of directions from left to right
    print("\nFinal sequence of actions from left to right:")
    print(" → ".join(results))