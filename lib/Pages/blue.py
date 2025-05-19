import cv2
import numpy as np

def detect_aruco_markers(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not load image")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize the ArUco dictionary and detector parameters
    # Using 4x4 dictionary with 50 markers as an example
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()

    # Create detector
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect markers
    corners, ids, rejected = detector.detectMarkers(gray)

    # If markers are detected
    if ids is not None:
        print(f"Found {len(ids)} ArUco markers")
        
        # Draw detected markers on the image
        cv2.aruco.drawDetectedMarkers(image, corners, ids)

        # Process each detected marker
        for i in range(len(ids)):
            # Get the center of the marker
            c = corners[i][0]
            center_x = int(c[:, 0].mean())
            center_y = int(c[:, 1].mean())
            
            # Put the ID text near the marker
            cv2.putText(image, 
                       str(ids[i][0]),
                       (center_x, center_y),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       1,
                       (0, 255, 0),
                       2,
                       cv2.LINE_AA)

            # Print marker details
            print(f"Marker ID: {ids[i][0]}")
            print(f"Corners: {c}")
            print("---")

    else:
        print("No ArUco markers detected")

    # Display the result
    cv2.imshow("ArUco Markers", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Optionally save the output
    output_path = "output_aruco_image.jpg"
    cv2.imwrite(output_path, image)
    print(f"Output saved as {output_path}")

def main():
    # Specify your image path here
    image_path = "path_to_your_image.jpg"  # Replace with your image path
    
    # You can test with different ArUco dictionary types if needed
    # Available dictionaries:
    # DICT_4X4_50, DICT_4X4_100, DICT_4X4_250, DICT_4X4_1000
    # DICT_5X5_50, DICT_5X5_100, DICT_5X5_250, DICT_5X5_1000
    # DICT_6X6_50, DICT_6X6_100, DICT_6X6_250, DICT_6X6_1000
    # DICT_7X7_50, DICT_7X7_100, DICT_7X7_250, DICT_7X7_1000
    
    detect_aruco_markers(image_path)

if __name__ == "__main__":
    main()