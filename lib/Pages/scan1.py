import cv2
import numpy as np
from cv2 import aruco

def process_marker_sequence(marker_ids):
    processed_output = []
    i = 0
    while i < len(marker_ids):
        if 32 <= marker_ids[i] <= 126:  # Check if it's an ASCII printable character
            char = chr(marker_ids[i])
            if i + 1 < len(marker_ids) and 0 <= marker_ids[i + 1] <= 9:  # Check if next is a number (0-9)
                count = marker_ids[i + 1]
                processed_output.append(char * count)  # Repeat character count times
                i += 2  # Move to the next set of markers
            else:
                processed_output.append(char)  # Just add the character if no number follows
                i += 1
        else:
            i += 1  # Skip invalid markers
    
    return "\n".join(processed_output)


def detect_aruco_markers(image_path, aruco_dict_type=aruco.DICT_6X6_250):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, _ = detector.detectMarkers(gray)
    
    if ids is not None:
        ids = ids.flatten().tolist()
        output = process_marker_sequence(ids)
        print("Processed Output:")
        print(output)
    else:
        print("No ArUco markers detected.")

# Example usage
detect_aruco_markers("aan.png")
