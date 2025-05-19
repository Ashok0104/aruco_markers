from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
import uvicorn
from fastapi.responses import JSONResponse
import socket
import paho.mqtt.client as mqtt
from PIL import Image
from pillow_heif import register_heif_opener
import io
import logging
from datetime import datetime
import os
import json

app = FastAPI()

# Set up logging configuration
log_directory = "aruco_logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Create separate loggers for different types of logs
def setup_logger(name, log_file):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Create file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(fh)
    return logger

# Initialize different loggers
detection_logger = setup_logger('detection', f'{log_directory}/detection.log')
mqtt_logger = setup_logger('mqtt', f'{log_directory}/mqtt.log')
error_logger = setup_logger('error', f'{log_directory}/error.log')
operation_logger = setup_logger('operation', f'{log_directory}/operation.log')

# Register HEIF format to Pillow
register_heif_opener()

# MQTT Configuration
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "aruco@123"

# Initialize MQTT client
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception as e:
        error_logger.error(f"IP address retrieval failed: {str(e)}")
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def save_detection_result(result_data):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{log_directory}/detection_result_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(result_data, f, indent=4)
        operation_logger.info(f"Detection result saved to {filename}")
    except Exception as e:
        error_logger.error(f"Failed to save detection result: {str(e)}")

def convert_to_png(image_bytes: bytes) -> bytes:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        output = io.BytesIO()
        image.save(output, format="PNG")
        operation_logger.info("Image successfully converted to PNG")
        return output.getvalue()
    except Exception as e:
        error_logger.error(f"Image conversion failed: {str(e)}")
        return None

def enhance_image(image):
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        denoised = cv2.fastNlMeansDenoising(thresh)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        operation_logger.info("Image enhancement completed successfully")
        return sharpened
    except Exception as e:
        error_logger.error(f"Image enhancement failed: {str(e)}")
        return None

def detect_markers_with_multiple_attempts(image):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    parameters = cv2.aruco.DetectorParameters()
    
    parameters.adaptiveThreshConstant = 7
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 10
    parameters.minMarkerPerimeterRate = 0.03
    parameters.maxMarkerPerimeterRate = 0.5
    parameters.polygonalApproxAccuracyRate = 0.05
    parameters.minCornerDistanceRate = 0.05
    parameters.minDistanceToBorder = 3
    
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    attempts = [
        ("original", cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)),
        ("enhanced", enhance_image(image)),
        ("blurred", cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)),
        ("sharpened", cv2.filter2D(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), -1, 
                                 np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])))
    ]
    
    best_result = None
    max_markers = 0
    
    for method, processed_image in attempts:
        corners, ids, _ = detector.detectMarkers(processed_image)
        
        if ids is not None:
            detection_logger.info(f"Method {method}: detected {len(ids)} markers")
            if len(ids) > max_markers:
                max_markers = len(ids)
                best_result = (corners, ids)
                
                if max_markers >= getExpectedMarkerCount(processed_image):
                    break
        else:
            detection_logger.info(f"Method {method}: no markers detected")
    
    return best_result if best_result else ([], None)

def getExpectedMarkerCount(image):
    image_area = image.shape[0] * image.shape[1]
    typical_marker_area = 50 * 50
    expected_count = min(20, int(image_area / (typical_marker_area * 4)))
    operation_logger.info(f"Expected marker count: {expected_count}")
    return expected_count

def is_same_row(y1: float, y2: float, threshold: float = 20.0) -> bool:
    return abs(y1 - y2) <= threshold

def group_markers_by_rows(markers: list, threshold: float = 20.0) -> list:
    if not markers:
        return []
    
    sorted_markers = sorted(markers, key=lambda m: m["position"]["y"])
    rows = []
    current_row = [sorted_markers[0]]
    
    for marker in sorted_markers[1:]:
        if is_same_row(marker["position"]["y"], current_row[0]["position"]["y"], threshold):
            current_row.append(marker)
        else:
            current_row.sort(key=lambda m: m["position"]["x"])
            rows.append(current_row)
            current_row = [marker]
    
    if current_row:
        current_row.sort(key=lambda m: m["position"]["x"])
        rows.append(current_row)
    
    operation_logger.info(f"Grouped markers into {len(rows)} rows")
    return rows

def get_marker_value(marker_id: int) -> dict:
    ascii_char = chr(marker_id) if 32 <= marker_id <= 126 else None
    special_chars = ['B', 'F', 'R', 'L']
    
    if ascii_char in special_chars:
        detection_logger.info(f"Marker {marker_id} interpreted as ASCII '{ascii_char}'")
        return {
            "id": marker_id,
            "display_value": ascii_char,
            "type": "ascii"
        }
    else:
        detection_logger.info(f"Marker {marker_id} interpreted as ID")
        return {
            "id": marker_id,
            "display_value": str(marker_id),
            "type": "id"
        }

def process_marker_sequence(sorted_markers):
    processed_markers = []
    i = 0
    
    while i < len(sorted_markers):
        current_marker = sorted_markers[i]
        current_value = current_marker["display_value"]
        
        # Check if current marker is ASCII and next marker is numeric
        if (current_marker["type"] == "ascii" and 
            i + 1 < len(sorted_markers) and
            sorted_markers[i+1]["type"] == "id" and
            sorted_markers[i+1]["display_value"].isdigit()):
            
            # Get the multiplier from the next marker
            multiplier = int(sorted_markers[i+1]["display_value"])
            
            # Create a new marker with the ASCII repeated according to multiplier
            new_marker = current_marker.copy()
            new_marker["display_value"] = current_value * multiplier
            new_marker["processed"] = True
            processed_markers.append(new_marker)
            
            detection_logger.info(f"Repeated ASCII '{current_value}' {multiplier} times to get '{new_marker['display_value']}'")
            
            # Skip the next marker (the multiplier)
            i += 2
        else:
            # Keep the original marker
            processed_markers.append(current_marker)
            i += 1
    
    return processed_markers

@app.post("/detect_markers/")
async def detect_markers(file: UploadFile = File(...)):
    operation_logger.info(f"Starting marker detection for file: {file.filename}")
    
    image_bytes = await file.read()
    
    png_bytes = convert_to_png(image_bytes)
    if png_bytes is None:
        error_logger.error("Invalid image file")
        return JSONResponse(status_code=400, content={"error": "Invalid image file."})
    
    image_array = np.frombuffer(png_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if image is None:
        error_logger.error("Image decoding failed")
        return JSONResponse(status_code=400, content={"error": "Image decoding failed."})

    corners, ids = detect_markers_with_multiple_attempts(image)

    if ids is None:
        detection_logger.warning("No ArUco markers detected")
        return JSONResponse(content={"message": "No ArUco markers detected."})

    detected_markers = []
    
    cv2.aruco.drawDetectedMarkers(image, corners, borderColor=(255, 255, 255))

    for marker_id, corner in zip(ids, corners):
        marker_id = int(marker_id[0])
        x = int(np.mean(corner[0][:, 0]))
        y = int(np.mean(corner[0][:, 1]))
        
        marker_info = get_marker_value(marker_id)
        
        detected_markers.append({
            "id": marker_id,
            "display_value": marker_info["display_value"],
            "type": marker_info["type"],
            "position": {"x": x, "y": y}
        })

    sorted_rows = group_markers_by_rows(detected_markers)
    
    # Process each row to handle ASCII + number sequences
    processed_rows = []
    for row in sorted_rows:
        processed_row = process_marker_sequence(row)
        processed_rows.append(processed_row)
    
    # Flatten the processed rows
    processed_markers = [marker for row in processed_rows for marker in row]
    display_values = [marker["display_value"] for marker in processed_markers]

    # Prepare result data
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "markers": processed_markers,
        "rows": [[marker["display_value"] for marker in row] for row in processed_rows]
    }

    # Save detection result
    save_detection_result(result_data)

    if display_values:
        mqtt_message = {
            "timestamp": datetime.now().isoformat(),
            "values": ''.join(display_values),
            "source": "camera or gallery"
        }
        result = mqtt_client.publish(MQTT_TOPIC, str(mqtt_message))
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            mqtt_logger.info(f"MQTT message sent successfully: {mqtt_message}")
        else:
            mqtt_logger.error("MQTT message sending failed")

    detection_logger.info(f"Detection completed. Found {len(detected_markers)} markers, processed to {len(processed_markers)} outputs")
    return JSONResponse(content=result_data)

if __name__ == "__main__":
    local_ip = get_local_ip()
    operation_logger.info(f"Server starting on: http://{local_ip}:8000")
    mqtt_client.loop_start()
    uvicorn.run(app, host=local_ip, port=8000)