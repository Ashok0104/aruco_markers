from fastapi import FastAPI, File, UploadFile, Query, Body
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
from typing import List, Optional
import asyncio
from bleak import BleakClient
from pydantic import BaseModel

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
bluetooth_logger = setup_logger('bluetooth', f'{log_directory}/bluetooth.log')
# Add new logger for result IDs
result_id_logger = setup_logger('result_id', f'{log_directory}/result_ids.log')

# Register HEIF format to Pillow
register_heif_opener()

# MQTT Configuration
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
MQTT_TOPIC = "aruco@123"

# Initialize MQTT client
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Define all ArUco dictionary types available in OpenCV
ARUCO_DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# Bluetooth configuration
ADDRESS = "3C:84:27:C2:A0:AD"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # TX UUID (sending)
UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # RX UUID (receiving)

# In-memory store for recent detection results
detection_results_cache = {}

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
    result_id = timestamp
    
    try:
        with open(filename, 'w') as f:
            json.dump(result_data, f, indent=4)
        operation_logger.info(f"Detection result saved to {filename}")
        
        # Store in memory cache with a unique ID
        detection_results_cache[result_id] = result_data
        
        # Log the result ID with additional information
        result_id_logger.info(
            json.dumps({
                "result_id": result_id,
                "timestamp": datetime.now().isoformat(),
                "filename": filename,
                "marker_count": len(result_data.get("markers", [])),
                "sequential_output": result_data.get("sequential_output", ""),
                "dictionary_used": result_data.get("dictionary_used", "")
            })
        )
        
        return result_id
    except Exception as e:
        error_logger.error(f"Failed to save detection result: {str(e)}")
        return None

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

def detect_markers_with_dictionary(image, dict_type):
    """Detect markers using a specific ArUco dictionary"""
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARIES[dict_type])
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
            if processed_image is None:
                continue
                
            corners, ids, _ = detector.detectMarkers(processed_image)
            
            if ids is not None:
                detection_logger.info(f"Dict {dict_type}, Method {method}: detected {len(ids)} markers")
                if len(ids) > max_markers:
                    max_markers = len(ids)
                    best_result = (corners, ids, dict_type)
            else:
                detection_logger.info(f"Dict {dict_type}, Method {method}: no markers detected")
        
        return best_result, max_markers
    except Exception as e:
        error_logger.error(f"Error detecting markers with dictionary {dict_type}: {str(e)}")
        return None, 0

def detect_markers_with_multiple_dictionaries(image, dict_types=None):
    """Try multiple ArUco dictionaries to find the best match"""
    if dict_types is None:
        dict_types = list(ARUCO_DICTIONARIES.keys())
    
    best_result = None
    max_markers = 0
    best_dict = None
    
    for dict_type in dict_types:
        result, num_markers = detect_markers_with_dictionary(image, dict_type)
        if result is not None and num_markers > max_markers:
            max_markers = num_markers
            best_result = result
            best_dict = dict_type
            
            # If we found a significant number of markers, we can stop early
            if max_markers >= getExpectedMarkerCount(image):
                break
    
    if best_result:
        detection_logger.info(f"Best dictionary was {best_dict} with {max_markers} markers")
        return best_result
    else:
        detection_logger.warning("No markers detected with any dictionary")
        return ([], None, None)

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
    ascii_markers = []
    id_markers = []
    
    # First, separate ASCII and ID markers
    for marker in sorted_markers:
        if marker["type"] == "ascii":
            ascii_markers.append(marker)
        else:  # type == "id"
            id_markers.append(marker)
    
    # If we have ID markers, create alternating pattern
    if id_markers:
        # Process pairs of ASCII and ID markers
        max_pairs = min(len(ascii_markers), len(id_markers))
        
        for i in range(max_pairs):
            # Add ASCII marker
            processed_markers.append(ascii_markers[i])
            # Add ID marker
            processed_markers.append(id_markers[i])
        
        # Add any remaining ASCII markers
        processed_markers.extend(ascii_markers[max_pairs:])
        # Add any remaining ID markers
        processed_markers.extend(id_markers[max_pairs:])
    else:
        # If no ID markers, just process ASCII markers normally
        for current_marker in sorted_markers:
            processed_markers.append(current_marker)
    
    detection_logger.info(f"Processed {len(processed_markers)} markers in sequence")
    return processed_markers

def expand_fbrl_sequence(sequence):
    expanded_output = ""
    i = 0
    while i < len(sequence):
        char = sequence[i]
        if (char in "FBRL" and i + 1 < len(sequence) and sequence[i + 1].isdigit()):
            repeat_count = int(sequence[i + 1])
            expanded_output += char * repeat_count
            i += 2  # Skip the number as well
        else:
            expanded_output += char
            i += 1
    return expanded_output

def process_sequential_output(markers):
    if not markers:
        return ""
    
    has_id_markers = any(marker["type"] == "id" for marker in markers)
    
    if has_id_markers:
        output = ""
        ascii_values = [m["display_value"] for m in markers if m["type"] == "ascii"]
        id_values = [m["display_value"] for m in markers if m["type"] == "id"]
        
        for ascii_val, id_val in zip(ascii_values, id_values):
            output += str(ascii_val) + str(id_val)
        
        if len(ascii_values) > len(id_values):
            output += ''.join(ascii_values[len(id_values):])
        elif len(id_values) > len(ascii_values):
            output += ''.join(id_values[len(ascii_values):])
    else:
        output = ''.join(str(marker["display_value"]) for marker in markers)
    
    return expand_fbrl_sequence(output)

async def send_message(client, message):
    try:
        await client.write_gatt_char(UART_RX_CHAR_UUID, message.encode(), response=True)
        bluetooth_logger.info(f"Sent: {message} - Success")
        return True
    except Exception as e:
        error_logger.error(f"Error sending message: {e}")
        return False

async def send_bluetooth_message(message):
    try:
        async with BleakClient(ADDRESS) as client:
            if not client.is_connected:
                error_logger.error("Failed to connect to Bluetooth device")
                return False

            bluetooth_logger.info(f"Connected to {ADDRESS}")
            result = await send_message(client, message)
            return result
    except Exception as e:
        error_logger.error(f"Bluetooth communication error: {e}")
        return False
    finally:
        bluetooth_logger.info("Bluetooth communication finished")

@app.post("/detect_markers/")
async def detect_markers(
    file: UploadFile = File(...),
    dict_types: Optional[List[str]] = Query(
        None,
        description="List of ArUco dictionary types to use. If not provided, all dictionaries will be tried.",
        example=["DICT_4X4_50", "DICT_6X6_250"]
    )
):
    operation_logger.info(f"Starting marker detection for file: {file.filename}")
    
    if dict_types:
        invalid_dicts = [d for d in dict_types if d not in ARUCO_DICTIONARIES]
        if invalid_dicts:
            error_logger.error(f"Invalid dictionary types: {invalid_dicts}")
            return JSONResponse(
                status_code=400, 
                content={"error": f"Invalid dictionary types: {invalid_dicts}", 
                         "valid_types": list(ARUCO_DICTIONARIES.keys())}
            )
    
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
    
    corners, ids, used_dict = detect_markers_with_multiple_dictionaries(image, dict_types)
    
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
    
    processed_rows = []
    for row in sorted_rows:
        processed_row = process_marker_sequence(row)
        processed_rows.append(processed_row)
    
    processed_markers = [marker for row in processed_rows for marker in row]
    
    final_output = process_sequential_output(processed_markers)
    operation_logger.info(f"Final Sequential Output: {final_output}")
    
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "dictionary_used": used_dict,
        "markers": processed_markers,
        "rows": [[marker["display_value"] for marker in row] for row in processed_rows],
        "sequential_output": final_output
    }
    
    # Save detection result and get the unique ID
    result_id = save_detection_result(result_data)
    
    if final_output:
        mqtt_message = {
            "timestamp": datetime.now().isoformat(),
            "values": final_output,
            "source": "camera or gallery",
            "dictionary": used_dict
        }
        result = mqtt_client.publish(MQTT_TOPIC, str(mqtt_message))
        if result.rc == mqtt.MQTT_ERR_SUCCESS:
            mqtt_logger.info(f"MQTT message sent successfully: {mqtt_message}")
        else:
            mqtt_logger.error("MQTT message sending failed")
        
        # Note: No longer sending Bluetooth message directly here
        # Instead, include the result_id in the response for Flutter to use
    
    detection_logger.info(f"Final Sequential Output: {final_output}")
    
    # Include the result_id in the response so Flutter can use it to trigger Bluetooth
    result_data["result_id"] = result_id
    return JSONResponse(content=result_data)

# Define the model for the Bluetooth trigger request
class BluetoothTriggerRequest(BaseModel):
    result_id: str

@app.post("/send_bluetooth/")
async def send_bluetooth(request: BluetoothTriggerRequest):
    """Endpoint for Flutter to trigger Bluetooth transmission of a previously detected result"""
    result_id = request.result_id
    
    # Check if the result exists in our cache
    if result_id not in detection_results_cache:
        error_logger.error(f"Result ID {result_id} not found in cache")
        return JSONResponse(
            status_code=404,
            content={"error": "Detection result not found. The result may have expired."}
        )
    
    # Get the result data
    result_data = detection_results_cache[result_id]
    final_output = result_data.get("sequential_output", "")
    
    if not final_output:
        error_logger.error(f"No output to send for result ID {result_id}")
        return JSONResponse(
            status_code=400,
            content={"error": "No output data available to send"}
        )
    
    # Send via Bluetooth
    bluetooth_logger.info(f"Sending via Bluetooth: {final_output} (triggered by Flutter)")
    success = await send_bluetooth_message(final_output)
    
    if success:
        return JSONResponse(
            content={
                "message": "Bluetooth transmission successful",
                "data_sent": final_output
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={"error": "Bluetooth transmission failed"}
        )

@app.get("/available_dictionaries/")
async def get_available_dictionaries():
    """Return a list of all available ArUco dictionary types"""
    return {"dictionaries": list(ARUCO_DICTIONARIES.keys())}

@app.get("/list_results/")
async def list_results():
    """Return a list of all stored result IDs with their details"""
    try:
        results = []
        log_file = f'{log_directory}/result_ids.log'
        
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        # Extract the JSON part from the log line
                        json_str = line.split(' - INFO - ')[1]
                        result_info = json.loads(json_str)
                        results.append(result_info)
                    except Exception as e:
                        error_logger.error(f"Error parsing log line: {str(e)}")
                        continue
        
        return {"results": results}
    except Exception as e:
        error_logger.error(f"Error listing results: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": "Failed to retrieve results"}
        )

if __name__ == "__main__":
    local_ip = get_local_ip()
    operation_logger.info(f"Server starting on: http://{local_ip}:8000")
    mqtt_client.loop_start()
    uvicorn.run(app, host=local_ip, port=8000)