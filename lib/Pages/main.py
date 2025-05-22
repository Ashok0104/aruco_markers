import logging
import asyncio
import os
import uuid
import json
import cv2
import numpy as np
from PIL import Image
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bleak import BleakClient
from datetime import datetime

# Set up logging configuration
log_directory = "aruco_logs"
if not os.path.exists(log_directory):
    os.makedirs(log_directory)

# Configure detection logger
detection_logger = logging.getLogger('detection')
detection_logger.setLevel(logging.INFO)
detection_fh = logging.FileHandler(f'{log_directory}/detection.log')
detection_fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
detection_logger.addHandler(detection_fh)
detection_logger.addHandler(logging.StreamHandler())

# Configure Bluetooth logger
bluetooth_logger = logging.getLogger('bluetooth')
bluetooth_logger.setLevel(logging.INFO)
bluetooth_fh = logging.FileHandler(f'{log_directory}/bluetooth.log')
bluetooth_fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
bluetooth_logger.addHandler(bluetooth_fh)

# Define allowed ArUco dictionary types
ARUCO_DICTIONARIES = {
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250
}

# Direction marker IDs mapped to visual representation
DIRECTIONS = {
    70: "F",   # Yellow with up arrow
    82: "R",   # Blue with right arrow  
    76: "L",   # Red with left arrow
    66: "B"    # Green with down arrow
}

# Multiplier marker IDs
MULTIPLIERS = {
    2: 2,
    3: 3,
    4: 4,
    5: 5
}

# Bluetooth configuration
ADDRESS = "3C:84:27:C2:A0:AD"
UART_TX_CHAR_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"  # TX UUID (sending)
UART_RX_CHAR_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"  # RX UUID (receiving)

# Initialize FastAPI app
app = FastAPI(title="ArUco Marker Detection and Bluetooth API")

# Add CORS middleware with updated configuration to expose X-Metadata header
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://192.168.4.51:5500"],  # Allow your web app's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
    expose_headers=["X-Metadata"],  # Expose the X-Metadata header to the client
)

# In-memory storage for sequence history
sequence_history = []

def standardize_image(image_data: bytes) -> np.ndarray:
    """Load and standardize image from bytes, converting to PNG"""
    try:
        # Use PIL to open and convert any image format to PNG
        pil_image = Image.open(io.BytesIO(image_data))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to PNG format
        png_buffer = io.BytesIO()
        pil_image.save(png_buffer, format="PNG")
        png_data = png_buffer.getvalue()
        
        # Convert PNG data to numpy array for OpenCV
        nparr = np.frombuffer(png_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detection_logger.info("Image successfully standardized to PNG")
        return image
    except Exception as e:
        detection_logger.error(f"Image standardization failed: {str(e)}")
        return None

def enhance_image(image):
    """Enhance image for better marker detection"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        denoised = cv2.fastNlMeansDenoising(thresh)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(denoised, -1, kernel)
        detection_logger.info("Image enhancement completed successfully")
        return sharpened
    except Exception as e:
        detection_logger.error(f"Image enhancement failed: {str(e)}")
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
            ("original", cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)),
            ("enhanced", enhance_image(image)),
            ("blurred", cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), (5, 5), 0)),
            ("sharpened", cv2.filter2D(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), -1, 
                                     np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))),
            ("equalized", cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY))),
            ("clahe", cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(
                        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)))
        ]
        
        best_result = None
        max_markers = 0
        best_method = None
        
        for method, processed_image in attempts:
            if processed_image is None:
                continue
                
            corners, ids, _ = detector.detectMarkers(processed_image)
            
            if ids is not None:
                detection_logger.info(f"Dict {dict_type}, Method {method}: detected {len(ids)} markers")
                if len(ids) > max_markers:
                    max_markers = len(ids)
                    best_result = (corners, ids, dict_type)
                    best_method = method
            else:
                detection_logger.info(f"Dict {dict_type}, Method {method}: no markers detected")
        
        if best_method:
            detection_logger.info(f"Best method for dict {dict_type} was: {best_method}")
            
        return best_result, max_markers
    except Exception as e:
        detection_logger.error(f"Error detecting markers with dictionary {dict_type}: {str(e)}")
        return None, 0

def detect_markers_with_multiple_dictionaries(image):
    """Try multiple ArUco dictionaries to find the best match"""
    dict_types = ["DICT_6X6_100", "DICT_6X6_250"]
    
    best_result = None
    max_markers = 0
    best_dict = None
    
    for dict_type in dict_types:
        result, num_markers = detect_markers_with_dictionary(image, dict_type)
        if result is not None and num_markers > max_markers:
            max_markers = num_markers
            best_result = result
            best_dict = dict_type
    
    if best_result:
        detection_logger.info(f"Best dictionary was {best_dict} with {max_markers} markers")
        return best_result
    else:
        detection_logger.warning("No markers detected with allowed dictionaries")
        return ([], None, None)

def classify_marker(marker_id: int) -> dict:
    """Classify marker by type and get display value"""
    if marker_id in DIRECTIONS:
        return {
            "id": marker_id,
            "display_value": DIRECTIONS[marker_id],
            "type": "direction"
        }
    elif marker_id in MULTIPLIERS:
        return {
            "id": marker_id,
            "display_value": str(MULTIPLIERS[marker_id]),
            "type": "multiplier"
        }
    else:
        detection_logger.info(f"Marker {marker_id} interpreted as ID")
        return {
            "id": marker_id,
            "display_value": str(marker_id),
            "type": "id"
        }

def find_multiplier_below(direction_marker, all_markers, threshold=200):
    """Find a multiplier marker below this direction marker"""
    dir_x = direction_marker["position"]["x"]
    dir_y = direction_marker["position"]["y"]
    
    multipliers = [m for m in all_markers if m["type"] == "multiplier"]
    if not multipliers:
        return 1
    
    for marker in multipliers:
        m_x = marker["position"]["x"]
        m_y = marker["position"]["y"]
        
        x_diff = abs(dir_x - m_x)
        y_diff = m_y - dir_y
        
        if 0 < y_diff < threshold and x_diff < 100:
            detection_logger.info(f"Found multiplier {marker['display_value']} below direction {direction_marker['display_value']} " +
                       f"at ({dir_x}, {dir_y}), multiplier at ({m_x}, {m_y})")
            return int(marker["display_value"])
    
    detection_logger.info(f"No multiplier found below direction {direction_marker['display_value']} at ({dir_x}, {dir_y})")
    return 1

def group_markers_by_rows(markers, row_threshold=50):
    """Group markers into rows based on y-coordinate proximity"""
    if not markers:
        return []
    
    sorted_by_y = sorted(markers, key=lambda m: m["position"]["y"])
    
    rows = []
    current_row = [sorted_by_y[0]]
    current_row_y = sorted_by_y[0]["position"]["y"]
    
    for marker in sorted_by_y[1:]:
        marker_y = marker["position"]["y"]
        
        if abs(marker_y - current_row_y) <= row_threshold:
            current_row.append(marker)
        else:
            rows.append(sorted(current_row, key=lambda m: m["position"]["x"]))
            current_row = [marker]
            current_row_y = marker_y
    
    if current_row:
        rows.append(sorted(current_row, key=lambda m: m["position"]["x"]))
    
    detection_logger.info(f"Grouped markers into {len(rows)} rows")
    
    for i, row in enumerate(rows):
        detection_logger.info(f"Row {i+1} contains {len(row)} markers with y-values: {[m['position']['y'] for m in row]}")
    
    return rows

def process_markers_for_output(direction_markers, all_markers):
    """Process all markers to generate the output sequence and return log message"""
    rows = group_markers_by_rows(direction_markers, row_threshold=50)
    
    rows.sort(key=lambda row: sum(m["position"]["y"] for m in row) / len(row))
    
    final_sequence = ""
    
    for row_idx, row in enumerate(rows):
        detection_logger.info(f"Processing row {row_idx + 1} with {len(row)} markers")
        
        for marker in row:
            direction = marker["display_value"]
            multiplier = find_multiplier_below(marker, all_markers, threshold=200)
            final_sequence += direction * multiplier
            detection_logger.info(f"Added {direction} x {multiplier} to sequence")
    
    log_message = f"Final sequence generated: {final_sequence}"
    detection_logger.info(log_message)
    return final_sequence, log_message

def annotate_image(image, markers, output_sequence):
    """Annotate image with marker information and final sequence"""
    annotated_image = image.copy()
    
    for marker in markers:
        x, y = marker["position"]["x"], marker["position"]["y"]
        marker_type = marker["type"]
        display_value = marker["display_value"]
        
        if marker_type == "direction":
            color = (0, 0, 255)
        elif marker_type == "multiplier":
            color = (255, 255, 0)
        else:
            color = (0, 255, 0)
        
        cv2.circle(annotated_image, (x, y), 20, color, 2)
        
        text = f"{display_value} ({x},{y})"
        cv2.putText(
            annotated_image,
            text,
            (x + 25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )
    
    cv2.putText(
        annotated_image,
        f"Sequence: {output_sequence}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )
    
    return annotated_image

def image_to_bytes(image: np.ndarray) -> bytes:
    """Convert image to PNG bytes for response"""
    _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return buffer.tobytes()

async def send_bluetooth_message(message: str) -> bool:
    """Send the complete sequence message via Bluetooth with proper chunking and termination"""
    try:
        async with BleakClient(ADDRESS) as client:
            if not client.is_connected:
                bluetooth_logger.error("Failed to connect to Bluetooth device")
                return False

            bluetooth_logger.info(f"Connected to {ADDRESS}")
            
            try:
                # Add newline terminator to indicate end of message
                full_message = (message + '\n').encode('utf-8')
                bluetooth_logger.info(f"Encoded message: {full_message!r} (length: {len(full_message)} bytes)")
                
                # Check message length and split if necessary
                MAX_CHUNK_SIZE = 20  # Common BLE characteristic limit
                
                if len(full_message) <= MAX_CHUNK_SIZE:
                    # Send as single message
                    await client.write_gatt_char(UART_RX_CHAR_UUID, full_message, response=False)
                    bluetooth_logger.info(f"Sent complete sequence as single message: '{message}' (length: {len(message)} chars, {len(full_message)} bytes)")
                else:
                    # Send in chunks
                    bluetooth_logger.info(f"Message too long ({len(full_message)} bytes), sending in {len(full_message) // MAX_CHUNK_SIZE + 1} chunks")
                    
                    for i in range(0, len(full_message), MAX_CHUNK_SIZE):
                        chunk = full_message[i:i + MAX_CHUNK_SIZE]
                        bluetooth_logger.info(f"Sending chunk {i // MAX_CHUNK_SIZE + 1}: {chunk!r} (length: {len(chunk)} bytes)")
                        await client.write_gatt_char(UART_RX_CHAR_UUID, chunk, response=False)
                        # Increased delay between chunks
                        await asyncio.sleep(0.1)
                
                # Increased final delay to ensure transmission completes
                await asyncio.sleep(0.5)
                
                bluetooth_logger.info(f"Successfully sent complete sequence: '{message}' (length: {len(message)} chars)")
                return True
                
            except Exception as e:
                bluetooth_logger.error(f"Error sending sequence '{message}': {str(e)}")
                return False
                
    except asyncio.CancelledError:
        bluetooth_logger.error("Bluetooth connection attempt was cancelled")
        return False
    except Exception as e:
        bluetooth_logger.error(f"Bluetooth communication error: {str(e)}")
        return False
    finally:
        bluetooth_logger.info("Bluetooth communication finished")

class DirectBluetoothRequest(BaseModel):
    message: str = None  # Make message optional since it won't be used

@app.post("/detect-markers")
async def detect_markers(file: UploadFile = File(...), send_bluetooth: str = "false"):
    """Endpoint to detect ArUco markers in an uploaded image and optionally send sequence via Bluetooth"""
    try:
        # Convert send_bluetooth query param to boolean
        send_bluetooth_bool = send_bluetooth.lower() == "true"
        
        # Read image data
        image_data = await file.read()
        detection_logger.info(f"Received image: {file.filename}, size: {len(image_data)} bytes")

        # Step 1: Standardize image to PNG
        image = standardize_image(image_data)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to process image")

        # Step 2: Detect ArUco markers
        corners, ids, used_dict = detect_markers_with_multiple_dictionaries(image)
        
        if ids is None or len(ids) == 0:
            detection_logger.warning("No ArUco markers detected")
            raise HTTPException(status_code=404, detail="No ArUco markers detected")

        # Step 3: Process detected markers
        all_markers = []
        for i, (marker_id, corner) in enumerate(zip(ids, corners)):
            marker_id = int(marker_id[0])
            x = int(np.mean(corner[0][:, 0]))
            y = int(np.mean(corner[0][:, 1]))
            
            marker_info = classify_marker(marker_id)
            marker_info["position"] = {"x": x, "y": y}
            marker_info["corner"] = corner
            all_markers.append(marker_info)
            detection_logger.info(f"Detected marker: ID={marker_id}, Type={marker_info['type']}, Position=({x},{y})")

        # Step 4: Filter out direction markers
        direction_markers = [m for m in all_markers if m["type"] == "direction"]

        # Step 5: Process markers to generate the output sequence
        final_sequence, sequence_log = process_markers_for_output(direction_markers, all_markers)

        # Step 6: Store the sequence
        sequence_history.append({
            "sequence": final_sequence,
            "timestamp": datetime.now().isoformat(),
            "source": "detect-markers"
        })
        detection_logger.info(f"Stored sequence: {final_sequence} from detect-markers")

        # Step 7: Optionally send the final_sequence via Bluetooth
        bluetooth_status = None
        if send_bluetooth_bool:
            bluetooth_logger.info(f"Preparing to send final sequence via Bluetooth: '{final_sequence}' (length: {len(final_sequence)} chars)")
            success = await send_bluetooth_message(final_sequence)
            bluetooth_status = f"Bluetooth transmission of final sequence '{final_sequence}' successful" if success else f"Bluetooth transmission of final sequence '{final_sequence}' failed"

        # Step 8: Annotate the image
        annotated_image = annotate_image(image, all_markers, final_sequence)

        # Convert annotated image to PNG bytes
        image_bytes = image_to_bytes(annotated_image)

        # Prepare metadata to include in headers
        metadata = {
            "sequence": final_sequence,
            "sequence_log": sequence_log,
            "markers_detected": len(all_markers),
            "bluetooth_status": bluetooth_status if send_bluetooth_bool else "Bluetooth not triggered"
        }

        # Return StreamingResponse with metadata in headers
        return StreamingResponse(
            io.BytesIO(image_bytes),
            media_type="image/png",
            headers={
                "Content-Disposition": f"inline; filename=annotated_{uuid.uuid4()}.png",
                "X-Metadata": json.dumps(metadata)
            }
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        detection_logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/send_bluetooth_direct/")
async def send_bluetooth_direct(request: DirectBluetoothRequest):
    """Endpoint for sending the latest stored sequence via Bluetooth"""
    # Fetch the latest sequence from sequence_history
    if not sequence_history:
        bluetooth_logger.error("No sequences available in history to send")
        return JSONResponse(
            status_code=400,
            content={"error": "No sequences available in history to send"}
        )
    
    final_sequence = sequence_history[-1]["sequence"]
    
    # Use exact same logging and approach as the working detect-markers endpoint
    bluetooth_logger.info(f"Preparing to send stored sequence via Bluetooth: '{final_sequence}' (length: {len(final_sequence)} chars)")
    success = await send_bluetooth_message(final_sequence)
    
    # Store the sequence if transmission is successful (to maintain history)
    if success:
        sequence_history.append({
            "sequence": final_sequence,
            "timestamp": datetime.now().isoformat(),
            "source": "send_bluetooth_direct"
        })
        bluetooth_logger.info(f"Stored sequence: {final_sequence} from send_bluetooth_direct")
    
    # Use exact same status message format as detect-markers endpoint
    bluetooth_status = f"Bluetooth transmission of final sequence '{final_sequence}' successful" if success else f"Bluetooth transmission of final sequence '{final_sequence}' failed"
    
    bluetooth_logger.info(bluetooth_status)
    
    if success:
        return JSONResponse(
            content={
                "message": "Bluetooth transmission successful",
                "data_sent": final_sequence,
                "bluetooth_status": bluetooth_status
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": "Bluetooth transmission failed", 
                "bluetooth_status": bluetooth_status
            }
        )

@app.get("/get_sequence_history/")
async def get_sequence_history():
    """Endpoint to fetch the history of stored sequences"""
    return JSONResponse(
        content={
            "message": "Sequence history retrieved successfully",
            "sequences": sequence_history
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)