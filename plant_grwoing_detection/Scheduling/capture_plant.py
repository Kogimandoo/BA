# Open the camera → discard a few frames for stabilization → capture 100 frames 
# → generate one depth map using the median → save as .npy

import time
from datetime import datetime
from pathlib import Path
import numpy as np
import ArducamDepthCamera as ac

DATA_DIR = Path("/home/pi4/farm/Arducam_tof_camera/plant_growth/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

MAX_DISTANCE = 2000

WARMUP_FRAMES = 20
CAPTURE_FRAMES = 100
FRAME_TIMEOUT_MS = 2000 # Treat it as a failure if a frame is not received within 2 seconds


# Function that converts the current time into a string 
# -> this timestamp is used in the saved file name
def get_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def open_camera():
    cam = ac.ArducamCamera() # Create an Arducam camera object

    ret = cam.open(ac.Connection.CSI, 0) # Open the camera using the CSI connection
    if ret != 0:
        raise RuntimeError(f"Failed to open camera. Error code: {ret}")

    ret = cam.start(ac.FrameType.DEPTH) # Start receiving depth frames from the camera
    if ret != 0:
        cam.close()
        raise RuntimeError(f"Failed to start camera. Error code: {ret}")

    cam.setControl(ac.Control.RANGE, MAX_DISTANCE)

    return cam


# Extracts depth data from the frame object received from the camera
def get_frame_data(frame):
    depth = np.array(frame.depth_data, dtype=np.float32) # Convert the depth data in the frame into a NumPy array
    return depth


# Requests one frame from the camera, extracts the depth data, and then releases the frame
def request_depth_frame(cam):
    frame = cam.requestFrame(FRAME_TIMEOUT_MS) # Request one depth frame from the camera

    if frame is None: # If no frame is received -> return None because there is no data
        return None

    try:
        return get_frame_data(frame) # Extract the depth data from the frame
    finally:
        cam.releaseFrame(frame) # Always release the frame back to the camera after use


# Captures multiple depth frames and creates one stable depth map
def capture_depth(cam) -> np.ndarray:
    for _ in range(WARMUP_FRAMES): # Discard the first 20 frames without saving them -> camera stabilization
        request_depth_frame(cam)
        time.sleep(0.03)

    frames = [] # Stores the depth frames used to calculate the median

    # Loop for capturing the depth frames that will actually be used -> 100 iterations (100 frames)
    for _ in range(CAPTURE_FRAMES): 
        depth = request_depth_frame(cam) # Request one frame and retrieve its depth data

        if depth is None:
            continue

        frames.append(depth.copy())
        time.sleep(0.03)


    # If no frames were captured, raise an error.
    # If the frames list is empty, the median cannot be calculated, so stop here.
    if len(frames) == 0: 
        raise RuntimeError("No valid depth frames captured.")

    # Create the final depth map
    # np.stack(frames, axis=0) -> stack multiple depth frames into one array
    # For example, if one depth frame has a shape of (180, 240) and 100 frames were captured,
    # the stacked result will have a shape of (100, 180, 240)
    # np.median(..., axis=0) -> calculate the median of the 100 values at each pixel position
    return np.median(np.stack(frames, axis=0), axis=0).astype(np.float32)


def main():
    timestamp = get_timestamp()
    output_path = DATA_DIR / f"{timestamp}_depth.npy"

    # Initialize the camera variable as None so it can be safely closed later in the finally block
    cam = None 

    try:
        print(f"Starting plant capture: {timestamp}")

        cam = open_camera()
        depth = capture_depth(cam) # Create the depth map

        np.save(output_path, depth)
        print(f"Saved depth file")

    finally:
        if cam is not None:
            cam.stop()
            cam.close()

        print("Camera closed.")


if __name__ == "__main__":
    main()