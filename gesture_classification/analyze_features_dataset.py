import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg") # Save graphs as image files without opening a Matplotlib display window
import matplotlib.pyplot as plt
import mediapipe as mp


#INPUT_VIDEO_PATH = Path("/home/pi4/farm/Arducam_tof_camera/touch_videos/plant_touch_2026-06-25_14-21-20.mp4")
#INPUT_VIDEO_PATH = Path("/home/pi4/farm/Arducam_tof_camera/touch_videos/plant_touch_2026-06-25_14-27-00.mp4")
#INPUT_VIDEO_PATH = Path("/home/pi4/farm/Arducam_tof_camera/touch_videos/plant_touch_2026-06-25_14-25-37.mp4")
INPUT_VIDEO_PATH = Path("/home/pi4/farm/Arducam_tof_camera/touch_videos/plant_touch_2026-06-25_14-37-06.mp4")

OUTPUT_DIR = None


# 1. MediaPipe landmark indices 

WRIST = 0

THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4

INDEX_MCP = 5
INDEX_PIP = 6
INDEX_DIP = 7
INDEX_TIP = 8

MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_DIP = 11
MIDDLE_TIP = 12


# 2. Settings 

PROCESS_EVERY_N_FRAMES = 3

# Minimum number of valid frames required to calculate active_x / active_y standard deviation
ACTIVE_STD_MIN_VALID_FRAMES = 3

# Noise filter to prevent very small MediaPipe jitter from being counted as direction changes
DIRECTION_CHANGE_MIN_DELTA_NORM = 0.02

# Pinch detection threshold
PINCH_RATIO_THRESHOLD = 0.7



# 3. Utility functions

# Function that checks whether a value is a valid number rather than None or NaN
def is_finite(value) -> bool: 
    try:
        return value is not None and math.isfinite(float(value))
    except Exception:
        return False

# Function that returns NaN for empty values
def nan() -> float:
    return float("nan")

# Calculate Euclidean distance
def distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# Function that converts boolean values to integers for easier CSV storage
def bool_to_int(value: bool) -> int:
    return 1 if value else 0

# Calculate angle ABC in degrees -> calculate the angle at point b
# Given three points a-b-c, calculate the angle centered at point b.
def angle_2d(
    a: Tuple[float, float],
    b: Tuple[float, float],
    c: Tuple[float, float]
) -> float:

    if not ( 
        is_finite(a[0]) and is_finite(a[1]) and
        is_finite(b[0]) and is_finite(b[1]) and
        is_finite(c[0]) and is_finite(c[1])
    ):
        return nan() # Return NaN if any coordinate is NaN or invalid

    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=np.float32) # Vector from b to a
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=np.float32) # Vector from b to c

    norm_ba = np.linalg.norm(ba) # Calculate the vector length
    norm_bc = np.linalg.norm(bc)

    # If two points are too close and the vector length is nearly zero, the angle cannot be calculated
    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return nan()

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc) # cos(angle) = dot product of two vectors / product of their lengths
    cos_angle = np.clip(cos_angle, -1.0, 1.0) # acos() requires an input value between -1 and 1

    return math.degrees(math.acos(cos_angle)) # Convert the angle from radians to degrees and return it


def write_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    # w means write mode / with automatically closes the file when finished
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter( # csv.DictWriter is an object that writes dictionary data in CSV format
            f,
            fieldnames=fieldnames, # Specify the column names and order to save in the CSV
            extrasaction="ignore"  # Ignore dictionary entries not included in fieldnames -> only fieldnames are saved
        )

        writer.writeheader() # Write the column names on the first line of the CSV file -> taken from fieldnames

        for row in rows:
            cleaned = {} # Clean each row before saving it to CSV -> especially used to replace NaN values with empty fields

            for key in fieldnames: # Check only the fieldnames selected for CSV output
                value = row.get(key, "") # Get the value corresponding to the current key from the row

                if isinstance(value, float) and math.isnan(value): # Check whether value is a float and whether it is NaN (not a number)
                    cleaned[key] = "" # Replace NaN values with an empty string
                else:
                    cleaned[key] = value

            writer.writerow(cleaned) # Save one cleaned dictionary as a row in the CSV file


# 4. Feature row structure

# Function that creates an empty row dictionary for storing feature information for one frame -> used when no hand is detected
def make_empty_row(frame_number: int, time_sec: float) -> Dict:
    return {
        "frame": frame_number,
        "time_sec": time_sec,

        "mediapipe_processed": 0,
        "hand_detected": 0,

        "thumb_x": nan(),
        "thumb_y": nan(),
        "index_x": nan(),
        "index_y": nan(),
        "middle_x": nan(),
        "middle_y": nan(),
        "wrist_x": nan(),
        "wrist_y": nan(),

        "active_x": nan(),
        "active_y": nan(),

        # Active point standard deviation inside recent hand-detected window
        "active_std_reference_distance": nan(),
        "active_x_std_px": nan(),
        "active_y_std_px": nan(),
        "active_std_total_px": nan(),
        "active_x_std_norm": nan(),
        "active_y_std_norm": nan(),
        "active_std_total_norm": nan(),
        "active_std_valid_frames": 0, # Number of valid frames actually used to calculate active point standard deviation

        # Internal normalization reference used for ratios and direction change
        "reference_distance": nan(),

        # Finger distance ratios
        "thumb_index_ratio": nan(),
        "index_middle_ratio": nan(),
        "thumb_middle_ratio": nan(),

        # Raw pinch detected
        "pinch_detected": 0,

        # Counts frames during a pinch segment.
        # If hand detection temporarily disappears, this count is not reset.
        # It resets only when a hand is detected but pinch is not detected.
        "pinch_detected_frame_count": 0,

        "thumb_ip_angle": nan(),
        "index_pip_angle": nan(),
        "middle_pip_angle": nan(),

        # Pinch-segment motion features
        # Keep the rolling_* column names for compatibility with the existing CSV format,
        # but the actual values are cumulative values over the current pinch segment, not a 30-frame window.
        "direction_sign_x": 0,
        "direction_sign_y": 0,
        "rolling_direction_change_count_x": 0,
        "rolling_direction_change_count_y": 0,
        "rolling_repetition_count": 0,
    }

# Function that returns the list of column names to save in the CSV -> write_csv() uses this list to create the CSV header
def get_frame_fieldnames() -> List[str]:
    return [
        "frame",
        "time_sec",
        "mediapipe_processed",
        "hand_detected",

        "thumb_x",
        "thumb_y",
        "index_x",
        "index_y",
        "middle_x",
        "middle_y",
        "wrist_x",
        "wrist_y",

        "active_x",
        "active_y",

        "active_std_reference_distance",
        "active_x_std_px",
        "active_y_std_px",
        "active_std_total_px",
        "active_x_std_norm",
        "active_y_std_norm",
        "active_std_total_norm",
        "active_std_valid_frames",
        "reference_distance",

        "thumb_index_ratio",
        "index_middle_ratio",
        "thumb_middle_ratio",
        "pinch_detected",
        "pinch_detected_frame_count",
        "thumb_ip_angle",
        "index_pip_angle",
        "middle_pip_angle",

        "direction_sign_x",
        "direction_sign_y",
        "rolling_direction_change_count_x",
        "rolling_direction_change_count_y",
        "rolling_repetition_count",
    ]


# 5. Landmark conversion and feature extraction

# Convert landmark coordinates to actual pixel coordinates
def landmarks_to_pixels(hand_landmarks, width: int, height: int) -> List[Tuple[float, float, float]]:
    points = []

    for lm in hand_landmarks.landmark:
        x = float(lm.x * width)
        y = float(lm.y * height)
        z = float(lm.z)

        points.append((x, y, z))

    return points

# Function that extracts the coordinates of a specific index from the landmark list
def point_xy(landmarks: List[Tuple[float, float, float]], index: int) -> Tuple[float, float]:
    return (landmarks[index][0], landmarks[index][1])

# Function that receives hand landmark information for one frame and creates the feature row for that frame
# In other words, it calculates the values for one CSV row and returns them as a row dictionary
def build_feature_row(frame_number: int, time_sec: float, landmarks: Optional[List[Tuple[float, float, float]]], mediapipe_processed: bool) -> Dict:
    row = make_empty_row(frame_number, time_sec) # Create an empty row

    row["mediapipe_processed"] = bool_to_int(mediapipe_processed) # Record whether MediaPipe was executed on this frame

    if landmarks is None or len(landmarks) < 21: # If landmarks are unavailable, return the empty row as-is
        return row

    row["hand_detected"] = 1 # If a hand is detected
    
    # Extract coordinates
    wrist = point_xy(landmarks, WRIST)

    thumb_mcp = point_xy(landmarks, THUMB_MCP)
    thumb_ip = point_xy(landmarks, THUMB_IP)
    thumb = point_xy(landmarks, THUMB_TIP)

    index_mcp = point_xy(landmarks, INDEX_MCP)
    index_pip = point_xy(landmarks, INDEX_PIP)
    index_dip = point_xy(landmarks, INDEX_DIP)
    index = point_xy(landmarks, INDEX_TIP)

    middle_mcp = point_xy(landmarks, MIDDLE_MCP)
    middle_pip = point_xy(landmarks, MIDDLE_PIP)
    middle_dip = point_xy(landmarks, MIDDLE_DIP)
    middle = point_xy(landmarks, MIDDLE_TIP)

    # Calculate the active point
    active = ((thumb[0] + index[0]) / 2.0, (thumb[1] + index[1]) / 2.0)

    # Calculate reference_distance (distance between the middle finger base and the wrist)
    reference_distance = distance_2d(wrist, middle_mcp)

    thumb_index_distance = distance_2d(thumb, index)
    index_middle_distance = distance_2d(index, middle)
    thumb_middle_distance = distance_2d(thumb, middle)

    thumb_index_ratio = thumb_index_distance / reference_distance
    index_middle_ratio = index_middle_distance / reference_distance
    thumb_middle_ratio = thumb_middle_distance / reference_distance

    # Raw pinch condition:
    # Consider it a pinch only when all three ratios are below or equal to the threshold.
    pinch_detected = (
        is_finite(thumb_index_ratio) and is_finite(index_middle_ratio) and is_finite(thumb_middle_ratio) and
        thumb_index_ratio <= PINCH_RATIO_THRESHOLD and index_middle_ratio <= PINCH_RATIO_THRESHOLD and thumb_middle_ratio <= PINCH_RATIO_THRESHOLD
    )

    row.update({
        "thumb_x": thumb[0],
        "thumb_y": thumb[1],
        "index_x": index[0],
        "index_y": index[1],
        "middle_x": middle[0],
        "middle_y": middle[1],
        "wrist_x": wrist[0],
        "wrist_y": wrist[1],

        "active_x": active[0],
        "active_y": active[1],
        "reference_distance": reference_distance,

        "thumb_index_ratio": thumb_index_ratio,
        "index_middle_ratio": index_middle_ratio,
        "thumb_middle_ratio": thumb_middle_ratio,
        "pinch_detected": bool_to_int(pinch_detected),

        "thumb_ip_angle": angle_2d(thumb_mcp, thumb_ip, thumb),
        "index_pip_angle": angle_2d(index_mcp, index_pip, index_dip),
        "middle_pip_angle": angle_2d(middle_mcp, middle_pip, middle_dip),
    })

    return row



# 6. Pinch-only direction change / repetition features

# Function that converts the direction of active point movement from one frame to the next into -1, 0, or 1
# delta: movement of the active point in the x or y direction, normalized by hand size
def get_direction_sign(delta: float) -> int:
    if not is_finite(delta):
        return 0

    if abs(delta) < DIRECTION_CHANGE_MIN_DELTA_NORM:
        # If the absolute normalized movement is less than 0.02, treat it as MediaPipe jitter and ignore it
        return 0

    return 1 if delta > 0 else -1


def add_motion_features(rows: List[Dict]) -> None:
    """
    Calculate active point movement direction only in frames where a pinch is detected.

    - First frame of pinch: no previous pinch coordinate to compare with, so sign = 0
    - During pinch: compare the active point of the previous pinch frame with the current frame
    - Visible non-pinch: treat it as the end of the pinch and reset the previous active point
    - Hand detection loss: movement cannot be determined, so reset only the previous active point
    """

    previous_active = None

    for row in rows:
        # Do not calculate direction if no hand is detected or if it is not a pinch.
        if (
            row.get("hand_detected", 0) != 1 # 0 is the default value returned if the hand_detected key does not exist in the row
            or row.get("pinch_detected", 0) != 1
        ):
            previous_active = None
            row["direction_sign_x"] = 0
            row["direction_sign_y"] = 0
            continue

        active_x = row.get("active_x", nan())
        active_y = row.get("active_y", nan())
        reference = row.get("reference_distance", nan())

        if not (
            is_finite(active_x)
            and is_finite(active_y)
            and is_finite(reference)
            and float(reference) > 1e-9
        ):
            previous_active = None
            row["direction_sign_x"] = 0
            row["direction_sign_y"] = 0
            continue

        current_active = (float(active_x), float(active_y))

        if previous_active is None:
            dx_norm = 0.0
            dy_norm = 0.0
        else:
            dx_norm = (current_active[0] - previous_active[0]) / float(reference)
            dy_norm = (current_active[1] - previous_active[1]) / float(reference)

        row["direction_sign_x"] = get_direction_sign(dx_norm)
        row["direction_sign_y"] = get_direction_sign(dy_norm)

        previous_active = current_active

    add_pinch_direction_change_features(rows)


# Accumulate the number of direction changes over the entire current pinch segment
def add_pinch_direction_change_features(rows: List[Dict]) -> None:

    direction_change_count_x = 0
    direction_change_count_y = 0
    previous_sign_x = 0
    previous_sign_y = 0

    for row in rows:
        if row.get("pinch_detected", 0) == 1:
            sign_x = int(row.get("direction_sign_x", 0))
            sign_y = int(row.get("direction_sign_y", 0))

            if sign_x != 0:
                if previous_sign_x != 0 and sign_x != previous_sign_x:
                    direction_change_count_x += 1
                previous_sign_x = sign_x

            if sign_y != 0:
                if previous_sign_y != 0 and sign_y != previous_sign_y:
                    direction_change_count_y += 1
                previous_sign_y = sign_y

            row["rolling_direction_change_count_x"] = direction_change_count_x
            row["rolling_direction_change_count_y"] = direction_change_count_y
            row["rolling_repetition_count"] = max(
                direction_change_count_x,
                direction_change_count_y
            )
            continue

        if row.get("hand_detected", 0) != 1:
            # Keep the count when hand detection is temporarily lost.
            # However, reset the sign to avoid a false change caused by directly connecting directions before and after the missing segment.
            previous_sign_x = 0
            previous_sign_y = 0

            row["rolling_direction_change_count_x"] = direction_change_count_x
            row["rolling_direction_change_count_y"] = direction_change_count_y
            row["rolling_repetition_count"] = max(
                direction_change_count_x,
                direction_change_count_y
            )
            continue

        # If the hand is visible but not pinching, treat it as the end of the pinch
        direction_change_count_x = 0
        direction_change_count_y = 0
        previous_sign_x = 0
        previous_sign_y = 0

        row["rolling_direction_change_count_x"] = 0
        row["rolling_direction_change_count_y"] = 0
        row["rolling_repetition_count"] = 0


# 7. Active point standard deviation features

# Calculate active point standard deviation for each complete pinch segment.
def add_active_point_std_features(rows: List[Dict]) -> None:

    # First initialize the standard deviation values for all rows
    for row in rows:
        row["active_std_reference_distance"] = nan()
        row["active_x_std_px"] = nan()
        row["active_y_std_px"] = nan()
        row["active_std_total_px"] = nan()
        row["active_x_std_norm"] = nan()
        row["active_y_std_norm"] = nan()
        row["active_std_total_norm"] = nan()
        row["active_std_valid_frames"] = 0

    current_segment_indices = []
    current_points = []
    current_references = []

    def finalize_segment():
        if len(current_points) < ACTIVE_STD_MIN_VALID_FRAMES:
            return

        xs = np.array([p[0] for p in current_points], dtype=np.float32)
        ys = np.array([p[1] for p in current_points], dtype=np.float32)

        median_reference_distance = float(np.median(current_references))

        if not (
            is_finite(median_reference_distance)
            and median_reference_distance > 1e-9
        ):
            return

        active_x_std_px = float(np.std(xs, ddof=0))
        active_y_std_px = float(np.std(ys, ddof=0))
        active_std_total_px = math.sqrt(
            active_x_std_px ** 2 + active_y_std_px ** 2
        )

        active_x_std_norm = active_x_std_px / median_reference_distance
        active_y_std_norm = active_y_std_px / median_reference_distance
        active_std_total_norm = math.sqrt(
            active_x_std_norm ** 2 + active_y_std_norm ** 2
        )

        for idx in current_segment_indices:
            row = rows[idx]

            row["active_std_reference_distance"] = median_reference_distance
            row["active_x_std_px"] = active_x_std_px
            row["active_y_std_px"] = active_y_std_px
            row["active_std_total_px"] = active_std_total_px

            row["active_x_std_norm"] = active_x_std_norm
            row["active_y_std_norm"] = active_y_std_norm
            row["active_std_total_norm"] = active_std_total_norm

            row["active_std_valid_frames"] = len(current_points)

    for i, row in enumerate(rows):
        if row.get("pinch_detected", 0) == 1:
            active_x = row.get("active_x", nan())
            active_y = row.get("active_y", nan())
            reference_distance = row.get("reference_distance", nan())

            if (
                is_finite(active_x)
                and is_finite(active_y)
                and is_finite(reference_distance)
                and reference_distance > 1e-9
            ):
                current_segment_indices.append(i)
                current_points.append((float(active_x), float(active_y)))
                current_references.append(float(reference_distance))

            continue

        # When the pinch ends, calculate the standard deviation over the entire segment collected so far
        if current_segment_indices:
            finalize_segment()

            current_segment_indices = []
            current_points = []
            current_references = []

    # Handle the case where the pinch continues until the final frame of the video
    if current_segment_indices:
        finalize_segment()



# Count frames during a pinch segment.
def add_pinch_detected_frame_count_features(rows: List[Dict]) -> None:

    pinch_detected_frame_count = 0

    for row in rows:
        if row.get("pinch_detected", 0) == 1:
            pinch_detected_frame_count += 1
            row["pinch_detected_frame_count"] = pinch_detected_frame_count
            continue

        if row.get("hand_detected", 0) != 1:
            # If hand detection is temporarily lost:
            # do not increase the count, but do not reset it either.
            row["pinch_detected_frame_count"] = pinch_detected_frame_count
            continue

        # If the hand is visible but not pinching, treat it as the actual end of the pinch and reset
        pinch_detected_frame_count = 0
        row["pinch_detected_frame_count"] = 0



# 8. Landmark video drawing

def draw_landmark_info(frame, row: Dict) -> None:
    if row["hand_detected"] != 1:
        cv2.putText(
            frame, # Image on which the text will be drawn
            "No hand detected", # Text to display
            (10, 30), # Starting position of the text
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, # Text size
            (0, 0, 255), # Text color
            2, # Text thickness
            cv2.LINE_AA # Option for drawing smoother text lines
        )
        return

    thumb = (int(row["thumb_x"]), int(row["thumb_y"]))
    index = (int(row["index_x"]), int(row["index_y"]))
    middle = (int(row["middle_x"]), int(row["middle_y"]))
    active = (int(row["active_x"]), int(row["active_y"]))

    cv2.circle(frame, thumb, 6, (255, 0, 255), -1)
    cv2.circle(frame, index, 6, (0, 255, 0), -1)
    cv2.circle(frame, middle, 6, (255, 255, 0), -1)
    cv2.circle(frame, active, 7, (0, 255, 255), -1)

    cv2.line(frame, thumb, index, (0, 255, 255), 2)
    cv2.line(frame, index, middle, (255, 255, 0), 2)



# 9. Video processing

def process_video():
    input_path = INPUT_VIDEO_PATH

    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    if OUTPUT_DIR is None:
        output_dir = input_path.parent / "feature_analysis"
    else:
        output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem # Get only the file name without the extension from the file path

    output_video_path = output_dir / f"{stem}_landmarks.mp4"
    output_csv_path = output_dir / f"{stem}_frame_features.csv"

    # Open the video file at input_path using OpenCV
    # cap is an object used to read the video frame by frame
    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    # Get information about the input video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # OpenCV may occasionally fail to read the video FPS correctly and return 0 or an invalid value,
    # so use 30 FPS as a fallback in that case
    if fps <= 0:
        fps = 30.0

    # Set the codec for the output video
    # fourcc is a code that determines how the video is compressed and saved
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") # *"mp4v" = 'm', 'p', '4', 'v'

    # Prepare to create the output video file
    writer = cv2.VideoWriter(
        str(output_video_path), # Path of the video file to save
        fourcc, # Video codec
        fps,
        (width, height) # Output video size
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter: {output_video_path}")

    mp_hands = mp.solutions.hands # Load the MediaPipe hand detection module
    mp_draw = mp.solutions.drawing_utils # Load the drawing utilities provided by MediaPipe

    # Hand object
    hands = mp_hands.Hands(
        static_image_mode=False, # False means the input is treated as video
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.7, # Minimum confidence required for initial hand detection
        min_tracking_confidence=0.5 # Minimum confidence required for tracking
    )

    rows = [] # List for storing the data calculated for each frame

    frame_number = 0 # Variable that stores the current frame number being processed
    processed_count = 0 # Variable that counts the number of frames actually processed
    detected_count = 0 # Variable that counts the number of frames in which a hand was detected

    last_results = None # Variable for storing the previous MediaPipe processing result
    last_landmarks = None # Variable for storing the hand landmarks detected in the previous frame

    start_time = time.time() # Store the current time -> used to calculate the total processing time

    try:
        while True:

            # Read the next frame from the video
            # success indicates whether the frame was read correctly, and frame contains the actual image data
            success, frame = cap.read() 

            if not success or frame is None:
                break

            frame_number += 1

            # Variable indicating whether MediaPipe hand detection was actually executed on the current frame -> initially set to False
            mediapipe_processed = False

            # Run MediaPipe only when the current frame number is a multiple of PROCESS_EVERY_N_FRAMES -> once every 3 frames
            if frame_number % PROCESS_EVERY_N_FRAMES == 0:

                # Convert BGR (OpenCV color order) -> RGB
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Prevent image data from being modified while MediaPipe processes it for a small performance optimization
                rgb.flags.writeable = False

                # MediaPipe detects the hand in the current frame and stores the hand landmark result in last_results
                last_results = hands.process(rgb)

                # Processing is complete, so make the image writable again
                rgb.flags.writeable = True

                # MediaPipe was executed on the current frame, so set this to True and increase the processed frame count
                mediapipe_processed = True
                processed_count += 1

                # Initially set to None because it is not yet known whether a hand was detected in this frame
                last_landmarks = None

                # Check whether MediaPipe produced a result and whether hand landmarks actually exist -> in other words, whether a hand was detected
                if last_results and last_results.multi_hand_landmarks:

                    # Convert MediaPipe landmark coordinates to pixel coordinates
                    # [0] means the first detected hand -> only one hand is used here
                    last_landmarks = landmarks_to_pixels(last_results.multi_hand_landmarks[0], width, height)

                    detected_count += 1 # Increase the count of frames in which a hand was detected

            time_sec = (frame_number - 1) / fps # Calculate the time position of the current frame in the video

            # Create one row containing the feature information for the current frame
            row = build_feature_row(
                frame_number=frame_number,
                time_sec=time_sec,
                landmarks=last_landmarks,
                mediapipe_processed=mediapipe_processed
            )

            # Add the analysis result for the current frame to the rows list
            # rows contains the frame-by-frame feature data for the entire video
            rows.append(row)

            # If a hand is detected, draw the hand landmarks on the original frame
            if last_results and last_results.multi_hand_landmarks:
                for hand_landmarks in last_results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Function that draws additional feature information on the current frame
            draw_landmark_info(frame, row)

            cv2.putText(
                frame,
                f"Frame: {frame_number}/{total_frames}",
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA
            )

            writer.write(frame) # Save the current frame to the output video

            # Print progress every 300 frames
            if frame_number % 300 == 0:
                print(f"Processed {frame_number}/{total_frames} frames...")

    # Clean up resources
    finally: 
        cap.release() # Close the input video file
        writer.release() # Close the output video writer
        hands.close() # Close the MediaPipe Hands object and release related resources


    # !!Post-processing order is important!!

    # 1. pinch-only motion features
    # Accumulate direction changes and repetition count over the entire current pinch segment without using a 30-frame window
    add_motion_features(rows)

    # 2. Calculate pinch-only active_x / active_y standard deviation
    # Standard deviation is shown only in segments where pinch_detected == 1
    add_active_point_std_features(rows)

    # 3. Calculate pinch detected frame count
    # Do not reset when hand detection is temporarily lost; continue counting when a pinch is detected again.
    add_pinch_detected_frame_count_features(rows)

    # Save the fully processed rows to a CSV file
    write_csv(output_csv_path, rows, get_frame_fieldnames())

    elapsed = time.time() - start_time # Calculate video processing time

    # Store summary information about the processing results in a dictionary
    metadata = {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "processed_count": processed_count,
        "detected_count": detected_count,
        "elapsed": elapsed, # Total processing time
        "output_video_path": output_video_path,
        "output_csv_path": output_csv_path,
        "output_dir": output_dir,
        "stem": stem, # Input file name without the extension
    }

    return rows, metadata



# 10. Feature plot generation

# Function that extracts values for a specific column/key from the rows list and converts them into a NumPy array
# rows is a list containing multiple dictionaries
def array_from_rows(rows: List[Dict], key: str) -> np.ndarray:
    return np.array(
        # Iterate through each row in rows and extract the value for the corresponding key
        [
            row.get(key, nan()) # Return nan() if the key does not exist

            # Condition for checking whether the value is a valid number
            if is_finite(row.get(key, nan()))
            else np.nan # If it is NaN, np.nan is inserted into the array -> np.nan represents "no valid numeric value" in NumPy
            for row in rows
        ],
        dtype=np.float32 # Set the numeric type of the array to float32
    )

# Function that extracts already calculated feature values from rows and saves them as a single graph image
# rows: input -> list containing feature information for each frame
def make_feature_plot(rows: List[Dict], output_path: Path) -> None:

    # Create an array containing the time of each frame
    time_values = np.array([row["time_sec"] for row in rows], dtype=np.float32)

    plot_mode_label = "Raw frame values"
    value_label_suffix = "raw"

    # 1. Finger distance ratios
    thumb_index_ratio = array_from_rows(rows, "thumb_index_ratio") # Get key (thumb_index_ratio) from rows -> retrieve all thumb_index_ratio values
    index_middle_ratio = array_from_rows(rows, "index_middle_ratio")
    thumb_middle_ratio = array_from_rows(rows, "thumb_middle_ratio")

    # 2. Finger joint angles
    thumb_ip_angle = array_from_rows(rows, "thumb_ip_angle")
    index_pip_angle = array_from_rows(rows, "index_pip_angle")
    middle_pip_angle = array_from_rows(rows, "middle_pip_angle")

    # 3. Active point x/y
    active_x = array_from_rows(rows, "active_x")
    active_y = array_from_rows(rows, "active_y")

    # 4. Normalized active point standard deviation
    # Plot the standard deviation only in segments where pinch_detected == 1
    active_x_std_norm = array_from_rows(rows, "active_x_std_norm")
    active_y_std_norm = array_from_rows(rows, "active_y_std_norm")
    active_std_total_norm = array_from_rows(rows, "active_std_total_norm")

    # 5. Pinch-segment direction change / repetition count
    # The existing column names are rolling_*, but the values are cumulative values for the current pinch segment.
    rolling_change_x = array_from_rows(rows, "rolling_direction_change_count_x")
    rolling_change_y = array_from_rows(rows, "rolling_direction_change_count_y")
    rolling_repetition = array_from_rows(rows, "rolling_repetition_count")

    # 6. Pinch detected frame count
    pinch_detected_frame_count = array_from_rows(rows, "pinch_detected_frame_count")

    fig, axes = plt.subplots(
        6, # 6 rows
        1, # 1 column
        figsize=(16, 18), # Graph image size
        sharex=True # All six graphs share the same x-axis (time axis)
    )

    # 1. Finger distance ratios
    axes[0].plot(
        time_values,
        thumb_index_ratio,
        label=f"thumb-index ratio {value_label_suffix}"
    )
    axes[0].plot(
        time_values,
        index_middle_ratio,
        label=f"index-middle ratio {value_label_suffix}"
    )
    axes[0].plot(
        time_values,
        thumb_middle_ratio,
        label=f"thumb-middle ratio {value_label_suffix}"
    )

    # Draw a horizontal reference line for the pinch threshold
    axes[0].axhline(
        y=PINCH_RATIO_THRESHOLD,
        linestyle="--",
        linewidth=1,
        label=f"pinch threshold ({PINCH_RATIO_THRESHOLD})"
    )
    axes[0].set_title(
        f"Normalized finger distance ratios - {plot_mode_label}"
    )
    axes[0].set_ylabel("ratio")
    axes[0].legend(loc="upper right")


    # 2. Finger joint angles
    axes[1].plot(
        time_values,
        thumb_ip_angle,
        label=f"thumb IP angle {value_label_suffix}"
    )
    axes[1].plot(
        time_values,
        index_pip_angle,
        label=f"index PIP angle {value_label_suffix}"
    )
    axes[1].plot(
        time_values,
        middle_pip_angle,
        label=f"middle PIP angle {value_label_suffix}"
    )
    axes[1].set_title(
        f"Finger joint angles - {plot_mode_label}"
    )
    axes[1].set_ylabel("degrees")
    axes[1].legend(loc="upper right")


    # 3. Active point x/y
    axes[2].plot(
        time_values,
        active_x,
        label=f"active point x {value_label_suffix}"
    )
    axes[2].plot(
        time_values,
        active_y,
        label=f"active point y {value_label_suffix}"
    )
    axes[2].set_title(
        f"Active point position over time - {plot_mode_label}"
    )
    axes[2].set_ylabel("pixel")
    axes[2].legend(loc="upper right")


    # 4. Active point standard deviation
    axes[3].plot(
        time_values,
        active_x_std_norm,
        label="active x std norm (pinch-only)"
    )
    axes[3].plot(
        time_values,
        active_y_std_norm,
        label="active y std norm (pinch-only)"
    )
    axes[3].plot(
        time_values,
        active_std_total_norm,
        label="active std total norm (pinch-only)"
    )
    axes[3].set_title(
    "Normalized active point standard deviation "
    "(pinch segment-level)"
    )
    axes[3].set_ylabel("normalized std")
    axes[3].legend(loc="upper right")


    # 5. Pinch-segment direction change / repetition count
    axes[4].step(
        time_values,
        rolling_change_x,
        where="post",
        label="pinch direction changes x"
    )
    axes[4].step(
        time_values,
        rolling_change_y,
        where="post",
        label="pinch direction changes y"
    )
    axes[4].step(
        time_values,
        rolling_repetition,
        where="post",
        label="pinch repetition count"
    )
    axes[4].set_title(
        "Pinch-segment direction change / repetition count "
        "(no rolling window)"
    )
    axes[4].set_ylabel("count")
    axes[4].legend(loc="upper right")


    # 6. Pinch detected frame count
    axes[5].step(
        time_values,
        pinch_detected_frame_count,
        where="post",
        label="pinch detected frame count"
    )
    axes[5].set_title(
        "Pinch detected frame count "
        "(keeps previous count during temporary hand detection loss)"
    )
    axes[5].set_ylabel("frames")
    axes[5].set_xlabel("time (sec)")
    axes[5].legend(loc="upper right")

    fig.tight_layout() # Automatically adjust spacing so the graphs do not overlap
    fig.savefig(output_path, dpi=150) # Save the graph as an image file
    plt.close(fig) # Close the figure to avoid wasting memory



# 11. Main

def main():
    input_path = INPUT_VIDEO_PATH

    if OUTPUT_DIR is None:
        output_dir = input_path.parent / "feature_analysis"
    else:
        output_dir = Path(OUTPUT_DIR)

    # Create the output folder if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem # File name without the extension

    # Determine where to save the feature plot image
    feature_plot_path = output_dir / f"{stem}_feature_plot_raw.png"

    print("---MediaPipe feature analysis---")
    print()

    rows, metadata = process_video() # Analyze the video
    make_feature_plot(rows=rows, output_path=feature_plot_path) # Create the graph

    print("Saved files")
    print("Finished.")


if __name__ == "__main__":
    main()
