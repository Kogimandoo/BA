import csv
import math
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import cv2
import mediapipe as mp


# 0. Input / output

#INPUT_VIDEO_PATH = Path("/home/pi4/farm/Arducam_tof_camera/touch_videos/plant_touch_2026-06-25_14-21-20.mp4")
#INPUT_VIDEO_PATH = Path("/home/pi4/farm/Arducam_tof_camera/touch_videos/plant_touch_2026-06-25_14-27-00.mp4")
#INPUT_VIDEO_PATH = Path("/home/pi4/farm/Arducam_tof_camera/touch_videos/plant_touch_2026-06-25_14-25-37.mp4")
INPUT_VIDEO_PATH = Path("/home/pi4/farm/Arducam_tof_camera/touch_videos/plant_touch_2026-06-25_14-37-06.mp4")

OUTPUT_DIR = None


# 1. MediaPipe landmark indices

WRIST = 0

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

PINKY_MCP = 17


# 2. Feature / threshold settings

PROCESS_EVERY_N_FRAMES = 3

# Minimum number of valid frames required to calculate the standard deviation over the entire pinch segment
ACTIVE_STD_MIN_VALID_FRAMES = 3

# Threshold used to avoid counting very small MediaPipe jitter as a direction change
DIRECTION_CHANGE_MIN_DELTA_NORM = 0.02

PINCH_RATIO_THRESHOLD = 0.7

RUB_MAX_ACTIVE_STD_TOTAL_NORM = 0.2
RUB_MIN_REPEATED_NUMBER = 2

PULL_MIN_ACTIVE_STD_TOTAL_NORM = 0.3
PULL_THUMB_IP_ANGLE_MAX = 170.0

TOUCH_INDEX_PIP_ANGLE_MIN = 160.0
TOUCH_MIDDLE_PIP_ANGLE_MAX = 120.0



# 3. Utility functions

def nan() -> float:
    return float("nan")


def is_finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except Exception:
        return False


def bool_to_int(value: bool) -> int:
    return 1 if value else 0


def safe_divide(a: float, b: float) -> float:
    if not is_finite(a) or not is_finite(b) or abs(float(b)) < 1e-9:
        return nan()
    return float(a) / float(b)


def distance_2d(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# Calculate angle ABC in degrees. b is the joint point.
# When the finger is extended, the angle is close to 180 degrees; when the finger is bent, the angle becomes smaller.
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
        return nan()

    ba_x = a[0] - b[0]
    ba_y = a[1] - b[1]
    bc_x = c[0] - b[0]
    bc_y = c[1] - b[1]

    norm_ba = math.hypot(ba_x, ba_y)
    norm_bc = math.hypot(bc_x, bc_y)

    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return nan()

    cos_angle = (ba_x * bc_x + ba_y * bc_y) / (norm_ba * norm_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return math.degrees(math.acos(cos_angle))


def format_float(value, digits=3) -> str:
    if not is_finite(value):
        return ""
    return f"{float(value):.{digits}f}"



# 4. Feature row structure

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
        "reference_distance": nan(),

        "thumb_index_ratio": nan(),
        "index_middle_ratio": nan(),
        "thumb_middle_ratio": nan(),
        "pinch_detected": 0,

        "thumb_ip_angle": nan(),
        "index_pip_angle": nan(),
        "middle_pip_angle": nan(),

        # Cumulative direction-change features within the pinch segment
        "direction_sign_x": 0,
        "direction_sign_y": 0,
        "rolling_direction_change_count_x": 0,
        "rolling_direction_change_count_y": 0,
        "rolling_repetition_count": 0,

        "touch_with_index": 0,
        "touch_with_index_thumb_opened": 0,
        "non_pinch_touch_candidate": 0,
    }


def get_action_log_fieldnames() -> List[str]:
    return [
        "event_id",
        "action_name",
        "event_source",

        "start_frame",
        "end_frame",
        "start_time_sec",
        "end_time_sec",

        "defined_at_frame",
        "defined_at_time_sec",

        "total_frames",
        "pinch_detected_frames",
        "active_std_total_norm",
        "minimum_thumb_ip_angle",
        "repeated_number",

        "touch_with_index_seen",
        "touch_with_index_thumb_opened_seen",
        "touching_pinch_or_supporting_seen",
        "rubbing_and_smelling_seen",
        "pulling_dead_leaves_seen",
    ]



# 5. Landmark conversion and feature extraction

def landmarks_to_pixels(hand_landmarks, width: int, height: int) -> List[Tuple[float, float, float]]:
    points = []

    for lm in hand_landmarks.landmark:
        x = float(lm.x * width)
        y = float(lm.y * height)
        z = float(lm.z)
        points.append((x, y, z))

    return points


def point_xy(landmarks: List[Tuple[float, float, float]], index: int) -> Tuple[float, float]:
    return (landmarks[index][0], landmarks[index][1])


def is_touch_with_index(row: Optional[Dict]) -> bool:

    if row is None:
        return False

    return (
        is_finite(row.get("index_middle_ratio", nan())) and
        is_finite(row.get("thumb_middle_ratio", nan())) and
        is_finite(row.get("index_pip_angle", nan())) and
        is_finite(row.get("middle_pip_angle", nan())) and
        row["index_middle_ratio"] > row["thumb_middle_ratio"] and
        row["index_pip_angle"] >= TOUCH_INDEX_PIP_ANGLE_MIN and
        row["middle_pip_angle"] <= TOUCH_MIDDLE_PIP_ANGLE_MAX
    )


def is_touch_with_index_thumb_opened(row: Optional[Dict]) -> bool:

    if row is None:
        return False

    return (
        is_finite(row.get("thumb_index_ratio", nan())) and
        is_finite(row.get("thumb_middle_ratio", nan())) and
        is_finite(row.get("index_pip_angle", nan())) and
        is_finite(row.get("middle_pip_angle", nan())) and
        row["thumb_index_ratio"] > row["thumb_middle_ratio"] and
        row["index_pip_angle"] >= TOUCH_INDEX_PIP_ANGLE_MIN and
        row["middle_pip_angle"] <= TOUCH_MIDDLE_PIP_ANGLE_MAX
    )


def is_non_pinch_touch_candidate(row: Optional[Dict]) -> bool:
    if row is None:
        return False

    if row.get("pinch_detected", 0) == 1:
        return False

    return (
        is_touch_with_index(row) or
        is_touch_with_index_thumb_opened(row)
    )


def build_feature_row(
    frame_number: int,
    time_sec: float,
    landmarks: Optional[List[Tuple[float, float, float]]],
    mediapipe_processed: bool
) -> Dict:
    row = make_empty_row(frame_number, time_sec)
    row["mediapipe_processed"] = bool_to_int(mediapipe_processed)

    if landmarks is None or len(landmarks) < 21:
        return row

    row["hand_detected"] = 1

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

    pinky_mcp = point_xy(landmarks, PINKY_MCP)

    active = ((thumb[0] + index[0]) / 2.0, (thumb[1] + index[1]) / 2.0)

    reference_distance = distance_2d(wrist, middle_mcp)

    if not is_finite(reference_distance) or reference_distance <= 1.0:
        reference_distance = distance_2d(index_mcp, pinky_mcp)

    thumb_index_distance = distance_2d(thumb, index)
    index_middle_distance = distance_2d(index, middle)
    thumb_middle_distance = distance_2d(thumb, middle)

    thumb_index_ratio = safe_divide(thumb_index_distance, reference_distance)
    index_middle_ratio = safe_divide(index_middle_distance, reference_distance)
    thumb_middle_ratio = safe_divide(thumb_middle_distance, reference_distance)

    pinch_detected = (
        is_finite(thumb_index_ratio) and
        is_finite(index_middle_ratio) and
        is_finite(thumb_middle_ratio) and
        thumb_index_ratio <= PINCH_RATIO_THRESHOLD and
        index_middle_ratio <= PINCH_RATIO_THRESHOLD and
        thumb_middle_ratio <= PINCH_RATIO_THRESHOLD
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

    row["touch_with_index"] = bool_to_int(is_touch_with_index(row))
    row["touch_with_index_thumb_opened"] = bool_to_int(is_touch_with_index_thumb_opened(row))
    row["non_pinch_touch_candidate"] = bool_to_int(is_non_pinch_touch_candidate(row))

    return row



# 6. Motion / active standard deviation features

def get_direction_sign(delta: float) -> int:
    if not is_finite(delta):
        return 0

    if abs(delta) < DIRECTION_CHANGE_MIN_DELTA_NORM:
        return 0

    return 1 if delta > 0 else -1


# Calculate the active point movement direction only while a pinch is detected, and accumulate the number of x/y direction changes from the start of the current pinch segment to the current frame
def add_motion_features(rows: List[Dict]) -> None:

    previous_active = None
    previous_sign_x = 0
    previous_sign_y = 0

    direction_change_count_x = 0
    direction_change_count_y = 0

    for row in rows:
        # If hand detection is temporarily lost, keep the accumulated count.
        # However, do not directly compare the coordinates before the detection loss with the coordinates after detection resumes.
        if row.get("hand_detected", 0) != 1:
            previous_active = None
            previous_sign_x = 0
            previous_sign_y = 0

            row["direction_sign_x"] = 0
            row["direction_sign_y"] = 0
            row["rolling_direction_change_count_x"] = direction_change_count_x
            row["rolling_direction_change_count_y"] = direction_change_count_y
            row["rolling_repetition_count"] = max(
                direction_change_count_x,
                direction_change_count_y
            )
            continue

        # If the hand is visible but not pinching, treat it as the actual end of the pinch or as a non-pinch segment.
        if row.get("pinch_detected", 0) != 1:
            previous_active = None
            previous_sign_x = 0
            previous_sign_y = 0
            direction_change_count_x = 0
            direction_change_count_y = 0

            row["direction_sign_x"] = 0
            row["direction_sign_y"] = 0
            row["rolling_direction_change_count_x"] = 0
            row["rolling_direction_change_count_y"] = 0
            row["rolling_repetition_count"] = 0
            continue

        active_x = row.get("active_x", nan())
        active_y = row.get("active_y", nan())
        reference = row.get("reference_distance", nan())

        if not (
            is_finite(active_x) and
            is_finite(active_y) and
            is_finite(reference) and
            reference > 1e-9
        ):
            previous_active = None
            previous_sign_x = 0
            previous_sign_y = 0

            row["direction_sign_x"] = 0
            row["direction_sign_y"] = 0
            row["rolling_direction_change_count_x"] = direction_change_count_x
            row["rolling_direction_change_count_y"] = direction_change_count_y
            row["rolling_repetition_count"] = max(
                direction_change_count_x,
                direction_change_count_y
            )
            continue

        current_active = (float(active_x), float(active_y))

        if previous_active is None:
            direction_sign_x = 0
            direction_sign_y = 0
        else:
            dx_norm = (current_active[0] - previous_active[0]) / reference
            dy_norm = (current_active[1] - previous_active[1]) / reference

            direction_sign_x = get_direction_sign(dx_norm)
            direction_sign_y = get_direction_sign(dy_norm)

        row["direction_sign_x"] = direction_sign_x
        row["direction_sign_y"] = direction_sign_y

        # A value of 0 represents small jitter, so ignore it without changing the previous valid direction.
        if direction_sign_x != 0:
            if previous_sign_x != 0 and direction_sign_x != previous_sign_x:
                direction_change_count_x += 1
            previous_sign_x = direction_sign_x

        if direction_sign_y != 0:
            if previous_sign_y != 0 and direction_sign_y != previous_sign_y:
                direction_change_count_y += 1
            previous_sign_y = direction_sign_y

        row["rolling_direction_change_count_x"] = direction_change_count_x
        row["rolling_direction_change_count_y"] = direction_change_count_y
        row["rolling_repetition_count"] = max(
            direction_change_count_x,
            direction_change_count_y
        )

        previous_active = current_active


# Calculate the standard deviation of active_x / active_y over the entire pinch segment and normalize it using the median reference_distance of that segment
def calculate_active_std_total_norm(rows: List[Dict]) -> float:

    active_x_values = []
    active_y_values = []
    reference_values = []

    for row in rows:
        if row.get("pinch_detected", 0) != 1:
            continue

        active_x = row.get("active_x", nan())
        active_y = row.get("active_y", nan())
        reference = row.get("reference_distance", nan())

        if not (
            is_finite(active_x) and
            is_finite(active_y) and
            is_finite(reference) and
            reference > 1e-9
        ):
            continue

        active_x_values.append(float(active_x))
        active_y_values.append(float(active_y))
        reference_values.append(float(reference))

    if len(active_x_values) < ACTIVE_STD_MIN_VALID_FRAMES:
        return nan()

    median_reference_distance = float(statistics.median(reference_values))

    if not (
        is_finite(median_reference_distance) and
        median_reference_distance > 1e-9
    ):
        return nan()

    active_x_std_px = float(statistics.pstdev(active_x_values))
    active_y_std_px = float(statistics.pstdev(active_y_values))

    active_x_std_norm = active_x_std_px / median_reference_distance
    active_y_std_norm = active_y_std_px / median_reference_distance

    return math.sqrt(
        active_x_std_norm ** 2 + active_y_std_norm ** 2
    )


def add_all_action_features(rows: List[Dict]) -> List[Dict]:
    copied_rows = [dict(row) for row in rows]
    add_motion_features(copied_rows)
    return copied_rows



# 7. Action classification

def summarize_pinch_records(records: List[Dict]) -> Dict:
    rows = add_all_action_features(records)

    active_std_total_norm = calculate_active_std_total_norm(rows)

    repeated_number = max(
        [int(row.get("rolling_repetition_count", 0)) for row in rows],
        default=0
    )

    pinch_detected_frames = sum(
        1 for row in rows
        if row.get("pinch_detected", 0) == 1
    )

    thumb_ip_angles = [
        float(row["thumb_ip_angle"])
        for row in rows
        if (
            row.get("pinch_detected", 0) == 1 and
            is_finite(row.get("thumb_ip_angle", nan()))
        )
    ]

    minimum_thumb_ip_angle = (
        min(thumb_ip_angles) if thumb_ip_angles else nan()
    )

    rubbing_and_smelling_seen = (
        is_finite(active_std_total_norm) and
        active_std_total_norm <= RUB_MAX_ACTIVE_STD_TOTAL_NORM 
        and repeated_number >= RUB_MIN_REPEATED_NUMBER
    )

    pulling_dead_leaves_seen = (
        is_finite(active_std_total_norm) and
        active_std_total_norm >= PULL_MIN_ACTIVE_STD_TOTAL_NORM and
        is_finite(minimum_thumb_ip_angle) and
        minimum_thumb_ip_angle <= PULL_THUMB_IP_ANGLE_MAX
    )

    return {
        "total_frames": len(rows),
        "pinch_detected_frames": pinch_detected_frames,
        "active_std_total_norm": active_std_total_norm,
        "minimum_thumb_ip_angle": minimum_thumb_ip_angle,
        "repeated_number": repeated_number,

        "touch_with_index_seen": False,
        "touch_with_index_thumb_opened_seen": False,
        "touching_pinch_or_supporting_seen": pinch_detected_frames > 0,

        "rubbing_and_smelling_seen": rubbing_and_smelling_seen,
        "pulling_dead_leaves_seen": pulling_dead_leaves_seen,
    }


def classify_pinch_records(records: List[Dict]) -> Tuple[Optional[str], Dict]:
    summary = summarize_pinch_records(records)

    # Check the pulling and rubbing criteria first.
    # Since pinch/supporting is a broad TOUCHING condition that includes all pinches, check it last.
    if summary["pulling_dead_leaves_seen"]:
        return "PULLING DEAD LEAVES", summary

    if summary["rubbing_and_smelling_seen"]:
        return "RUBBING & SMELLING", summary

    if summary["touching_pinch_or_supporting_seen"]:
        return "TOUCHING", summary

    return None, summary


def summarize_non_pinch_touch_records(records: List[Dict]) -> Dict:
    touch_with_index_frames = sum(
        1 for row in records
        if is_touch_with_index(row)
    )

    touch_with_index_thumb_opened_frames = sum(
        1 for row in records
        if is_touch_with_index_thumb_opened(row)
    )

    return {
        "total_frames": len(records),
        "pinch_detected_frames": 0,
        "active_std_total_norm": nan(),
        "minimum_thumb_ip_angle": nan(),
        "repeated_number": 0,

        "touch_with_index_seen": touch_with_index_frames > 0,
        "touch_with_index_thumb_opened_seen": touch_with_index_thumb_opened_frames > 0,
        "touching_pinch_or_supporting_seen": False,

        "rubbing_and_smelling_seen": False,
        "pulling_dead_leaves_seen": False,
    }


def make_event(
    action_name: str,
    event_source: str,
    records: List[Dict],
    defined_at_frame: int,
    defined_at_time_sec: float,
    summary: Dict,
) -> Dict:
    return {
        "action_name": action_name,
        "event_source": event_source,

        "start_frame": records[0]["frame"],
        "end_frame": records[-1]["frame"],
        "start_time_sec": records[0]["time_sec"],
        "end_time_sec": records[-1]["time_sec"],

        "defined_at_frame": defined_at_frame,
        "defined_at_time_sec": defined_at_time_sec,

        "summary": summary,
    }


def print_event(event: Dict) -> None:
    summary = event["summary"]
    active_std_text = format_float(
        summary.get("active_std_total_norm", nan()),
        digits=3
    )
    thumb_ip_text = format_float(
        summary.get("minimum_thumb_ip_angle", nan()),
        digits=1
    )



# 8. Drawing


def draw_landmark_info(frame, row: Dict) -> None:

    if row.get("hand_detected", 0) != 1:
        return

    thumb = (int(row["thumb_x"]), int(row["thumb_y"]))
    index = (int(row["index_x"]), int(row["index_y"]))
    middle = (int(row["middle_x"]), int(row["middle_y"]))
    active = (int(row["active_x"]), int(row["active_y"]))

    cv2.circle(frame, thumb, 6, (255, 0, 255), -1)
    cv2.circle(frame, index, 6, (0, 255, 0), -1)
    cv2.circle(frame, middle, 6, (255, 255, 0), -1)
    cv2.circle(frame, active, 7, (0, 255, 255), -1)


def draw_action_label(frame, event: Dict) -> None:

    cv2.putText(
        frame,
        event["action_name"],
        (15, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )



# 9. CSV output


def write_action_log_csv(path: Path, events: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=get_action_log_fieldnames())
        writer.writeheader()

        for event_id, event in enumerate(events, start=1):
            summary = event["summary"]

            writer.writerow({
                "event_id": event_id,
                "action_name": event["action_name"],
                "event_source": event["event_source"],

                "start_frame": event["start_frame"],
                "end_frame": event["end_frame"],
                "start_time_sec": format_float(event["start_time_sec"]),
                "end_time_sec": format_float(event["end_time_sec"]),

                "defined_at_frame": event["defined_at_frame"],
                "defined_at_time_sec": format_float(event["defined_at_time_sec"]),

                "total_frames": summary.get("total_frames", 0),
                "pinch_detected_frames": summary.get("pinch_detected_frames", 0),
                "active_std_total_norm": format_float(
                    summary.get("active_std_total_norm", nan())
                ),
                "minimum_thumb_ip_angle": format_float(
                    summary.get("minimum_thumb_ip_angle", nan())
                ),
                "repeated_number": summary.get("repeated_number", 0),

                "touch_with_index_seen": bool_to_int(
                    summary.get("touch_with_index_seen", False)
                ),
                "touch_with_index_thumb_opened_seen": bool_to_int(
                    summary.get("touch_with_index_thumb_opened_seen", False)
                ),
                "touching_pinch_or_supporting_seen": bool_to_int(
                    summary.get("touching_pinch_or_supporting_seen", False)
                ),
                "rubbing_and_smelling_seen": bool_to_int(
                    summary.get("rubbing_and_smelling_seen", False)
                ),
                "pulling_dead_leaves_seen": bool_to_int(
                    summary.get("pulling_dead_leaves_seen", False)
                ),
            })



# 10. Video processing


def process_video():
    input_path = INPUT_VIDEO_PATH

    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")

    if OUTPUT_DIR is None:
        output_dir = input_path.parent / "classification_offline"
    else:
        output_dir = Path(OUTPUT_DIR)

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem

    output_video_path = output_dir / f"{stem}_action_label_after_definition.mp4"
    output_csv_path = output_dir / f"{stem}_action_log.csv"

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_video_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open VideoWriter: {output_video_path}")

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )

    events = []

    pinch_records = []
    touch_records = []

    current_action_event = None

    frame_number = 0
    processed_count = 0
    detected_count = 0

    last_results = None
    last_landmarks = None

    start_time = time.time()

    try:
        while True:
            success, frame = cap.read()

            if not success or frame is None:
                break

            frame_number += 1

            mediapipe_processed = False

            if frame_number % PROCESS_EVERY_N_FRAMES == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                last_results = hands.process(rgb)
                rgb.flags.writeable = True

                mediapipe_processed = True
                processed_count += 1

                last_landmarks = None

                if last_results and last_results.multi_hand_landmarks:
                    last_landmarks = landmarks_to_pixels(
                        last_results.multi_hand_landmarks[0],
                        width,
                        height
                    )
                    detected_count += 1

            time_sec = (frame_number - 1) / fps

            row = build_feature_row(
                frame_number=frame_number,
                time_sec=time_sec,
                landmarks=last_landmarks,
                mediapipe_processed=mediapipe_processed
            )

            
            # 1) Pinch-based actions
            
            if row.get("pinch_detected", 0) == 1:
                # Clear the previous action label when a new action candidate starts
                if len(pinch_records) == 0:
                    current_action_event = None

                pinch_records.append(row)

            else:
                # As soon as pinch_detected becomes != 1, classify the pinch segment that has just ended
                if len(pinch_records) > 0:
                    action_name, summary = classify_pinch_records(pinch_records)

                    if action_name is not None:
                        event = make_event(
                            action_name=action_name,
                            event_source="pinch",
                            records=pinch_records,
                            defined_at_frame=frame_number,
                            defined_at_time_sec=time_sec,
                            summary=summary
                        )

                        events.append(event)
                        current_action_event = event
                        print_event(event)

                    pinch_records = []


            # 2) Non-pinch Touch

            if row.get("non_pinch_touch_candidate", 0) == 1:
                # Clear the previous action label when a new action candidate starts
                if len(touch_records) == 0:
                    current_action_event = None

                touch_records.append(row)

            else:
                # As soon as the touch condition becomes False, classify the completed segment as TOUCHING
                if len(touch_records) > 0:
                    summary = summarize_non_pinch_touch_records(touch_records)

                    event = make_event(
                        action_name="TOUCHING",
                        event_source="non_pinch_touch",
                        records=touch_records,
                        defined_at_frame=frame_number,
                        defined_at_time_sec=time_sec,
                        summary=summary
                    )

                    events.append(event)
                    current_action_event = event
                    print_event(event)

                    touch_records = []


            # 3) Drawing

            if last_results and last_results.multi_hand_landmarks:
                for hand_landmarks in last_results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )
                    break

            draw_landmark_info(frame, row)

            if current_action_event is not None:
                draw_action_label(frame, current_action_event)

            writer.write(frame)

            if frame_number % 300 == 0:
                print(f"Processed {frame_number}/{total_frames} frames...")

        # If a pinch/touch candidate remains active until the end of the video,
        # it cannot be displayed in the video because there is no next frame, but it is still recorded in the CSV.
        final_time_sec = (frame_number - 1) / fps if frame_number > 0 else 0.0

        if len(pinch_records) > 0:
            action_name, summary = classify_pinch_records(pinch_records)

            if action_name is not None:
                event = make_event(
                    action_name=action_name,
                    event_source="pinch",
                    records=pinch_records,
                    defined_at_frame=frame_number,
                    defined_at_time_sec=final_time_sec,
                    summary=summary
                )

                events.append(event)
                print_event(event)

        if len(touch_records) > 0:
            summary = summarize_non_pinch_touch_records(touch_records)

            event = make_event(
                action_name="TOUCHING",
                event_source="non_pinch_touch",
                records=touch_records,
                defined_at_frame=frame_number,
                defined_at_time_sec=final_time_sec,
                summary=summary
            )

            events.append(event)
            print_event(event)

    finally:
        cap.release()
        writer.release()
        hands.close()

    write_action_log_csv(output_csv_path, events)

    elapsed = time.time() - start_time

    metadata = {
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "processed_count": processed_count,
        "detected_count": detected_count,
        "elapsed": elapsed,
        "output_video_path": output_video_path,
        "output_csv_path": output_csv_path,
        "output_dir": output_dir,
        "stem": stem,
    }

    return events, metadata



# 11. Main


def main():
    input_path = INPUT_VIDEO_PATH

    if OUTPUT_DIR is None:
        output_dir = input_path.parent / "classification_offline"
    else:
        output_dir = Path(OUTPUT_DIR)

    print("--- Offline gesture classification ---")

    events, metadata = process_video()

    print("Finished.")


if __name__ == "__main__":
    main()
