"""
- Arducam ToF camera connected via CSI  -> /dev/video0
- USB webcam C310 connected via USB     -> /dev/video1

http://192.168.0.128:8000

"""

import time
import math
import statistics
import json
import threading
import cv2
import mediapipe as mp
import paho.mqtt.client as mqtt
from flask import Flask, Response, jsonify

app = Flask(__name__)


# MQTT / Smart-farm light control

MQTT_BROKER = "192.168.0.128"
MQTT_PORT = 1883
MQTT_TOPIC = "balconyfarmN/1020BAC9AC7A/server/command"

MQTT_ENABLED = True
LIGHT_EFFECT_LOCK = threading.Lock()

mqtt_client = mqtt.Client()

try:
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
    print(f"Connected to MQTT Broker: {MQTT_BROKER}")
except Exception as e:
    MQTT_ENABLED = False
    print(f"MQTT Connection Error: {e}")


def send_light_cmd(brightness=100, color_mode=1):

    if not MQTT_ENABLED:
        print("[MQTT] Skipped because MQTT is not connected.")
        return

    payload = {
        "cmd": "light",
        "auto_mode": 0,
        "led_mode": 0,
        "brightness": int(brightness),
        "color_mode": int(color_mode),
        "on_time": 360,
        "off_time": 1080,
        "everyday": 0
    }

    mqtt_client.publish(MQTT_TOPIC, json.dumps(payload))

# Run light effects in a separate thread so the video stream is not blocked.
def run_light_sequence(sequence, effect_name):

    with LIGHT_EFFECT_LOCK:
        print(f"[LIGHT] Effect start: {effect_name}")

        for brightness, color_mode, delay_after in sequence:
            send_light_cmd(brightness=brightness, color_mode=color_mode)

            if delay_after > 0:
                time.sleep(delay_after)

        print(f"[LIGHT] Effect end: {effect_name}")


def trigger_light_effect(action_name):

    if action_name == "TOUCHING":
        sequence = [
            # (brightness, color_mode, delay)
            (80, 1, 0.3),
            (50, 1, 0.3),
            (80, 1, 0.3),
            (100, 1, 0.0),
        ]

    elif action_name == "PULLING DEAD LEAVES":
        sequence = [
            # (brightness, color_mode, delay)
            (100, 3, 0.3),
            (80, 1, 0.3),
            (50, 3, 0.3),
            (100, 1, 0.0)
        ]

    elif action_name == "RUBBING & SMELLING":
        sequence = [
            # (brightness, color_mode, delay)
            (100, 3, 0.3),
            (100, 1, 0.0),
        ]

    else:
        return

    threading.Thread(
        target=run_light_sequence,
        args=(sequence, action_name),
        daemon=True
    ).start()


mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

CAMERA_DEVICE = "/dev/video1"

cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print(f"ERROR: Could not open USB webcam: {CAMERA_DEVICE}")
else:
    print(f"USB webcam opened successfully: {CAMERA_DEVICE}")

    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    fourcc_str = "".join([
        chr((actual_fourcc >> 8 * i) & 0xFF)
        for i in range(4)
    ])

    print(f"Actual width : {actual_width}")
    print(f"Actual height: {actual_height}")
    print(f"Actual FPS   : {actual_fps}")
    print(f"Actual FOURCC: {fourcc_str}")



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


# 2. Processing / threshold settings

PROCESS_EVERY_N_FRAMES = 3

latest_finger_coords = None

# Number of grace frames used to continue the same gesture even if the hand briefly disappears
GESTURE_END_GRACE_FRAMES = 3

# Time interval used to prevent the same action from being printed too frequently
ACTION_PRINT_COOLDOWN = 0.8

# pinch threshold
PINCH_RATIO_THRESHOLD = 0.7

# Minimum number of valid frames required to calculate the active point standard deviation over the entire pinch segment
ACTIVE_STD_MIN_VALID_FRAMES = 3

# Threshold used to avoid counting very small MediaPipe jitter as a direction change
# The rolling repetition count is accumulated over the entire pinch segment without using a 30-frame window.
DIRECTION_CHANGE_MIN_DELTA_NORM = 0.02

# Action definition thresholds
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


def safe_divide(a: float, b: float) -> float:
    if not is_finite(a) or not is_finite(b) or abs(float(b)) < 1e-9:
        return nan()
    return float(a) / float(b)


def distance(p1, p2) -> float:
    if p1 is None or p2 is None:
        return 0.0

    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def angle_2d(a, b, c) -> float:

    if a is None or b is None or c is None:
        return nan()

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

    norm_ba = math.sqrt(ba_x ** 2 + ba_y ** 2)
    norm_bc = math.sqrt(bc_x ** 2 + bc_y ** 2)

    if norm_ba < 1e-9 or norm_bc < 1e-9:
        return nan()

    cos_angle = (ba_x * bc_x + ba_y * bc_y) / (norm_ba * norm_bc)
    cos_angle = max(-1.0, min(1.0, cos_angle))

    return math.degrees(math.acos(cos_angle))


def landmark_to_pixel(landmark, width, height):
    x = float(landmark.x * width)
    y = float(landmark.y * height)
    return (x, y)


def int_point(point):
    return (int(point[0]), int(point[1]))



# 4. Motion / standard-deviation helper functions


def get_direction_sign(delta: float) -> int:
    if not is_finite(delta):
        return 0

    if abs(delta) < DIRECTION_CHANGE_MIN_DELTA_NORM:
        return 0

    return 1 if delta > 0 else -1


def add_live_motion_features(rows):

    previous_active = None
    previous_sign_x = 0
    previous_sign_y = 0

    direction_change_count_x = 0
    direction_change_count_y = 0

    for row in rows:
        active_point = row.get("active_point")
        reference = row.get("reference_distance", nan())

        # If the current record is not a pinch or the coordinates/reference distance are invalid
        # treat the current pinch segment as finished and reset all direction states.
        if (
            not row.get("pinch_detected", False) or
            active_point is None or
            not is_finite(active_point[0]) or
            not is_finite(active_point[1]) or
            not is_finite(reference) or
            reference <= 1e-9
        ):
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

        current_active = (
            float(active_point[0]),
            float(active_point[1])
        )

        # The first valid frame of a pinch segment has no previous coordinate, so set the movement direction to 0
        if previous_active is None:
            direction_sign_x = 0
            direction_sign_y = 0
        else:
            dx_norm = (
                current_active[0] - previous_active[0]
            ) / reference
            dy_norm = (
                current_active[1] - previous_active[1]
            ) / reference

            direction_sign_x = get_direction_sign(dx_norm)
            direction_sign_y = get_direction_sign(dy_norm)

        row["direction_sign_x"] = direction_sign_x
        row["direction_sign_y"] = direction_sign_y

        # A value of 0 represents a small movement, so do not change the previous valid direction.
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


def add_live_active_std_features(rows):

    for row in rows:
        row["active_std_total_norm"] = nan()

    current_segment_indices = []
    current_points = []
    current_references = []

    def finalize_segment():
        if len(current_points) < ACTIVE_STD_MIN_VALID_FRAMES:
            return

        median_reference = statistics.median(current_references)

        if not is_finite(median_reference) or median_reference <= 1e-9:
            return

        xs = [point[0] for point in current_points]
        ys = [point[1] for point in current_points]

        active_x_std_norm = statistics.pstdev(xs) / median_reference
        active_y_std_norm = statistics.pstdev(ys) / median_reference

        active_std_total_norm = math.sqrt(
            active_x_std_norm ** 2 +
            active_y_std_norm ** 2
        )

        for idx in current_segment_indices:
            rows[idx]["active_std_total_norm"] = active_std_total_norm

    for i, row in enumerate(rows):
        if row.get("pinch_detected", False):
            active_point = row.get("active_point")
            reference = row.get("reference_distance", nan())

            if (
                active_point is not None and
                is_finite(active_point[0]) and
                is_finite(active_point[1]) and
                is_finite(reference) and
                reference > 1e-9
            ):
                current_segment_indices.append(i)
                current_points.append(
                    (float(active_point[0]), float(active_point[1]))
                )
                current_references.append(float(reference))

            continue

        if current_segment_indices:
            finalize_segment()
            current_segment_indices = []
            current_points = []
            current_references = []

    # If the pinch remains active until the end of the records
    if current_segment_indices:
        finalize_segment()

# Copy the current gesture records and calculate the features required for action recognition.
def add_all_runtime_analysis_features(records):

    rows = [dict(r) for r in records]

    add_live_motion_features(rows)
    add_live_active_std_features(rows)

    return rows



# 5. Hand feature extraction


def extract_hand_features(hand_landmarks, width, height):
    wrist = landmark_to_pixel(hand_landmarks.landmark[WRIST], width, height)

    thumb_mcp = landmark_to_pixel(hand_landmarks.landmark[THUMB_MCP], width, height)
    thumb_ip = landmark_to_pixel(hand_landmarks.landmark[THUMB_IP], width, height)
    thumb_tip = landmark_to_pixel(hand_landmarks.landmark[THUMB_TIP], width, height)

    index_mcp = landmark_to_pixel(hand_landmarks.landmark[INDEX_MCP], width, height)
    index_pip = landmark_to_pixel(hand_landmarks.landmark[INDEX_PIP], width, height)
    index_dip = landmark_to_pixel(hand_landmarks.landmark[INDEX_DIP], width, height)
    index_tip = landmark_to_pixel(hand_landmarks.landmark[INDEX_TIP], width, height)

    middle_mcp = landmark_to_pixel(hand_landmarks.landmark[MIDDLE_MCP], width, height)
    middle_pip = landmark_to_pixel(hand_landmarks.landmark[MIDDLE_PIP], width, height)
    middle_dip = landmark_to_pixel(hand_landmarks.landmark[MIDDLE_DIP], width, height)
    middle_tip = landmark_to_pixel(hand_landmarks.landmark[MIDDLE_TIP], width, height)

    pinky_mcp = landmark_to_pixel(hand_landmarks.landmark[PINKY_MCP], width, height)

    # Reference distance used for hand-size normalization
    reference_distance = distance(wrist, middle_mcp)

    if not is_finite(reference_distance) or reference_distance <= 1.0:
        reference_distance = distance(index_mcp, pinky_mcp)

    thumb_index_distance = distance(thumb_tip, index_tip)
    index_middle_distance = distance(index_tip, middle_tip)
    thumb_middle_distance = distance(thumb_tip, middle_tip)

    thumb_index_ratio = safe_divide(thumb_index_distance, reference_distance)
    index_middle_ratio = safe_divide(index_middle_distance, reference_distance)
    thumb_middle_ratio = safe_divide(thumb_middle_distance, reference_distance)

    # Pinch condition defined in the table
    pinch_detected = (
        is_finite(thumb_index_ratio) and
        is_finite(index_middle_ratio) and
        is_finite(thumb_middle_ratio) and
        thumb_index_ratio <= PINCH_RATIO_THRESHOLD and
        index_middle_ratio <= PINCH_RATIO_THRESHOLD and
        thumb_middle_ratio <= PINCH_RATIO_THRESHOLD
    )

    # Close flags used for display
    thumb_index_close = (
        is_finite(thumb_index_ratio) and
        thumb_index_ratio <= PINCH_RATIO_THRESHOLD
    )

    index_middle_close = (
        is_finite(index_middle_ratio) and
        index_middle_ratio <= PINCH_RATIO_THRESHOLD
    )

    # Use the midpoint between the thumb and index finger as the active point during a pinch, and the index fingertip for a normal touch
    if pinch_detected:
        active_point = (
            (thumb_tip[0] + index_tip[0]) / 2.0,
            (thumb_tip[1] + index_tip[1]) / 2.0
        )
    else:
        active_point = index_tip

    return {
        "t": time.time(),

        "thumb_tip": thumb_tip,
        "index_tip": index_tip,
        "middle_tip": middle_tip,
        "active_point": active_point,

        "reference_distance": reference_distance,

        "thumb_index_ratio": thumb_index_ratio,
        "index_middle_ratio": index_middle_ratio,
        "thumb_middle_ratio": thumb_middle_ratio,

        "pinch_detected": pinch_detected,
        "thumb_index_pinch": thumb_index_close,
        "index_middle_pinch": index_middle_close,

        "thumb_ip_angle": angle_2d(thumb_mcp, thumb_ip, thumb_tip),
        "index_pip_angle": angle_2d(index_mcp, index_pip, index_dip),
        "middle_pip_angle": angle_2d(middle_mcp, middle_pip, middle_dip),
    }



# 6. Gesture tracker


class GestureTracker:
    def __init__(self):
        self.active = False
        self.records = []
        self.missing_frames = 0
        self.last_global_print_time = 0.0

    def reset(self):
        self.active = False
        self.records = []
        self.missing_frames = 0

    def is_valid_record(self, record):
        # Record every frame in which a hand is detected as a gesture candidate.
        return record is not None

    def update(self, record):
        valid_record = self.is_valid_record(record)

        if valid_record:
            if not self.active:
                self.active = True
                self.records = []
                self.missing_frames = 0

            self.records.append(record)
            self.missing_frames = 0

        else:
            if self.active:
                self.missing_frames += 1

                if self.missing_frames >= GESTURE_END_GRACE_FRAMES:
                    action_name, summary = self.classify()

                    if action_name is not None:
                        self.print_action(action_name, summary)

                    self.reset()

    def summarize(self):
        if len(self.records) == 0:
            return None

        rows = add_all_runtime_analysis_features(self.records)

        repeated_number = max(
            [int(r.get("rolling_repetition_count", 0)) for r in rows],
            default=0
        )

        pinch_detected_frames = sum(
            1 for r in rows
            if r.get("pinch_detected", False)
        )


        std_values = [
            float(r.get("active_std_total_norm", nan()))
            for r in rows
            if (
                r.get("pinch_detected", False) and
                is_finite(r.get("active_std_total_norm", nan()))
            )
        ]

        active_std_total_norm = max(std_values) if std_values else nan()

        thumb_ip_values = [
            float(r.get("thumb_ip_angle", nan()))
            for r in rows
            if (
                r.get("pinch_detected", False) and
                is_finite(r.get("thumb_ip_angle", nan()))
            )
        ]

        minimum_thumb_ip_angle = min(thumb_ip_values) if thumb_ip_values else nan()

        # Touch 1: with index

        touch_with_index_seen = any(
            is_finite(r.get("index_middle_ratio", nan())) and
            is_finite(r.get("thumb_middle_ratio", nan())) and
            is_finite(r.get("index_pip_angle", nan())) and
            is_finite(r.get("middle_pip_angle", nan())) and
            r["index_middle_ratio"] > r["thumb_middle_ratio"] and
            r["index_pip_angle"] >= TOUCH_INDEX_PIP_ANGLE_MIN and
            r["middle_pip_angle"] <= TOUCH_MIDDLE_PIP_ANGLE_MAX
            for r in rows
        )

        # Touch 2: with index, thumb opened

        touch_with_index_thumb_opened_seen = any(
            is_finite(r.get("thumb_index_ratio", nan())) and
            is_finite(r.get("thumb_middle_ratio", nan())) and
            is_finite(r.get("index_pip_angle", nan())) and
            is_finite(r.get("middle_pip_angle", nan())) and
            r["thumb_index_ratio"] > r["thumb_middle_ratio"] and
            r["index_pip_angle"] >= TOUCH_INDEX_PIP_ANGLE_MIN and
            r["middle_pip_angle"] <= TOUCH_MIDDLE_PIP_ANGLE_MAX
            for r in rows
        )

        # Touch 3: Pinch OR supporting

        touching_pinch_or_supporting_seen = any(
            r.get("pinch_detected", False)
            for r in rows
        )

        touching_seen = (
            touch_with_index_seen or
            touch_with_index_thumb_opened_seen or
            touching_pinch_or_supporting_seen
        )

        # Rubbing and smelling:
        rubbing_and_smelling_seen = any(
            r.get("pinch_detected", False) and
            is_finite(r.get("active_std_total_norm", nan())) and
            r["active_std_total_norm"] <= RUB_MAX_ACTIVE_STD_TOTAL_NORM and
            int(r.get("rolling_repetition_count", 0)) >= RUB_MIN_REPEATED_NUMBER
            for r in rows
        )

        # Pulling dead leaves:
        pulling_dead_leaves_seen = any(
            r.get("pinch_detected", False) and
            is_finite(r.get("thumb_ip_angle", nan())) and
            is_finite(r.get("active_std_total_norm", nan())) and
            r["thumb_ip_angle"] <= PULL_THUMB_IP_ANGLE_MAX and
            r["active_std_total_norm"] >= PULL_MIN_ACTIVE_STD_TOTAL_NORM
            for r in rows
        )

        return {
            "total_frames": len(rows),
            "pinch_detected_frames": pinch_detected_frames,
            "active_std_total_norm": active_std_total_norm,
            "minimum_thumb_ip_angle": minimum_thumb_ip_angle,
            "repeated_number": repeated_number,
            "touch_with_index_seen": touch_with_index_seen,
            "touch_with_index_thumb_opened_seen": touch_with_index_thumb_opened_seen,
            "touching_pinch_or_supporting_seen": touching_pinch_or_supporting_seen,
            "touching_seen": touching_seen,
            "rubbing_and_smelling_seen": rubbing_and_smelling_seen,
            "pulling_dead_leaves_seen": pulling_dead_leaves_seen,
        }

    def classify(self):

        summary = self.summarize()

        if summary is None:
            return None, None

        # Pulling dead leaves:
        if summary["pulling_dead_leaves_seen"]:
            return "PULLING DEAD LEAVES", summary

        # Rubbing and smelling:
        if summary["rubbing_and_smelling_seen"]:
            return "RUBBING & SMELLING", summary

        # Touching:
        # 1. With index
        # 2. With index, thumb opened
        # 3. Pinch OR supporting

        if summary["touching_seen"]:
            return "TOUCHING", summary

        return None, summary

    def print_action(self, action_name, summary):
        now = time.time()

        if now - self.last_global_print_time < ACTION_PRINT_COOLDOWN:
            return

        self.last_global_print_time = now
        now_text = time.strftime("%H:%M:%S")

        active_std = summary.get("active_std_total_norm", nan())
        minimum_thumb_ip = summary.get("minimum_thumb_ip_angle", nan())

        active_std_text = (
            f"{active_std:.3f}"
            if is_finite(active_std)
            else "NaN"
        )

        minimum_thumb_ip_text = (
            f"{minimum_thumb_ip:.1f}"
            if is_finite(minimum_thumb_ip)
            else "NaN"
        )

        print(f"[{now_text}] ACTION DETECTED -> {action_name}")

        trigger_light_effect(action_name)


gesture_tracker = GestureTracker()



# 7. Flask video stream

def gen_frames():
    global latest_finger_coords

    prev_time = time.time()
    frame_count = 0
    last_results = None

    while True:
        if not cap.isOpened():
            print("Camera is not opened.")
            time.sleep(1)
            continue

        success, frame = cap.read()

        if not success or frame is None:
            print("Failed to read frame from /dev/video1")
            time.sleep(0.1)
            continue

        frame_count += 1

        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if current_time != prev_time else 0.0
        prev_time = current_time

        processed_this_frame = False

        if frame_count % PROCESS_EVERY_N_FRAMES == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            rgb.flags.writeable = False
            last_results = hands.process(rgb)
            rgb.flags.writeable = True

            processed_this_frame = True

        results = last_results

        latest_finger_coords = None
        hand_feature_record = None

        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape
                features = extract_hand_features(hand_landmarks, w, h)

                index_tip = int_point(features["index_tip"])
                thumb_tip = int_point(features["thumb_tip"])
                middle_tip = int_point(features["middle_tip"])
                active_point = int_point(features["active_point"])

                latest_finger_coords = index_tip
                hand_feature_record = features

                cv2.circle(frame, index_tip, 6, (0, 255, 0), -1)
                cv2.circle(frame, thumb_tip, 5, (255, 0, 255), -1)
                cv2.circle(frame, middle_tip, 5, (255, 0, 0), -1)
                cv2.circle(frame, active_point, 6, (0, 255, 255), -1)

                if features["pinch_detected"]:
                    cv2.line(frame, thumb_tip, index_tip, (0, 255, 255), 2)


                break

        else:
            cv2.putText(
                frame,
                "No hand detected",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

        # Update action recognition only on frames where MediaPipe was actually executed.
        if processed_this_frame:
            gesture_tracker.update(hand_feature_record)

        cv2.putText(
            frame,
            f"FPS: {fps:.0f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            print("Failed to encode frame as JPEG")
            continue

        frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/")
def index():
    return """
    <html>
      <head>
        <title>Raspberry Pi MediaPipe Hand Tracking</title>
      </head>
      <body style="text-align: center; background-color: #222; color: white; font-family: Arial;">
        <h1>MediaPipe Hand Tracking</h1>

        <img src="/stream"
             style="border: 3px solid white; border-radius: 10px; max-width: 95%;">

      </body>
    </html>
    """


@app.route("/stream")
def stream():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/coords")
def coords():
    return jsonify({"index_finger": latest_finger_coords})


if __name__ == "__main__":
    try:
        app.run(host="0.0.0.0", port=8000, threaded=True, debug=False)
    finally:
        print("Releasing camera...")
        cap.release()
        hands.close()

        if MQTT_ENABLED:
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            print("MQTT disconnected.")
