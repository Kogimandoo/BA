import cv2
import time
from datetime import datetime
from pathlib import Path


# settings

SAVE_DIR = Path("/home/pi4/farm/Arducam_tof_camera/touch_videos")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

RECORD_SECONDS = 60

CAMERA_DEVICE = "/dev/video1"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30


# Main recording logic

def main():
    print(f"Opening USB webcam: {CAMERA_DEVICE}")

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    if not cap.isOpened():
        print(f"ERROR: Could not open USB webcam: {CAMERA_DEVICE}")
        print()
        print("Check available devices with:")
        print("  v4l2-ctl --list-devices")
        print("  ls /dev/video*")
        return

    ret, frame = cap.read()

    if not ret or frame is None:
        print(f"ERROR: Camera opened but could not read frame from {CAMERA_DEVICE}")
        cap.release()
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))

    fourcc_str = "".join([
        chr((actual_fourcc >> 8 * i) & 0xFF)
        for i in range(4)
    ])

    print("Camera opened successfully.")
    print(f"Device       : {CAMERA_DEVICE}")
    print(f"Width        : {actual_width}")
    print(f"Height       : {actual_height}")
    print(f"FPS          : {actual_fps}")
    print(f"FOURCC       : {fourcc_str}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = SAVE_DIR / f"plant_touch_{timestamp}.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        FPS,
        (actual_width, actual_height)
    )

    if not writer.isOpened():
        print("ERROR: Could not open VideoWriter.")
        cap.release()
        return

    print()
    print("============================================")
    print("Recording started.")
    print(f"Duration : {RECORD_SECONDS} seconds")
    print(f"Device   : {CAMERA_DEVICE}")
    print(f"Save to  : {output_path}")
    print("============================================")
    print()

    start_time = time.time()
    frame_count = 0

    try:
        while True:
            elapsed = time.time() - start_time

            if elapsed >= RECORD_SECONDS:
                break

            ret, frame = cap.read()

            if not ret or frame is None:
                print("WARNING: Failed to read frame.")
                time.sleep(0.01)
                continue

            frame_count += 1

            writer.write(frame)

    finally:
        cap.release()
        writer.release()

    total_time = time.time() - start_time
    actual_recording_fps = frame_count / total_time if total_time > 0 else 0

    print()
    print("============================================")
    print("Recording finished.")
    print(f"Saved file : {output_path}")
    print(f"Frames     : {frame_count}")
    print(f"Actual FPS : {actual_recording_fps:.2f}")
    print("============================================")


if __name__ == "__main__":
    main()