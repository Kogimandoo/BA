'''
Combine the 06:00 and 18:00 npy files to create a daily median,
then compare the initial n-day baseline with the most recent n days
'''

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

DATA_DIR = Path("/home/pi4/farm/Arducam_tof_camera/plant_growth/data")
OUTPUT_DIR = Path("/home/pi4/farm/Arducam_tof_camera/plant_growth/analysis")
OUTPUT_IMAGE = OUTPUT_DIR / "compare6.png"

CHANGE_THRESHOLD_MM = 10.0

# Image cell size
CELL_WIDTH = 320
CELL_HEIGHT = 240

# Width for displaying the date
DATE_LABEL_WIDTH = 140

ROI = {
    "x": 35,
    "y": 0,
    "w": 100,
    "h": 135
}

DRAW_ROI_RECTANGLE = True

# Depth visualization range
VIS_MIN_DEPTH_MM = 500
VIS_MAX_DEPTH_MM = 1000

# Mask colors (BGR)
COLOR_DISTANCE_INCREASED = (0, 0, 255)    # Red
COLOR_DISTANCE_DECREASED = (0, 255, 0)    # Green
MASK_ALPHA = 0.6

# For noise reduction
SPATIAL_MEDIAN_KERNEL = 5

# Comparison settings
BASELINE_WINDOW_DAYS = 3
CURRENT_WINDOW_DAYS = 3

# Takes a file path in Path format as input and returns a datetime object
def parse_timestamp_from_filename(path: Path) -> datetime:
    name = path.name.replace("_depth.npy", "")
    return datetime.strptime(name, "%Y-%m-%d_%H-%M-%S") # Convert the string into actual date and time data


def find_depth_files() -> list[Path]:
    files = sorted(DATA_DIR.glob("*_depth.npy"))
    if len(files) == 0:
        raise RuntimeError(f"No .npy depth files found in {DATA_DIR}")
    return files

# Load the depth data stored in the .npy file and convert its data type
def load_depth(path: Path) -> np.ndarray: # Return a NumPy array
    return np.load(path).astype(np.float32)

# Function that applies a median blur to reduce spatial noise within each depth image
def smooth_depth(depth: np.ndarray) -> np.ndarray:
    if SPATIAL_MEDIAN_KERNEL <= 1:
        return depth

    if SPATIAL_MEDIAN_KERNEL % 2 == 0:
        raise ValueError("SPATIAL_MEDIAN_KERNEL must be odd.")

    depth = np.ascontiguousarray(depth.astype(np.float32))
    return cv2.medianBlur(depth, SPATIAL_MEDIAN_KERNEL)

# Function that creates a mask where only the ROI area is marked as True
def create_roi_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape # -> Size of the depth image

    if ROI is None: # -> If None, analyze the entire image without restricting the area
        return np.ones((height, width), dtype=bool)

    mask = np.zeros((height, width), dtype=bool) # Create a mask where the entire image is initially set to False

    x = int(ROI["x"])
    y = int(ROI["y"])
    w = int(ROI["w"])
    h = int(ROI["h"])

    # Safety measure to prevent the ROI from extending outside the image
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + w)
    y2 = min(height, y + h)

    mask[y1:y2, x1:x2] = True # Set the ROI area to True -> only this area will be analyzed
    return mask # -> Return a mask where the ROI is True and the rest is False

# Function that draws an ROI rectangle on the image
def draw_roi_rectangle(image: np.ndarray) -> np.ndarray:
    if ROI is None: # If there is no ROI, return the original image without drawing a rectangle
        return image

    result = image.copy()
    height, width = result.shape[:2]

    x = int(ROI["x"])
    y = int(ROI["y"])
    w = int(ROI["w"])
    h = int(ROI["h"])

    # Limit the coordinates so the rectangle does not extend outside the image
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width - 1, x + w) # If width is 240, the last x-coordinate is 239, so subtract 1
    y2 = min(height - 1, y + h)

    cv2.rectangle(
        result,
        (x1, y1),
        (x2, y2),
        (0, 255, 255),
        1
    )

    return result

# Function that converts a numeric depth array into a grayscale image
def depth_to_gray(depth: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    gray = (depth - min_depth) / (max_depth - min_depth) * 255 # Convert depth values into brightness values between 0 and 255
    gray = np.clip(gray, 0, 255) # Limit values below 0 to 0 and values above 255 to 255
    gray = gray.astype(np.uint8) # Convert to uint8 format for use as an OpenCV image -> integer values between 0 and 255

    return gray # Return the final grayscale depth image


def add_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    size: float = 0.45,
    color: tuple[int, int, int] = (0, 0, 0),
    background: bool = False
) -> None:
    if background:
        text_size, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            1
        )
        text_width, text_height = text_size

        cv2.rectangle(
            image,
            (x - 4, y - text_height - 4),
            (x + text_width + 4, y + 4),
            (255, 255, 255),
            -1
        )

    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        1,
        cv2.LINE_AA
    )


# Function that groups depth files by date
def group_files_by_date(files: list[Path]) -> dict[str, list[Path]]:
    grouped = defaultdict(list) # Dictionary for storing files grouped by date

    for file in files: # Read the date from each file name and add the file to the corresponding date group
        timestamp = parse_timestamp_from_filename(file)
        date_key = timestamp.strftime("%Y-%m-%d") # Extract only the date as a string
        grouped[date_key].append(file) # Add the current file to the corresponding date group

    for date_key in grouped: # Sort the files within each date in chronological order
        grouped[date_key] = sorted( # Sort based on the actual date and time extracted from the file name
            grouped[date_key],
            key=parse_timestamp_from_filename
        )

    return dict(sorted(grouped.items()))

# Function that loads multiple depth npy files captured on the same day and creates one representative depth image for that date
def load_daily_median_depth(
    date_files: list[Path], # List of depth files corresponding to a specific date
    depth_cache: dict[Path, np.ndarray] # Dictionary for storing already loaded depth data
) -> np.ndarray:

    if len(date_files) == 0: # Raise an error if there are no files for the corresponding date
        raise RuntimeError("No depth files for this date.")

    depth_list = [] # List for storing loaded depth arrays

    for file in date_files: # Process each file for the corresponding date
        if file not in depth_cache: # Check whether the file has already been loaded
            depth = load_depth(file) # Load the npy file
            depth = smooth_depth(depth) # Apply median blur to the loaded depth image
            depth_cache[file] = depth # Store the blurred depth array in depth_cache

        depth_list.append(depth_cache[file])

    stacked = np.stack(depth_list, axis=0)

    return np.median(stacked, axis=0).astype(np.float32) # Create a representative daily depth image


def median_of_depths(depth_list: list[np.ndarray]) -> np.ndarray:
    if len(depth_list) == 0:
        raise RuntimeError("Cannot compute median from empty depth list.")

    stacked = np.stack(depth_list, axis=0)
    return np.median(stacked, axis=0).astype(np.float32)


def create_change_overlay(
    baseline_depth: np.ndarray,
    current_depth: np.ndarray,
    roi_mask: np.ndarray,
    min_depth: float,
    max_depth: float
) -> np.ndarray:
    current_gray = depth_to_gray(current_depth, min_depth, max_depth)
    result = cv2.cvtColor(current_gray, cv2.COLOR_GRAY2BGR)

    diff = current_depth - baseline_depth

    distance_increased_mask = roi_mask & (diff >= CHANGE_THRESHOLD_MM)
    distance_decreased_mask = roi_mask & (diff <= -CHANGE_THRESHOLD_MM)

    red = np.array(COLOR_DISTANCE_INCREASED, dtype=np.float32)
    green = np.array(COLOR_DISTANCE_DECREASED, dtype=np.float32)

    result[distance_increased_mask] = (
        result[distance_increased_mask].astype(np.float32) * (1.0 - MASK_ALPHA)
        + red * MASK_ALPHA
    ).astype(np.uint8)

    result[distance_decreased_mask] = (
        result[distance_decreased_mask].astype(np.float32) * (1.0 - MASK_ALPHA)
        + green * MASK_ALPHA
    ).astype(np.uint8)

    if DRAW_ROI_RECTANGLE:
        result = draw_roi_rectangle(result)

    return result


def create_cell(
    date: str,
    baseline_depth: np.ndarray,
    current_depth: np.ndarray,
    roi_mask: np.ndarray,
    min_depth: float,
    max_depth: float,
    baseline_count: int,
    current_count: int
) -> np.ndarray:
    image = create_change_overlay(
        baseline_depth=baseline_depth,
        current_depth=current_depth,
        roi_mask=roi_mask,
        min_depth=min_depth,
        max_depth=max_depth
    )

    image = cv2.resize(image, (CELL_WIDTH, CELL_HEIGHT))

    return image

# Function that creates the title area at the top of the final image
def create_legend(width: int) -> np.ndarray:
    legend_height = 100
    legend = np.full((legend_height, width, 3), 255, dtype=np.uint8)

    add_text(legend, "Daily median growth comparison", 10, 25, 0.55)

    return legend


def main():

    files = find_depth_files() # Find npy files
    grouped = group_files_by_date(files) # Group them by date

    first_depth = load_depth(files[0]) # Load the first depth file to determine the image size
    roi_mask = create_roi_mask(first_depth.shape) # Create an ROI mask according to that size

    min_depth = VIS_MIN_DEPTH_MM
    max_depth = VIS_MAX_DEPTH_MM

    depth_cache = {} # Storage for avoiding repeated loading of depth files -> reuse values stored in memory when the same npy file is needed multiple times
    daily_depths_by_date = {} # Store the representative depth image for each date

    # Precompute the daily median depth for each date
    for date, date_files in grouped.items():
        daily_depths_by_date[date] = load_daily_median_depth(date_files, depth_cache)

    all_dates = sorted(daily_depths_by_date.keys()) # Sort the dates

    rows = [] # List for storing each date row in the final image

    for i, date in enumerate(all_dates):
        # Baseline = initial n days
        baseline_dates = all_dates[:BASELINE_WINDOW_DAYS]

        # Current = most recent n days up to the current date
        start_idx = max(0, i - CURRENT_WINDOW_DAYS + 1) # At the beginning, fewer than 3 days may be available, so use only the available dates -> max prevents the index from becoming negative
        current_dates = all_dates[start_idx:i + 1]

        baseline_depths = [daily_depths_by_date[d] for d in baseline_dates]
        current_depths = [daily_depths_by_date[d] for d in current_dates]

        baseline_depth = median_of_depths(baseline_depths) # Calculate the median again using the daily median depths from the initial 3 days
        current_depth = median_of_depths(current_depths) # Calculate the median again using the daily median depths from the most recent 3 days

        cell = create_cell(
            date=date,
            baseline_depth=baseline_depth,
            current_depth=current_depth,
            roi_mask=roi_mask,
            min_depth=min_depth,
            max_depth=max_depth,
            baseline_count=len(baseline_dates),
            current_count=len(current_dates)
        )

        label_area = np.full((CELL_HEIGHT, DATE_LABEL_WIDTH, 3), 255, dtype=np.uint8)
        add_text(label_area, date, 10, 35, 0.45)

        row = np.hstack([label_area, cell]) # Horizontally concatenate the date label area and the comparison image
        rows.append(row) # Add the row to the rows list

    # The number of captures may differ by date -> some days may have 2 images while others may have only 1, so find the width of the longest row
    max_width = max(row.shape[1] for row in rows)

    padded_rows = [] # List for adding white padding to shorter rows
    for row in rows: # Get the height and width of each row
        h, w = row.shape[:2] # -> Height, width
        if w < max_width: # If the current row is narrower than max_width, add white padding to the right -> all rows must have the same width for np.vstack()
            padding = np.full((h, max_width - w, 3), 255, dtype=np.uint8)
            row = np.hstack([row, padding])
        padded_rows.append(row)

    final_image = np.vstack(padded_rows) # Vertically concatenate all date rows

    legend = create_legend(final_image.shape[1])
    final_image = np.vstack([legend, final_image]) # Add the title area at the top

    cv2.imwrite(str(OUTPUT_IMAGE), final_image)

    print("Daily median growth comparison created successfully.")
    print(f"Saved result image: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()