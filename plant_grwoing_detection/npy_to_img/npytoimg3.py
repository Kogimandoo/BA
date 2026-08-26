'''
Compare with the data from the same time on the previous day
'''

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

DATA_DIR = Path("/home/pi4/farm/Arducam_tof_camera/plant_growth/data")
OUTPUT_DIR = Path("/home/pi4/farm/Arducam_tof_camera/plant_growth/analysis")
OUTPUT_IMAGE = OUTPUT_DIR / "compare3.png"

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

# Visualization range used when converting depth values into a grayscale image
VIS_MIN_DEPTH_MM = 500 # Map 500 to black
VIS_MAX_DEPTH_MM = 1000 # Map 1000 to white


# Mask colors -> OpenCV uses BGR, not RGB
COLOR_DISTANCE_INCREASED = (0, 0, 255)    # Red
COLOR_DISTANCE_DECREASED = (0, 255, 0)    # Green
MASK_ALPHA = 0.6                          # Controls how strongly the mask color is overlaid


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

# Function that creates a mask where only the ROI area is marked as True
def create_roi_mask(shape: tuple[int, int]) -> np.ndarray:
    height, width = shape # -> Size of the depth image

    if ROI is None: # -> If None, analyze the entire image without restricting the area
        return np.ones((height, width), dtype=bool)

    mask = np.zeros((height, width), dtype=bool) # Create a mask where the entire image is initially set to False

    # Get the ROI coordinates
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
    if ROI is None: # If there is no ROI, there is no need to draw a rectangle, so return the original image
        return image

    result = image.copy() # Create a copy instead of modifying the original image directly

    height, width = result.shape[:2] # Get the height and width of the image

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
        (0, 255, 255), # Yellow
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

    if background: # If background=True, draw a white box behind the text first
        text_size, _ = cv2.getTextSize( # Calculate the width and height of the text and draw a white box that fits it
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            size,
            1
        )

        text_width, text_height = text_size

        cv2.rectangle( # Draw the white background box behind the text
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

# Function that compares the baseline depth with the current depth and overlays colored masks on changed areas
def create_change_overlay(
    baseline_depth: np.ndarray,
    current_depth: np.ndarray,
    roi_mask: np.ndarray,
    min_depth: float,
    max_depth: float
) -> np.ndarray:

    # First convert the current depth into a grayscale image, then convert it into a BGR color image
    # because the image must be in color to add the change overlay mask
    current_gray = depth_to_gray(current_depth, min_depth, max_depth)
    result = cv2.cvtColor(current_gray, cv2.COLOR_GRAY2BGR)

    diff = current_depth - baseline_depth

    # Create masks only for areas where the change exceeds the threshold
    distance_increased_mask = roi_mask & (diff >= CHANGE_THRESHOLD_MM)
    distance_decreased_mask = roi_mask & (diff <= -CHANGE_THRESHOLD_MM)

    # Prepare the mask colors as NumPy arrays
    red = np.array(COLOR_DISTANCE_INCREASED, dtype=np.float32)
    green = np.array(COLOR_DISTANCE_DECREASED, dtype=np.float32)

    # Overlay red on areas where the distance increased -> mix 40% original image and 60% red
    result[distance_increased_mask] = (
        result[distance_increased_mask].astype(np.float32) * (1.0 - MASK_ALPHA)
        + red * MASK_ALPHA
    ).astype(np.uint8)

    # Overlay green on areas where the distance decreased -> mix 40% original image and 60% green
    result[distance_decreased_mask] = (
        result[distance_decreased_mask].astype(np.float32) * (1.0 - MASK_ALPHA)
        + green * MASK_ALPHA
    ).astype(np.uint8)

    if DRAW_ROI_RECTANGLE: # If DRAW_ROI_RECTANGLE = True, draw the ROI rectangle on the result image
        result = draw_roi_rectangle(result)

    return result # Return the final color image with the change masks overlaid


# Function that groups depth files by date
def group_files_by_date(files: list[Path]) -> dict[str, list[Path]]:
    grouped = defaultdict(list) # Dictionary for storing files grouped by date

    for file in files: # Read the date from each file name and add the file to the corresponding date group
        timestamp = parse_timestamp_from_filename(file)
        date_key = timestamp.strftime("%Y-%m-%d")
        grouped[date_key].append(file)

    for date_key in grouped: # Sort the files within each date in chronological order
        grouped[date_key] = sorted(grouped[date_key])

    return dict(sorted(grouped.items())) # Sort the dates themselves and return the result

# Function that creates the title area at the top of the final image
def create_legend(width: int) -> np.ndarray:
    # Create a white image with a height of 80 pixels
    legend_height = 80
    legend = np.full((legend_height, width, 3), 255, dtype=np.uint8)

    add_text(legend, "Plant growth detection", 10, 25, 0.55)

    return legend

# Function that organizes the file list into a dictionary so files can be easily found by date and time
def build_file_map_by_date_time(files: list[Path]) -> dict[tuple[str, str], Path]:
    file_map = {} # Empty dictionary

    for file in files: # Iterate through each file in the input file list
        timestamp = parse_timestamp_from_filename(file) # Extract the date and time from the file name
        date_key = timestamp.strftime("%Y-%m-%d") # Extract only the date from the timestamp as a string
        time_key = timestamp.strftime("%H:%M:%S") # Extract only the time from the timestamp as a string
        file_map[(date_key, time_key)] = file # Store it in the dictionary

    return file_map


def main():
    files = find_depth_files() # Find *_depth.npy files inside DATA_DIR

    # Load the first depth file to determine the image size -> create the ROI mask based on this size
    first_depth = load_depth(files[0])
    roi_mask = create_roi_mask(first_depth.shape)

    min_depth = VIS_MIN_DEPTH_MM
    max_depth = VIS_MAX_DEPTH_MM

    grouped = group_files_by_date(files) # Group the files by date

    # Create a dictionary that allows files to be found directly by date and time
    # -> used to quickly find the file from the same time on the previous day
    file_map = build_file_map_by_date_time(files)

    rows = [] # List for storing each date row in the final image

    for date, date_files in grouped.items(): # Iterate through each date
        cells = [] # List for storing image cells for each time on the corresponding date

        for file in date_files:
            current_depth = load_depth(file) # Current depth data
            current_timestamp = parse_timestamp_from_filename(file) # Date and time extracted from the file name

            # Subtract one day from the current timestamp
            previous_day_timestamp = current_timestamp - timedelta(days=1)
            '''
            previous_day_timestamp = current_timestamp.replace(
                day=current_timestamp.day
            ) - timedelta(days=1)
            '''

            previous_date_key = previous_day_timestamp.strftime("%Y-%m-%d") # Convert the previous date into a string
            time_key = current_timestamp.strftime("%H:%M:%S")

            previous_day_file = file_map.get((previous_date_key, time_key)) # Find the file from the same time on the previous day in file_map

            if previous_day_file is None:
                # If there is no file from the same time on the previous day, compare the current file with itself
                baseline_depth = current_depth
            else:
                # Compare with the file from the same time on the previous day
                baseline_depth = load_depth(previous_day_file)

            image = create_change_overlay(
                baseline_depth=baseline_depth,
                current_depth=current_depth,
                roi_mask=roi_mask,
                min_depth=min_depth,
                max_depth=max_depth
            )

            image = cv2.resize(image, (CELL_WIDTH, CELL_HEIGHT))

            label = current_timestamp.strftime("%H:%M") # Add a time label to the top-left corner of the image
            add_text(image, label, 10, 20, size=0.5, color=(0, 0, 0), background=True)

            cells.append(image) # Add the completed image for each time to the cells list

        row = np.hstack(cells) # Horizontally concatenate the images for each time on the corresponding date

        label_area = np.full((CELL_HEIGHT, DATE_LABEL_WIDTH, 3), 255, dtype=np.uint8)
        add_text(label_area, date, 10, 35, 0.45)

        row = np.hstack([label_area, row]) # Horizontally concatenate the date label area and the images for each time
        rows.append(row) # Add this row to the rows list

    max_width = max(row.shape[1] for row in rows) # The number of captures may differ by date

    padded_rows = [] # List for adding white padding to shorter rows

    # Add white space to the right side of shorter rows so that all rows have the same width
    # This is necessary because all rows must have the same width to concatenate them vertically using np.vstack()
    for row in rows:
        h, w = row.shape[:2]

        if w < max_width:
            padding = np.full((h, max_width - w, 3), 255, dtype=np.uint8)
            row = np.hstack([row, padding])

        padded_rows.append(row)

    final_image = np.vstack(padded_rows) # Vertically concatenate the rows for each date

    legend = create_legend(final_image.shape[1])
    final_image = np.vstack([legend, final_image]) # Add the title area at the top

    cv2.imwrite(str(OUTPUT_IMAGE), final_image) # Save the final image

    print("Depth change sheet with ROI created successfully.")
    print(f"Saved result image: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    main()