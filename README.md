# Object Tracking with YOLO and OpenCV

This project performs object tracking using YOLO for detection and OpenCV for tracking. The script processes a video, detects objects (specifically people), and tracks them using various OpenCV tracking algorithms.

## Prerequisites
Make sure you have Python 3.9 installed.

## Setting Up the Virtual Environment
To create a virtual environment and install dependencies, follow these steps:

### 1. Create a Virtual Environment
```sh
python -m venv venv
```

### 2. Activate the Virtual Environment
- **Windows:**
  ```sh
  venv\Scripts\activate
  ```
- **Linux/Mac:**
  ```sh
  source venv/bin/activate
  ```

### 3. Install Dependencies
```sh
pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the following manually:
```sh
pip install opencv-python numpy ultralytics
```

## Running the Script
To process a video, use the following command:
```sh
python track_objects.py --input input_video.mp4 --output output_video.mp4 --tracker CSRT
```

### Arguments:
- `--input`: Path to the input video.
- `--output`: Path to save the processed video.
- `--tracker`: Type of OpenCV tracker (CSRT, KCF, MOSSE, MIL, etc.). Default is `CSRT`.

## OpenCV Tracker Options
The script supports the following OpenCV trackers:
- **CSRT** (Default, best accuracy)
- **KCF** (Fast, good for small objects)
- **MOSSE** (Very fast, but less accurate)
- **MIL** (Handles object occlusion well)
- **MedianFlow** (Good for slow-moving objects)
- **TLD** (Tracks long-term but slow)
- **Boosting** (Older, not recommended)

## Notes
- Make sure your video file exists and is accessible.
- The YOLO model is loaded from `yolov8n.pt` by default, but you can specify a different model.
- The script processes only `person` class detections (class ID 0).
- The output video will be saved with bounding boxes and tracking information.

## Deactivating the Virtual Environment
After running the script, you can deactivate the virtual environment with:
```sh
deactivate
```

## License
This project is open-source and can be modified as needed.

