# yolo-opencv-tracking

Multi-person tracking in video that combines **YOLOv8 person detection** with **OpenCV single-object trackers** (CSRT, KCF, MIL, …).
Each detected person gets its own OpenCV tracker and a numeric ID. Detections are matched to trackers frame by frame using IoU, and the annotated video is written to disk.

## How it works

For every frame of the input video:

1. **Detect**: YOLOv8 runs on the frame and keeps only the `person` class (COCO class 0).
2. **Predict**: every active OpenCV tracker is updated and predicts where its person moved.
3. **Associate**: each detection is compared (IoU) with every tracker prediction.

   | Best IoU | Action |
   |---|---|
   | ≥ `--track_high_thresh` | strong match: the tracker keeps its ID |
   | between `--match_thresh` and `--track_high_thresh` | intermediate match: the tracker keeps its ID |
   | otherwise | a **new** tracker/ID is created if the detection confidence ≥ `--new_track_thresh` |

4. **Clean up**: trackers that go `--track_buffer` frames without a matching detection are removed.
5. **Draw**: boxes and `ID n` labels of the matched detections are drawn and the frame is written to the output video.

## Installation

Requires Python 3.9+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

`opencv-contrib-python` is required because CSRT, KCF and the legacy trackers live in the contrib modules. Do not install it together with `opencv-python`.
The YOLO weights (`yolov8n.pt` by default) are downloaded automatically by Ultralytics on first run.

## Usage

```bash
python track_objects.py --input path/to/video.mp4
```

The output is saved as `<input name>_proc.mp4` next to the input video, unless `--output` is given (a file path or a directory).

```bash
python track_objects.py --input video.mp4 --output results/ --tracker CSRT --model yolov8s.pt
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `--input` | *(required)* | input video |
| `--output` | `<input>_proc.mp4` | output video file or directory |
| `--model` | `yolov8n.pt` | YOLOv8 weights |
| `--tracker` | `KCF` | OpenCV tracker type |
| `--start_frame` | `0` | first frame to process |
| `--frames` | `3000000` | maximum number of frames to process |
| `--track_high_thresh` | `0.7` | IoU for a strong match |
| `--track_low_thresh` | `0.3` | IoU below which a detection is treated as unmatched |
| `--match_thresh` | `0.3` | IoU for an intermediate match |
| `--new_track_thresh` | `0.5` | minimum detection confidence to start a new tracker |
| `--track_buffer` | `30` | frames a tracker may go without a detection before it is removed |

### Tracker options

The choices offered by `--tracker` depend on your OpenCV build; trackers that are not available are hidden automatically.

| Tracker | Notes |
|---|---|
| **CSRT** | most accurate, slower |
| **KCF** | good speed/accuracy trade-off (default) |
| **MIL** | robust to partial occlusion |
| **MOSSE** | very fast, less accurate (legacy) |
| **MedianFlow** | smooth, slow motion (legacy) |
| **TLD** | long-term, slow (legacy) |
| **Boosting** | oldest, mainly for comparison (legacy) |

## Project structure

```
track_objects.py   # detection + tracking pipeline and CLI
requirements.txt
```

## License

[AGPL-3.0](LICENSE), because the project uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
