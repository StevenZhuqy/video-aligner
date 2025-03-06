# Video Aligner

Align two videos of the same event recorded with different start times. This tool uses SIFT matching to detect the exact moment when each video changes content. The detected trim points are ideal for later processing or fine-tuning with a large vision model.

## Install

1. **Install FFmpeg:**

   ```bash
   # Linux
   sudo apt install ffmpeg

   # Mac
   brew install ffmpeg

   # Windows: Download from ffmpeg.org
   ```

2. **Install Python packages:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

```bash
python3 src/align.py data/video1.mp4 data/video2.mp4 -o output_directory
```

- **video1.mp4:** Ideally the one that starts earlier
- **video2.mp4:** Ideally the one that starts later
- **output_directory:** Directory where the aligned videos will be saved.

## How It Works

- **SIFT Matching:**\
Extracts SIFT features from frames and uses a FLANN-based matcher to compare them.

- **Delta Threshold:**\
Scans through a time window with a step (you can change this in the code as you want) and detects a drop (you can change the drop threshold in the code as you want) in matches to determine the trim point.
