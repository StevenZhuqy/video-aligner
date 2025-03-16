# Video Aligner

Align two videos of the same event recorded with different start times. This tool uses feature matching (SIFT or ORB) to detect the exact moment when each video changes content. The detected trim points are ideal for later processing or fine-tuning with a large vision model.

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
python3 src/align.py data/video1.mp4 data/video2.mp4 -o output_directory -f "sift", "orb"
```

- **video1.mp4:** Ideally the one that starts earlier
- **video2.mp4:** Ideally the one that starts later
- **output_directory:** Directory where the aligned videos will be saved.
- **feature:** Feature detector to use: options are sift (default) or orb.

## How It Works

- **Feature Matching:**\
Extracts features from video frames using either SIFT (with a FLANN-based matcher) or ORB (with a BFMatcher using Hamming distance). You can select the desired method using the --feature (or -f) option.

- **Delta Threshold:**\
Scans through a time window with a configurable step and records the highest match count seen so far. If a subsequent frame drops more than the specified delta threshold in match count compared to the highest recorded value, the last frame is marked as the trim point.
