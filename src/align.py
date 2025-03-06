import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import argparse
import os

def compute_match_count(frame1, frame2, sift, flann, ratio_thresh=0.7):
    """
    Compute the number of good SIFT matches between two frames.
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None:
        return 0
    
    raw_matches = flann.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in raw_matches if m.distance < ratio_thresh * n.distance]
    return len(good_matches)


def find_last_good_time(video_path, ref_frame, start_time, end_time, step, delta_threshold, sift, flann):
    """
    Scan video from start_time to end_time (in seconds), comparing each frame with the ref_frame.
    Returns the last time (in seconds) where the drop in good matches from the previous frame is NOT over delta_threshold.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    last_good = start_time
    prev_match_count = None

    print(f"\nScanning {video_path} from {start_time}s to {end_time}s ...")
    t = start_time
    while t <= end_time:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret:
            break
        
        current_match_count = compute_match_count(frame, ref_frame, sift, flann)
        print(f"Time {t:.1f}s: {current_match_count} good matches")
        
        if prev_match_count is not None:
            # If the drop compared to the previous frame is over the threshold, break.
            if (prev_match_count - current_match_count) > delta_threshold:
                break
        # Update last good time and previous count
        last_good = t
        prev_match_count = current_match_count
        t += step
    cap.release()
    return last_good


def align_pointers(video1_path, video2_path, delta_threshold=25, step=0.5, window=10):
    """
    Two-phase alignment using a drop threshold:
      Phase 1: Use video2's frame at time 0 as a reference and scan video1 (0..window seconds)
               to determine its trim time based on a drop of over delta_threshold matches.
      Phase 2: Use video1's frame at the determined trim time as reference and scan video2 (0..window seconds)
               to determine its trim time using the same criteria.
    """
    # Create SIFT detector and FLANN matcher
    sift = cv2.SIFT_create()
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    
    # --- Phase 1: Align Video1 using Video2's initial frame as reference ---
    cap2 = cv2.VideoCapture(video2_path)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, ref_frame2 = cap2.read()
    if not ret:
        print("Error reading Video2 at time 0.")
        return None, None
    cap2.release()
    
    print("Phase 1: Using Video2's frame at 0s as reference to scan Video1.")
    video1_trim = find_last_good_time(video1_path, ref_frame2, start_time=0, end_time=window, 
                                      step=step, delta_threshold=delta_threshold, sift=sift, flann=flann)
    print(f"Determined Video1 trim time: {video1_trim}s")
    
    # --- Phase 2: Align Video2 using Video1's frame at the trim time as reference ---
    cap1 = cv2.VideoCapture(video1_path)
    fps1 = cap1.get(cv2.CAP_PROP_FPS)
    cap1.set(cv2.CAP_PROP_POS_FRAMES, int(video1_trim * fps1))
    ret, ref_frame1 = cap1.read()
    if not ret:
        print("Error reading Video1 at the determined trim time.")
        return video1_trim, None
    cap1.release()
    
    print("Phase 2: Using Video1's frame at the trim time as reference to scan Video2.")
    video2_trim = find_last_good_time(video2_path, ref_frame1, start_time=0, end_time=window, 
                                      step=step, delta_threshold=delta_threshold, sift=sift, flann=flann)
    print(f"Determined Video2 trim time: {video2_trim}s")
    
    return video1_trim, video2_trim


def trim_videos(video1_path, video2_path, trim1, trim2, output_dir):
    clip1 = VideoFileClip(video1_path).subclip(trim1, None)
    clip2 = VideoFileClip(video2_path).subclip(trim2, None)
    
    output_path1 = os.path.join(output_dir, "aligned_video1.mp4")
    output_path2 = os.path.join(output_dir, "aligned_video2.mp4")
    
    print(f"Writing trimmed Video1 to {output_path1}")
    clip1.write_videofile(output_path1)
    print(f"Writing trimmed Video2 to {output_path2}")
    clip2.write_videofile(output_path2)


def align_videos(video1_path, video2_path, output_dir="aligned"):
    # Determine trim times using our two-phase alignment
    video1_trim, video2_trim = align_pointers(video1_path, video2_path)
    
    if video1_trim is None or video2_trim is None:
        print("Failed to determine proper trim times.")
        return
    
    print(f"\nFinal determined trim points:\n  Video1: {video1_trim}s\n  Video2: {video2_trim}s")
    trim_videos(video1_path, video2_path, video1_trim, video2_trim, output_dir)
    print(f"Aligned videos saved to directory: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Align two videos using SIFT')
    parser.add_argument('video1', help='Path to the first video')
    parser.add_argument('video2', help='Path to the second video')
    parser.add_argument("-o", "--output", default="aligned", help="Output directory")

    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True) # Create output directory if needed

    align_videos(args.video1, args.video2, args.output)
    