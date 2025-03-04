import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import os


def extract_keyframes(video_path, interval_sec=1):
    pass


def match_frames(des1, des2):
    pass


def find_temporal_offset(vid1_keyframes, vid2_keyframes, fps):
    pass


def align_videos(video1_path, video2_path, output_dir="aligned"):
    pass


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Align two videos using SIFT')
    parser.add_argument('video1', help='Path to the first video')
    parser.add_argument('video2', help='Path to the second video')
    parser.add_argument("-o", "--output", default="aligned", help="Output directory")
    args = parser.parse_args()

    align_videos(args.video1, args.video2, args.output)
    