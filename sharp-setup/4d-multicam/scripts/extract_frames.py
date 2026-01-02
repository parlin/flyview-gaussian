#!/usr/bin/env python3
"""
Extract synchronized frames from multiple videos.

Usage:
    1. Edit SYNC_OFFSETS with values from sync_audio.py
    2. Edit START_TIME, DURATION, FPS as needed
    3. Run: python extract_frames.py

Expects:
    raw_videos/
        camera1.mp4
        camera2.mp4
        ...

Output:
    dataset/
        cam00/
            frame_00001.jpg
            ...
        cam01/
            ...
"""

import subprocess
import os
import shutil
from pathlib import Path

# ============================================================================
# CONFIGURATION - EDIT THIS SECTION
# ============================================================================

# Sync offsets from sync_audio.py (copy from sync_offsets.txt)
# Format: "filename": offset_in_seconds
SYNC_OFFSETS = {
    "camera1.mp4": 0.0,      # Reference video
    "camera2.mp4": 0.342,    # Starts 0.342s after reference
    "camera3.mp4": -0.156,   # Starts 0.156s before reference
    "camera4.mp4": 0.089,    # Starts 0.089s after reference
    # Add all your cameras here...
}

# Scene selection (in reference video's timeline)
START_TIME = "00:01:00"  # Format: HH:MM:SS or HH:MM:SS.mmm
DURATION = 5             # Seconds to extract
FPS = 30                 # Frames per second (30 recommended)

# Output settings
OUTPUT_DIR = "dataset"
JPEG_QUALITY = 2         # 2-5 (lower = better quality, larger files)

# ============================================================================
# END CONFIGURATION
# ============================================================================


def time_to_seconds(time_str):
    """Convert HH:MM:SS.mmm to seconds."""
    parts = time_str.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    else:
        return float(parts[0])


def seconds_to_time(seconds):
    """Convert seconds to HH:MM:SS.mmm format."""
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def extract_frames(video_path, output_dir, start_time, duration, fps, quality):
    """Extract frames from video using ffmpeg."""
    cmd = [
        'ffmpeg', '-y',
        '-ss', start_time,
        '-i', str(video_path),
        '-t', str(duration),
        '-vf', f'fps={fps}',
        '-q:v', str(quality),
        '-start_number', '1',
        str(output_dir / 'frame_%05d.jpg')
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def get_video_duration(video_path):
    """Get video duration in seconds."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except:
        return None


def main():
    raw_dir = Path("raw_videos")
    output_base = Path(OUTPUT_DIR)
    
    # Validation
    if not raw_dir.exists():
        print(f"Error: {raw_dir} directory not found")
        return
    
    if not SYNC_OFFSETS:
        print("Error: SYNC_OFFSETS is empty. Edit this script first!")
        return
    
    # Clear previous output
    if output_base.exists():
        print(f"Removing existing {output_base}/")
        shutil.rmtree(output_base)
    
    base_start_seconds = time_to_seconds(START_TIME)
    
    print("=" * 60)
    print("FRAME EXTRACTION")
    print("=" * 60)
    print(f"Start time: {START_TIME} ({base_start_seconds:.3f}s)")
    print(f"Duration: {DURATION}s")
    print(f"FPS: {FPS}")
    print(f"Expected frames per camera: {DURATION * FPS}")
    print("=" * 60)
    print()
    
    extracted_cameras = []
    
    for i, (video_name, offset) in enumerate(SYNC_OFFSETS.items()):
        video_path = raw_dir / video_name
        
        if not video_path.exists():
            print(f"⚠️  {video_name} not found, skipping")
            continue
        
        # Calculate adjusted start time for this video
        adjusted_start = base_start_seconds + offset
        
        if adjusted_start < 0:
            print(f"⚠️  {video_name}: adjusted start time is negative ({adjusted_start:.3f}s)")
            adjusted_start = 0
        
        adjusted_start_str = seconds_to_time(adjusted_start)
        
        # Check video duration
        video_duration = get_video_duration(video_path)
        if video_duration and adjusted_start + DURATION > video_duration:
            print(f"⚠️  {video_name}: requested end time exceeds video duration")
        
        # Output directory (cam00, cam01, etc.)
        output_dir = output_base / f"cam{i:02d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📹 {video_name} → {output_dir.name}/")
        print(f"   Offset: {offset:+.3f}s | Extract from: {adjusted_start_str}")
        
        success = extract_frames(
            video_path, output_dir, adjusted_start_str, 
            DURATION, FPS, JPEG_QUALITY
        )
        
        if success:
            frame_count = len(list(output_dir.glob("*.jpg")))
            print(f"   ✅ Extracted {frame_count} frames")
            extracted_cameras.append((output_dir.name, frame_count))
        else:
            print(f"   ❌ Extraction failed")
    
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Output directory: {output_base.absolute()}")
    print(f"Cameras extracted: {len(extracted_cameras)}")
    
    if extracted_cameras:
        frame_counts = [c[1] for c in extracted_cameras]
        if len(set(frame_counts)) > 1:
            print(f"⚠️  Warning: Frame counts differ across cameras!")
            for cam, count in extracted_cameras:
                print(f"   {cam}: {count} frames")
        else:
            print(f"Frames per camera: {frame_counts[0]}")
    
    print()
    print("Next steps:")
    print(f"  1. Verify alignment: open {output_base}/cam*/frame_00001.jpg")
    print(f"  2. Create zip: zip -r dataset.zip {output_base}/")
    print(f"  3. Upload to Google Colab")


if __name__ == "__main__":
    main()
