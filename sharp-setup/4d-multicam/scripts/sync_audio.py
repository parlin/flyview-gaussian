#!/usr/bin/env python3
"""
Audio-based video synchronization.
Finds time offsets between videos using audio cross-correlation.

Usage:
    python sync_audio.py

Expects:
    raw_videos/
        camera1.mp4
        camera2.mp4
        ...

Output:
    sync_offsets.txt
"""

import subprocess
import numpy as np
import os
from pathlib import Path

def extract_audio(video_path, audio_path):
    """Extract audio from video as mono WAV."""
    cmd = [
        'ffmpeg', '-y', '-i', str(video_path),
        '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
        str(audio_path)
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0

def load_audio(path):
    """Load WAV file as numpy array."""
    import wave
    with wave.open(str(path), 'rb') as w:
        frames = w.readframes(w.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
    return audio

def find_offset(audio1, audio2, sample_rate=16000):
    """
    Find offset of audio2 relative to audio1 using cross-correlation.
    Returns offset in seconds (positive = audio2 starts later).
    """
    # Use first 60 seconds max for speed
    max_samples = sample_rate * 60
    a1 = audio1[:max_samples]
    a2 = audio2[:max_samples]
    
    # Normalize
    a1 = a1 / (np.max(np.abs(a1)) + 1e-8)
    a2 = a2 / (np.max(np.abs(a2)) + 1e-8)
    
    # Cross-correlation
    correlation = np.correlate(a1, a2, mode='full')
    offset_samples = np.argmax(correlation) - len(a2) + 1
    offset_seconds = offset_samples / sample_rate
    
    # Confidence (peak height relative to mean)
    confidence = np.max(correlation) / (np.mean(np.abs(correlation)) + 1e-8)
    
    return offset_seconds, confidence

def main():
    raw_dir = Path("raw_videos")
    audio_dir = Path("temp_audio")
    
    if not raw_dir.exists():
        print(f"Error: {raw_dir} directory not found")
        print("Create it and add your MP4 files")
        return
    
    audio_dir.mkdir(exist_ok=True)
    
    # Find all video files
    videos = sorted(list(raw_dir.glob("*.mp4")) + list(raw_dir.glob("*.MP4")) + 
                   list(raw_dir.glob("*.mov")) + list(raw_dir.glob("*.MOV")))
    
    if len(videos) < 2:
        print(f"Error: Need at least 2 videos, found {len(videos)}")
        return
    
    print(f"Found {len(videos)} videos:")
    for v in videos:
        print(f"  - {v.name}")
    print()
    
    # Extract audio from all videos
    print("Extracting audio...")
    audio_files = []
    for video in videos:
        audio_path = audio_dir / f"{video.stem}.wav"
        print(f"  {video.name} → {audio_path.name}")
        if extract_audio(video, audio_path):
            audio_files.append((video, audio_path))
        else:
            print(f"    Warning: Failed to extract audio")
    
    if len(audio_files) < 2:
        print("Error: Could not extract audio from enough videos")
        return
    
    # Load reference audio (first video)
    ref_video, ref_audio_path = audio_files[0]
    ref_audio = load_audio(ref_audio_path)
    
    print(f"\nReference video: {ref_video.name}")
    print("\nCalculating offsets...")
    
    # Find offsets relative to reference
    offsets = [(ref_video.name, 0.0, 1.0)]
    
    for video, audio_path in audio_files[1:]:
        audio = load_audio(audio_path)
        offset, confidence = find_offset(ref_audio, audio)
        offsets.append((video.name, offset, confidence))
        
        conf_indicator = "✓" if confidence > 5 else "?"
        print(f"  {video.name}: {offset:+.3f}s (confidence: {confidence:.1f}) {conf_indicator}")
    
    # Save offsets
    with open("sync_offsets.txt", "w") as f:
        f.write("# Video sync offsets\n")
        f.write("# Offset = seconds relative to reference (first video)\n")
        f.write("# Positive = video starts LATER than reference\n")
        f.write("# Negative = video starts EARLIER than reference\n")
        f.write("#\n")
        f.write("# Copy these values to extract_frames.py SYNC_OFFSETS dict\n\n")
        
        for video_name, offset, confidence in offsets:
            f.write(f'    "{video_name}": {offset:.3f},\n')
    
    print(f"\n✅ Offsets saved to sync_offsets.txt")
    print("\nNext step: Edit extract_frames.py with these offsets")
    
    # Cleanup
    for f in audio_dir.glob("*.wav"):
        f.unlink()
    audio_dir.rmdir()

if __name__ == "__main__":
    main()
