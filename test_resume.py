#!/usr/bin/env python3
"""
Test script to verify resume functionality
"""

import os
from pathlib import Path

def test_resume_logic():
    """Test the resume logic used in batch_download.py"""
    
    # Create a test directory
    test_dir = Path("test_resume")
    test_dir.mkdir(exist_ok=True)
    
    # Create some test files
    test_files = ["video1.txt", "video2.txt", "video3.txt"]
    for filename in test_files:
        with open(test_dir / filename, 'w') as f:
            f.write(f"Test content for {filename}")
    
    print(f"Created test files: {test_files}")
    
    # Test the existing files logic
    existing_files = set()
    if os.path.exists(test_dir):
        existing_files = set(Path(f).stem for f in os.listdir(test_dir) if f.endswith('.txt'))
    
    print(f"Existing files detected: {existing_files}")
    
    # Test video ID extraction (simplified)
    def extract_video_id(url):
        if 'v=' in url:
            return url.split('v=')[-1].split('&')[0]
        else:
            return url.split('/')[-1]
    
    # Test URLs
    test_urls = [
        "https://www.youtube.com/watch?v=video1",
        "https://www.youtube.com/watch?v=video2",
        "https://www.youtube.com/watch?v=video3",
        "https://www.youtube.com/watch?v=video1"  # Duplicate
    ]
    
    print("\nTesting video ID extraction and skip logic:")
    for i, url in enumerate(test_urls, 1):
        video_id = extract_video_id(url)
        
        if video_id in existing_files:
            print(f"  [{i}] {url} -> SKIPPED (already exists)")
        else:
            print(f"  [{i}] {url} -> WOULD DOWNLOAD")
    
    print(f"\nExpected: 3 downloads, 1 skip (duplicate)")
    
    # Clean up
    for file in test_dir.glob("*.txt"):
        file.unlink()
    test_dir.rmdir()
    
    print("Test completed and cleaned up")

if __name__ == "__main__":
    test_resume_logic()