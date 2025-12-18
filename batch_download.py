#!/usr/bin/env python3
"""
Batch YouTube Subtitle Downloader with Resume Capability

This script downloads subtitles for YouTube videos with resume capability.
If the process is interrupted, it will skip already downloaded files when resumed.

Usage:
    python3 batch_download.py --input-file video_urls.txt --output-dir subtitles
    python3 batch_download.py --video-urls "url1,url2,url3" --output-dir subtitles
    python3 batch_download.py --input-file video_urls.txt --verbose-yt  # Enable verbose yt-dlp logging
"""

import argparse
import os
import sys
import subprocess
import time
import random
import re
from pathlib import Path
import yt_dlp

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Batch download YouTube subtitles with resume capability")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--input-file", 
        type=str, 
        help="Text file containing YouTube URLs (one per line)"
    )
    group.add_argument(
        "--video-urls", 
        type=str, 
        help="Comma-separated list of YouTube URLs"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="subtitles",
        help="Directory to save subtitle files (default: subtitles)"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=3,
        help="Number of videos to process in each batch (default: 3)"
    )
    
    parser.add_argument(
        "--delay", 
        type=float, 
        default=1.0,
        help="Delay between batches in seconds (default: 1.0)"
    )
    
    parser.add_argument(
        "--timeout", 
        type=int, 
        default=300,
        help="Timeout for each video download in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retries for each video (default: 3)"
    )
    
    parser.add_argument(
        "--verbose-yt",
        action="store_true",
        help="Enable verbose yt-dlp logging to show detailed command output, stdout, and stderr"
    )
    
    parser.add_argument(
        "--min-delay",
        type=float,
        default=0.0,
        help="Minimum delay between video downloads in seconds (default: 0.0)"
    )
    
    parser.add_argument(
        "--max-delay",
        type=float,
        default=0.0,
        help="Maximum delay between video downloads in seconds (default: 0.0)"
    )
    
    # Parse arguments first
    args = parser.parse_args()
    
    # Validate delay arguments
    if args.min_delay < 0 or args.max_delay < 0:
        parser.error("Delay values must be non-negative")
    
    if args.max_delay < args.min_delay:
        parser.error("max-delay must be greater than or equal to min-delay")
    
    return args

def get_video_urls(args):
    """Get video URLs from arguments"""
    if args.input_file:
        with open(args.input_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    elif args.video_urls:
        return [url.strip() for url in args.video_urls.split(',')]
    else:
        return []

def get_existing_files(output_dir):
    """Get set of already downloaded video IDs"""
    existing_files = set()
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith('.txt'):
                # Extract video ID from filename (handle various formats)
                stem = Path(filename).stem
                # Try to extract ID from common patterns
                if '[' in stem:
                    # Format: "Video Title [video_id]"
                    video_id = stem.split('[')[-1].split(']')[0]
                else:
                    # Use the full stem as identifier
                    video_id = stem
                existing_files.add(video_id)
    return existing_files

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    if 'v=' in url:
        return url.split('v=')[-1].split('&')[0]
    else:
        return url.split('/')[-1]

def get_video_id_from_filename(filename):
    """Extract video ID from various filename formats"""
    stem = Path(filename).stem
    
    # Handle different filename formats
    if '[' in stem and ']' in stem:
        # Format: "Video Title [video_id]"
        try:
            return stem.split('[')[-1].split(']')[0]
        except:
            return stem
    else:
        # Use the full stem as identifier
        return stem

def convert_srt_to_txt(srt_file_path, txt_file_path):
    """Convert SRT subtitle file to plain text"""
    try:
        with open(srt_file_path, 'r', encoding='utf-8') as srt_file:
            content = srt_file.read()
        
        # Remove subtitle numbers and timestamps
        lines = content.split('\n')
        text_lines = []
        
        for line in lines:
            # Skip empty lines, subtitle numbers, and timestamp lines
            if (line.strip() and
                not line.strip().isdigit() and
                not '-->' in line and
                not line.strip().startswith('WEBVTT')):
                text_lines.append(line.strip())
        
        # Join text lines with spaces
        text_content = '\n'.join(text_lines)
        
        # Write to TXT file
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text_content)
        
        # Remove the original SRT file
        os.remove(srt_file_path)
        
        return True
    except Exception as e:
        print(f"Error converting SRT to TXT: {str(e)}")
        return False

def convert_vtt_to_txt(vtt_file_path, txt_file_path):
    """Convert VTT subtitle file to plain text"""
    try:
        with open(vtt_file_path, 'r', encoding='utf-8') as vtt_file:
            content = vtt_file.read()
        
        # Remove VTT header, timestamps, and formatting
        lines = content.split('\n')
        text_lines = []
        
        for line in lines:
            # Skip empty lines, subtitle numbers, timestamp lines, and WEBVTT header
            line_stripped = line.strip()
            if (line_stripped and
                not line_stripped.isdigit() and
                not '-->' in line_stripped and
                not line_stripped.startswith('WEBVTT') and
                not line_stripped.startswith('Kind:') and
                not line_stripped.startswith('Language:')):
                # Remove VTT formatting tags like <c> </c>
                clean_line = re.sub(r'<[^>]+>', '', line_stripped)
                if clean_line:
                    text_lines.append(clean_line)
        
        # Join text lines with newlines
        text_content = '\n'.join(text_lines)
        
        # Write to TXT file
        with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
            txt_file.write(text_content)
        
        # Remove the original VTT file
        os.remove(vtt_file_path)
        
        return True
    except Exception as e:
        print(f"Error converting VTT to TXT: {str(e)}")
        return False

def download_subtitles(video_urls, output_dir, batch_size=3, delay=1.0, timeout=300, retries=3, verbose_yt=False, min_delay=0.0, max_delay=0.0):
    """
    Download subtitles with resume capability
    
    Args:
        video_urls: List of YouTube video URLs
        output_dir: Directory to save subtitle files
        batch_size: Number of videos to process in each batch
        delay: Delay between batches in seconds
        timeout: Timeout for each video download in seconds
        retries: Number of retries for each video
        verbose_yt: If True, enable verbose yt-dlp logging
        min_delay: Minimum delay between individual video downloads in seconds
        max_delay: Maximum delay between individual video downloads in seconds
    """
    import glob
    import time as time_module
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get existing files to skip
    existing_files = get_existing_files(output_dir)
    
    results = {
        'total': len(video_urls),
        'success': 0,
        'skipped': 0,
        'failed': 0,
        'errors': []
    }
    
    print(f"Starting download of {results['total']} videos")
    print(f"Output directory: {output_dir}")
    print(f"Already downloaded: {len(existing_files)} files")
    
    # Process videos in batches
    for i in range(0, len(video_urls), batch_size):
        batch = video_urls[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(video_urls) - 1) // batch_size + 1
        
        print(f"\nProcessing batch {batch_num}/{total_batches} ({len(batch)} videos)")
        
        for j, video_url in enumerate(batch, 1):
            video_idx = i + j
            video_id = extract_video_id(video_url)
            
            # Check if already downloaded (try multiple matching approaches)
            already_downloaded = False
            
            # First try exact match with video ID
            if video_id in existing_files:
                already_downloaded = True
            else:
                # Try to match against filename stems (case-insensitive)
                video_id_lower = video_id.lower()
                for existing_id in existing_files:
                    if existing_id.lower() == video_id_lower:
                        already_downloaded = True
                        break
            
            if already_downloaded:
                print(f"  [{video_idx}/{results['total']}] Skipping {video_url} (already downloaded)")
                results['skipped'] += 1
                continue
            
            print(f"  [{video_idx}/{results['total']}] Downloading {video_url}")
            
            # Use yt-dlp to download subtitles - simplified command matching working version
            cmd = [
                "yt-dlp",
                "--write-sub",
                "--write-auto-sub",
                "--sub-lang", "en",
                "--skip-download",
                "--sub-format", "srt/vtt/best",
                "--no-check-certificate",
                "--output", f"{output_dir}/%(title)s.%(ext)s",
                video_url
            ]
            
            # Print command if verbose mode is enabled
            if verbose_yt:
                print(f"    Executing: {' '.join(cmd)}")
            
            try:
                # Download subtitles with enhanced error capture
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                
                # Print verbose output if enabled
                if verbose_yt:
                    print(f"    Return code: {result.returncode}")
                    if result.stdout:
                        print(f"    STDOUT:\n{result.stdout}")
                    if result.stderr:
                        print(f"    STDERR:\n{result.stderr}")
                
                if result.returncode != 0:
                    # Create detailed error block
                    stderr_output = result.stderr.strip() if result.stderr else "No stderr output"
                    stdout_output = result.stdout.strip() if result.stdout else "No stdout output"
                    
                    # Trim output to reasonable length (max 500 chars each)
                    if len(stderr_output) > 500:
                        stderr_output = stderr_output[:500] + "... (truncated)"
                    if len(stdout_output) > 500:
                        stdout_output = stdout_output[:500] + "... (truncated)"
                    
                    error_msg = f"Error downloading {video_url}"
                    print(f"    ERROR: {error_msg}")
                    
                    # Print structured error output with clear separators
                    print("=" * 60)
                    print(f"YT-DLP ERROR DETAILS")
                    print("=" * 60)
                    print(f"Video URL: {video_url}")
                    print(f"Exit Code: {result.returncode}")
                    print(f"STDERR:\n{stderr_output}")
                    if result.stdout and result.stdout.strip():
                        print(f"STDOUT:\n{stdout_output}")
                    
                    if not verbose_yt:
                        print("\nSuggestion: Re-run with --verbose-yt flag for complete output")
                    
                    print("=" * 60)
                    
                    # Add detailed error info to results
                    detailed_error = {
                        'url': video_url,
                        'exit_code': result.returncode,
                        'stderr': result.stderr,
                        'stdout': result.stdout,
                        'message': error_msg
                    }
                    results['errors'].append(detailed_error)
                    results['failed'] += 1
                else:
                    # Find downloaded subtitle file for THIS video and convert it to TXT
                    
                    # Look for SRT and VTT files matching this video
                    srt_patterns = [
                        f"{output_dir}/*.en.srt",
                        f"{output_dir}/*.srt",
                    ]
                    vtt_patterns = [
                        f"{output_dir}/*.en.vtt",
                        f"{output_dir}/*.vtt",
                    ]
                    
                    srt_files = []
                    vtt_files = []
                    
                    for pattern in srt_patterns:
                        srt_files.extend(glob.glob(pattern))
                    for pattern in vtt_patterns:
                        vtt_files.extend(glob.glob(pattern))
                    
                    # Filter to only recently modified files (within last 60 seconds)
                    current_time = time_module.time()
                    recent_srt = [f for f in srt_files if current_time - os.path.getmtime(f) < 60]
                    recent_vtt = [f for f in vtt_files if current_time - os.path.getmtime(f) < 60]
                    
                    if not recent_srt and not recent_vtt:
                        print(f"    No subtitles found for {video_id}")
                        results['failed'] += 1
                        results['errors'].append({
                            'url': video_url,
                            'exit_code': 0,
                            'stderr': 'No subtitle file found after download',
                            'stdout': result.stdout,
                            'message': f"No subtitles available for {video_url}"
                        })
                    else:
                        converted = False
                        
                        # Prefer SRT files, fallback to VTT
                        for srt_file in recent_srt:
                            # Create corresponding TXT filename
                            txt_file = srt_file.replace('.srt', '.txt')
                            
                            # Convert SRT to TXT
                            if convert_srt_to_txt(srt_file, txt_file):
                                print(f"    Successfully converted to TXT format: {Path(txt_file).name}")
                                converted = True
                                # Add video_id to existing_files so we skip it if we retry
                                existing_files.add(video_id)
                            else:
                                print(f"    Failed to convert to TXT format: {srt_file}")
                        
                        # If no SRT converted, try VTT
                        if not converted:
                            for vtt_file in recent_vtt:
                                # Create corresponding TXT filename
                                txt_file = vtt_file.replace('.vtt', '.txt')
                                
                                # Convert VTT to TXT
                                if convert_vtt_to_txt(vtt_file, txt_file):
                                    print(f"    Successfully converted VTT to TXT format: {Path(txt_file).name}")
                                    converted = True
                                    # Add video_id to existing_files so we skip it if we retry
                                    existing_files.add(video_id)
                                else:
                                    print(f"    Failed to convert VTT to TXT format: {vtt_file}")
                        
                        if converted:
                            results['success'] += 1
                        else:
                            results['failed'] += 1
                    
            except subprocess.TimeoutExpired:
                error_msg = f"Timeout downloading {video_url} (>{timeout}s)"
                print(f"    ERROR: {error_msg}")
                
                # Add detailed error info to results
                detailed_error = {
                    'url': video_url,
                    'exit_code': None,
                    'stderr': f"Process timed out after {timeout} seconds",
                    'stdout': None,
                    'message': error_msg
                }
                results['errors'].append(detailed_error)
                results['failed'] += 1
                
            except Exception as e:
                error_msg = f"Unexpected error for {video_url}: {e}"
                print(f"    ERROR: {error_msg}")
                
                # Add detailed error info to results
                detailed_error = {
                    'url': video_url,
                    'exit_code': None,
                    'stderr': str(e),
                    'stdout': None,
                    'message': error_msg
                }
                results['errors'].append(detailed_error)
                results['failed'] += 1
            
            # Add random delay between videos if min_delay or max_delay is greater than 0
            if min_delay > 0 or max_delay > 0:
                # Calculate delay time
                delay_time = random.uniform(min_delay, max_delay)
                
                # Only log and sleep if delay_time is greater than 0
                if delay_time > 0:
                    print(f"  Sleeping {delay_time:.1f}s to respect rate limits")
                    time.sleep(delay_time)
        
        # Add delay between batches (except for the last one)
        if i + batch_size < len(video_urls):
            print(f"  Waiting {delay}s before next batch...")
            time.sleep(delay)
    
    # Print summary
    print(f"\n=== Download Summary ===")
    print(f"Total videos: {results['total']}")
    print(f"Successfully downloaded: {results['success']}")
    print(f"Skipped (already exists): {results['skipped']}")
    print(f"Failed: {results['failed']}")
    
    if results['errors']:
        print(f"\nErrors:")
        for error in results['errors']:
            if isinstance(error, dict):
                # Handle detailed error format
                print(f"  - {error['message']} (Exit code: {error['exit_code']})")
                if error['stderr'] and error['stderr'].strip():
                    # Show first 100 chars of stderr for summary
                    stderr_preview = error['stderr'][:100] + "..." if len(error['stderr']) > 100 else error['stderr']
                    print(f"    Details: {stderr_preview}")
            else:
                # Handle legacy string format
                print(f"  - {error}")
    
    return results

def main():
    """Main function"""
    args = parse_arguments()
    
    # Get video URLs
    video_urls = get_video_urls(args)
    
    if not video_urls:
        print("No video URLs provided. Use --input-file or --video-urls.")
        sys.exit(1)
    
    # Download subtitles
    download_subtitles(
        video_urls,
        args.output_dir,
        batch_size=args.batch_size,
        delay=args.delay,
        timeout=args.timeout,
        retries=args.retries,
        verbose_yt=args.verbose_yt,
        min_delay=args.min_delay,
        max_delay=args.max_delay
    )

if __name__ == "__main__":
    main()
