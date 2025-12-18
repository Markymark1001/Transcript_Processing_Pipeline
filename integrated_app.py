import streamlit as st
import yt_dlp
import pandas as pd
from datetime import datetime, date, timedelta
import re
import sys
import os
import subprocess
import threading
import streamlit.components.v1 as components
import json
from pathlib import Path
import base64
import random
import time

# Time formatting helper functions
def format_elapsed_time(seconds):
    """
    Convert elapsed seconds to HH:MM:SS format
    
    Args:
        seconds (float): Elapsed time in seconds
        
    Returns:
        str: Formatted time string in HH:MM:SS format
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def calculate_eta(start_time, current_item, total_items):
    """
    Calculate estimated time remaining
    
    Args:
        start_time (float): Process start time in seconds
        current_item (int): Current item number (1-based)
        total_items (int): Total number of items
        
    Returns:
        str: Formatted ETA string in HH:MM:SS format, or empty if not enough data
    """
    if current_item <= 1 or total_items <= 1:
        return ""
    
    elapsed = time.time() - start_time
    avg_time_per_item = elapsed / current_item
    remaining_items = total_items - current_item
    eta_seconds = avg_time_per_item * remaining_items
    
    return format_elapsed_time(eta_seconds)

# Import transcript processing modules
import sys
sys.path.append('.')
from text_processor import TextProcessor
from transcript_processor import TranscriptProcessor
from embedding_generator import EmbeddingGenerator
import config

# Import prescriptive insights modules
from prescriptive_insights import (
    ChunkBuilder, TopicRegistry, TOPICS, RetrievalEngine,
    LLMClient, InsightsRequest, InsightsOrchestrator,
    build_persona_prompt, create_insights_orchestrator
)

# Set page configuration to wide layout
st.set_page_config(layout="wide", page_title="YouTube ID & Transcript Processor")

def parse_relative_date(text):
    """
    Parse relative date strings like "6 days ago", "2 weeks ago", "1 month ago", "1 year ago"
    and return actual date.
    
    Args:
        text (str): Text containing relative date information
        
    Returns:
        datetime.date: The calculated date, or None if parsing fails
    """
    if not text or not isinstance(text, str):
        return None
        
    # Convert to lowercase for easier matching
    text = text.lower()
    
    # Pattern to match "X unit(s) ago" where unit can be day, week, month, year
    pattern = r'(\d+)\s+(day|week|month|year)s?\s+ago'
    match = re.search(pattern, text)
    
    if not match:
        return None
        
    number = int(match.group(1))
    unit = match.group(2)
    
    today = date.today()
    
    if unit == 'day':
        return today - timedelta(days=number)
    elif unit == 'week':
        return today - timedelta(weeks=number)
    elif unit == 'month':
        # Approximate month as 30 days
        return today - timedelta(days=number * 30)
    elif unit == 'year':
        # Approximate year as 365 days
        return today - timedelta(days=number * 365)
    
    return None

class ProgressLogger:
    """Custom logger for yt-dlp to show progress in Streamlit"""
    def __init__(self, status_placeholder):
        self.status_placeholder = status_placeholder
        self.video_count = 0
        
    def debug(self, msg):
        # Only process messages that contain video information
        if "[info]" in msg and "Downloading video" in msg:
            self.video_count += 1
            self.status_placeholder.text(f"Processing video #{self.video_count}: {msg.split('Processing video')[-1].strip()}")
            
    def warning(self, msg):
        pass
        
    def error(self, msg):
        pass

def get_video_list_flat(url, status_placeholder, start_time=None):
    """Get video list using flat extraction for speed"""
    import time
    
    # Update status with elapsed time if start_time is provided
    if start_time:
        elapsed = time.time() - start_time
        status_placeholder.text(f"Getting video list... (Elapsed: {format_elapsed_time(elapsed)})")
    else:
        status_placeholder.text("Getting video list...")
    
    ydl_opts = {
        'extract_flat': True,
        'ignoreerrors': True,
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    
    if info is None:
        return []
    
    # Handle channel URLs by finding videos playlist or direct video entries
    if info.get('_type') == 'playlist' and ('/@' in url or '/channel/' in url or '/user/' in url):
        # This is a channel, find videos playlist or direct video entries
        entries = info.get('entries', [])
        videos_playlist = None
        direct_video_entries = []
        
        # First, check if we have direct video entries (type: url)
        for entry in entries:
            if entry and entry.get('_type') == 'url':
                direct_video_entries.append(entry)
            elif entry and 'Videos' in entry.get('title', ''):
                videos_playlist = entry
                break
        
        # If we found direct video entries, use those
        if direct_video_entries:
            entries = direct_video_entries
        # Otherwise, try to find videos playlist
        elif videos_playlist:
            # Extract videos from videos playlist
            videos_url = videos_playlist.get('webpage_url')
            if videos_url:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    videos_info = ydl.extract_info(videos_url, download=False)
                
                if videos_info and videos_info.get('_type') == 'playlist':
                    entries = videos_info.get('entries', [])
                else:
                    entries = []
            else:
                entries = []
        else:
            entries = []
    elif info.get('_type') == 'playlist':
        # Regular playlist
        entries = info.get('entries', [])
    else:
        # Single video
        entries = [info]
    
    videos = []
    for entry in entries:
        if not entry:
            continue
            
        video_id = entry.get('id', '')
        # For direct video entries, webpage_url might be None, so construct it
        webpage_url = entry.get('webpage_url') or f"https://www.youtube.com/watch?v={video_id}" if video_id else ''
        videos.append({
            'id': video_id,
            'title': entry.get('title', ''),
            'webpage_url': webpage_url,
            'full_url': webpage_url,  # Add full_url to match what the app expects
            'upload_date': None,  # Will be filled in later
            'Selected': True
        })
    
    return videos

def get_video_metadata_batch(video_urls, status_placeholder, batch_size=3, delay=1.0, start_time=None):
    """Get metadata for videos in small batches with delays"""
    import time
    def get_single_video_metadata(url):
        ydl_opts = {
            'ignoreerrors': True,
            'quiet': True,
            'no_warnings': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if info:
                    upload_date = info.get('upload_date')
                    if upload_date:
                        try:
                            # Convert YYYYMMDD to datetime object
                            upload_date = datetime.strptime(upload_date, '%Y%m%d').date()
                        except ValueError:
                            upload_date = None
                    
                    return {
                        'id': info.get('id'),
                        'upload_date': upload_date,
                    }
        except Exception as e:
            print(f"Error getting metadata for {url}: {str(e)}")
            return None
    
    import time
    results = {}
    total_videos = len(video_urls)
    total_batches = (total_videos - 1) // batch_size + 1
    
    # Process videos in small batches with delays
    for i in range(0, total_videos, batch_size):
        batch = video_urls[i:i+batch_size]
        batch_num = i // batch_size + 1
        start_video = i + 1
        end_video = min(i + batch_size, total_videos)
        
        # Include elapsed time and ETA if start_time is provided
        if start_time:
            elapsed = time.time() - start_time
            eta = calculate_eta(start_time, start_video, total_videos)
            eta_text = f" | ETA: {eta}" if eta else ""
            status_placeholder.text(f"Getting upload dates: videos {start_video}-{end_video} of {total_videos} (batch {batch_num}/{total_batches})... (Elapsed: {format_elapsed_time(elapsed)}){eta_text}")
        else:
            status_placeholder.text(f"Getting upload dates: videos {start_video}-{end_video} of {total_videos} (batch {batch_num}/{total_batches})...")
        
        for url in batch:
            result = get_single_video_metadata(url)
            if result:
                results[url] = result
        
        # Add delay between batches (except for the last one)
        if i + batch_size < len(video_urls):
            time.sleep(delay)
    
    return results

def fetch_metadata(url):
    """Fetch metadata from YouTube URL and return as DataFrame"""
    # Create a placeholder for progress updates
    status_placeholder = st.empty()
    status_placeholder.text("Initializing video extraction...")
    
    # Track elapsed time
    import time
    start_time = time.time()
    
    try:
        # Step 1: Get video list quickly with flat extraction
        videos = get_video_list_flat(url, status_placeholder, start_time)
        
        if not videos:
            status_placeholder.text("No videos found.")
            status_placeholder.empty()
            return pd.DataFrame()
        
        # Step 2: Get upload dates using hybrid approach
        video_urls = [v['webpage_url'] for v in videos if v['webpage_url']]
        
        if video_urls:
            status_placeholder.text(f"Found {len(videos)} videos. Getting upload dates...")
            metadata = get_video_metadata_batch(video_urls, status_placeholder, batch_size=3, delay=1.0, start_time=start_time)
            
            # Combine flat data with metadata
            for video in videos:
                url = video['webpage_url']
                video_metadata = metadata.get(url, {})
                video['upload_date'] = video_metadata.get('upload_date')
        
        # Include elapsed time in final message
        elapsed = time.time() - start_time
        status_placeholder.text(f"Finalizing results... (Total elapsed: {format_elapsed_time(elapsed)})")
        
        # Create DataFrame and sort by upload_date (newest first)
        df = pd.DataFrame(videos)
        
        # Ensure upload_date is properly converted to datetime for sorting
        # This handles both string dates and already-converted datetime objects
        df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce').dt.date
        
        # Sort by upload_date in descending order (newest first)
        # Put videos with no upload_date at end
        df = df.sort_values(by='upload_date', ascending=False, na_position='last')
        
        # Clear status placeholder
        status_placeholder.empty()
        
        return df
        
    except Exception as e:
        status_placeholder.text(f"Error: {str(e)}")
        st.error(f"Error fetching videos: {str(e)}")
        status_placeholder.empty()
        return pd.DataFrame()

def convert_srt_to_txt(srt_file_path, txt_file_path):
    """
    Convert SRT subtitle file to plain text by removing timestamps and formatting
    
    Args:
        srt_file_path (str): Path to SRT file
        txt_file_path (str): Path to save TXT file
    """
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

def download_subtitles_for_videos(video_urls, output_dir="subtitles", status_placeholder=None, progress_bar=None, min_delay=0.0, max_delay=0.0):
    """
    Download subtitles for a list of YouTube videos in TXT format
    
    Args:
        video_urls (list): List of YouTube video URLs
        output_dir (str): Directory to save subtitles
        status_placeholder: Streamlit placeholder for status updates
        progress_bar: Streamlit progress bar for visual progress tracking
        min_delay (float): Minimum delay between video downloads in seconds
        max_delay (float): Maximum delay between video downloads in seconds
        
    Returns:
        dict: Results with success count, error count, and detailed error entries
    """
    import glob
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Check for existing downloads to resume
    existing_files = set()
    if os.path.exists(output_dir):
        existing_files = set(Path(f).stem for f in os.listdir(output_dir) if f.endswith('.txt'))
    
    results = {
        'success_count': 0,
        'error_count': 0,
        'errors': [],  # Now contains structured error entries
        'skipped_count': 0
    }
    
    # Track start time for ETA calculations
    download_start_time = time.time()
    
    # Download subtitles for each video
    for i, video_url in enumerate(video_urls, 1):
        # Extract video ID/title to check if already downloaded
        video_id = video_url.split('v=')[-1].split('&')[0] if 'v=' in video_url else video_url.split('/')[-1]
        
        # Check if this video's subtitles already exist
        if video_id in existing_files:
            # Calculate ETA for skip messages
            eta = calculate_eta(download_start_time, i, len(video_urls))
            eta_text = f" | ETA: {eta}" if eta else ""
            
            if status_placeholder:
                status_placeholder.text(f"Skipping video {i}/{len(video_urls)}: {video_url} (already downloaded){eta_text}")
            
            results['skipped_count'] += 1
            
            # Update progress bar if provided
            if progress_bar:
                progress_bar.progress(i / len(video_urls))
            continue
        
        # Calculate ETA for processing messages
        eta = calculate_eta(download_start_time, i, len(video_urls))
        eta_text = f" | ETA: {eta}" if eta else ""
        
        if status_placeholder:
            status_placeholder.text(f"Processing video {i}/{len(video_urls)}: {video_url}{eta_text}")
        
        # Update progress bar if provided
        if progress_bar:
            progress_bar.progress(i / len(video_urls))
        
        # Use yt-dlp to download subtitles as SRT first (most reliable format)
        # Add timeout and retry options for better reliability
        cmd = [
            "yt-dlp",
            "--write-sub",
            "--write-auto-sub",
            "--sub-lang",
            "en",
            "--skip-download",
            "--sub-format",
            "srt/vtt/best",
            "--no-check-certificate",
            "--output",
            f"{output_dir}/%(id)s.%(ext)s",
            "--socket-timeout", "300",  # 5 minute timeout
            "--retries", "3",  # Retry up to 3 times
            "--fragment-retries", "5",  # Retry fragments
            video_url
        ]
        
        try:
            # Download subtitles with full output capture (similar to batch_download.py)
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            # Check if the command failed
            if result.returncode != 0:
                # Create structured error entry
                stderr_output = result.stderr.strip() if result.stderr else "No stderr output"
                stdout_output = result.stdout.strip() if result.stdout else "No stdout output"
                
                # Trim output to reasonable length (max 500 chars each)
                if len(stderr_output) > 500:
                    stderr_output = stderr_output[:500] + "... (truncated)"
                if len(stdout_output) > 500:
                    stdout_output = stdout_output[:500] + "... (truncated)"
                
                error_entry = {
                    'url': video_url,
                    'exit_code': result.returncode,
                    'stdout': stdout_output,
                    'stderr': stderr_output,
                    'command': ' '.join(cmd)
                }
                results['errors'].append(error_entry)
                results['error_count'] += 1
                
                # Calculate ETA for error messages
                eta = calculate_eta(download_start_time, i, len(video_urls))
                eta_text = f" | ETA: {eta}" if eta else ""
                
                if status_placeholder:
                    if result.returncode == 124:  # Timeout
                        status_placeholder.text(f"Download timeout for video {i}. The video may be too long or unavailable.{eta_text}")
                    else:
                        status_placeholder.text(f"Download failed for video {i}: {stderr_output[:100]}...{eta_text}")
            else:
                # Find downloaded subtitle files and convert them to TXT
                srt_files = glob.glob(f"{output_dir}/*.en.srt") + glob.glob(f"{output_dir}/*.srt")
                vtt_files = glob.glob(f"{output_dir}/*.en.vtt") + glob.glob(f"{output_dir}/*.vtt")
                
                # Filter to only recently modified files (within last 60 seconds)
                current_time = time.time()
                recent_srt = [f for f in srt_files if current_time - os.path.getmtime(f) < 60]
                recent_vtt = [f for f in vtt_files if current_time - os.path.getmtime(f) < 60]
                
                if not recent_srt and not recent_vtt:
                    # No subtitles found - create structured error entry
                    error_entry = {
                        'url': video_url,
                        'exit_code': 0,
                        'stdout': result.stdout.strip() if result.stdout else "No stdout output",
                        'stderr': 'No subtitle file found after download',
                        'command': ' '.join(cmd)
                    }
                    results['errors'].append(error_entry)
                    results['error_count'] += 1
                    
                    # Calculate ETA for no subtitles message
                    eta = calculate_eta(download_start_time, i, len(video_urls))
                    eta_text = f" | ETA: {eta}" if eta else ""
                    
                    if status_placeholder:
                        status_placeholder.text(f"No subtitles found for video {i}{eta_text}")
                else:
                    converted = False
                    
                    # Prefer SRT files, fallback to VTT
                    for srt_file in recent_srt:
                        # Create corresponding TXT filename
                        txt_file = srt_file.replace('.srt', '.txt')
                        
                        # Convert SRT to TXT
                        if convert_srt_to_txt(srt_file, txt_file):
                            # Calculate ETA for success message
                            eta = calculate_eta(download_start_time, i, len(video_urls))
                            eta_text = f" | ETA: {eta}" if eta else ""
                            
                            if status_placeholder:
                                status_placeholder.text(f"Successfully downloaded and converted subtitles for video {i} to TXT format{eta_text}")
                            converted = True
                            existing_files.add(video_id)
                        else:
                            # If conversion fails, keep the SRT file
                            # Calculate ETA for conversion failed message
                            eta = calculate_eta(download_start_time, i, len(video_urls))
                            eta_text = f" | ETA: {eta}" if eta else ""
                            
                            if status_placeholder:
                                status_placeholder.text(f"Downloaded subtitles for video {i} in SRT format (conversion to TXT failed){eta_text}")
                    
                    # If no SRT converted, try VTT
                    if not converted:
                        for vtt_file in recent_vtt:
                            # Create corresponding TXT filename
                            txt_file = vtt_file.replace('.vtt', '.txt')
                            
                            # Convert VTT to TXT (reuse SRT conversion as format is similar)
                            if convert_srt_to_txt(vtt_file, txt_file):
                                # Calculate ETA for VTT success message
                                eta = calculate_eta(download_start_time, i, len(video_urls))
                                eta_text = f" | ETA: {eta}" if eta else ""
                                
                                if status_placeholder:
                                    status_placeholder.text(f"Successfully converted VTT subtitles for video {i} to TXT format{eta_text}")
                                converted = True
                                existing_files.add(video_id)
                    
                    if converted:
                        results['success_count'] += 1
                    else:
                        results['error_count'] += 1
            
        except subprocess.TimeoutExpired:
            # Calculate ETA for timeout error
            eta = calculate_eta(download_start_time, i, len(video_urls))
            eta_text = f" | ETA: {eta}" if eta else ""
            
            error_entry = {
                'url': video_url,
                'exit_code': None,
                'stdout': None,
                'stderr': f"Process timed out after 300 seconds",
                'command': ' '.join(cmd)
            }
            results['errors'].append(error_entry)
            results['error_count'] += 1
            if status_placeholder:
                status_placeholder.text(f"Timeout downloading subtitles for video {i}{eta_text}")
                
        except subprocess.CalledProcessError as e:
            # Calculate ETA for called process error
            eta = calculate_eta(download_start_time, i, len(video_urls))
            eta_text = f" | ETA: {eta}" if eta else ""
            
            error_entry = {
                'url': video_url,
                'exit_code': e.returncode,
                'stdout': e.stdout if hasattr(e, 'stdout') else None,
                'stderr': e.stderr if hasattr(e, 'stderr') else str(e),
                'command': ' '.join(cmd)
            }
            results['errors'].append(error_entry)
            results['error_count'] += 1
            if status_placeholder:
                status_placeholder.text(f"Error downloading subtitles for video {i}: {e}{eta_text}")
        
        except Exception as e:
            # Calculate ETA for general exception
            eta = calculate_eta(download_start_time, i, len(video_urls))
            eta_text = f" | ETA: {eta}" if eta else ""
            
            error_entry = {
                'url': video_url,
                'exit_code': None,
                'stdout': None,
                'stderr': str(e),
                'command': ' '.join(cmd)
            }
            results['errors'].append(error_entry)
            results['error_count'] += 1
            if status_placeholder:
                status_placeholder.text(f"Unexpected error for video {i}: {e}{eta_text}")
        
        # Add random delay between videos if min_delay or max_delay is greater than 0
        if min_delay > 0 or max_delay > 0:
            delay_time = random.uniform(min_delay, max_delay)
            if delay_time > 0 and i < len(video_urls):  # Don't delay after last video
                # Calculate ETA for delay message
                eta = calculate_eta(download_start_time, i, len(video_urls))
                eta_text = f" | ETA: {eta}" if eta else ""
                
                if status_placeholder:
                    status_placeholder.text(f"Waiting {delay_time:.1f}s before next download (rate limiting)...{eta_text}")
                time.sleep(delay_time)
    
    return results

def process_transcripts_with_pipeline(transcript_files, status_placeholder=None, progress_bar=None):
    """
    Process transcript files using spaCy pipeline
    
    Args:
        transcript_files (list): List of transcript file paths
        status_placeholder: Streamlit placeholder for status updates
        progress_bar: Streamlit progress bar for visual progress tracking
        
    Returns:
        dict: Results with processed data and statistics
    """
    # Create output directory if it doesn't exist
    output_dir = Path("processed_transcripts")
    output_dir.mkdir(exist_ok=True)
    
    results = {
        'processed_count': 0,
        'error_count': 0,
        'errors': [],
        'processed_files': []
    }
    
    # Initialize transcript processor
    try:
        processor = TranscriptProcessor()
        if status_placeholder:
            status_placeholder.text("Initializing transcript processor...")
    except Exception as e:
        if status_placeholder:
            status_placeholder.text(f"Error initializing processor: {str(e)}")
        return results
    
    # Process each transcript file
    for i, transcript_file in enumerate(transcript_files, 1):
        if status_placeholder:
            status_placeholder.text(f"Processing transcript {i}/{len(transcript_files)}: {Path(transcript_file).name}")
        
        # Update progress bar if provided
        if progress_bar:
            progress_bar.progress(i / len(transcript_files))
        
        try:
            # Read transcript file
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
            
            # Process transcript
            result = processor.process_single_transcript(
                transcript_text, 
                transcript_id=Path(transcript_file).stem
            )
            
            if "error" not in result:
                # Save processed result
                output_file = output_dir / f"{Path(transcript_file).stem}_processed.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                
                results['processed_files'].append({
                    'original_file': transcript_file,
                    'processed_file': str(output_file),
                    'statements_count': result.get('statement_count', 0),
                    'entities_count': result.get('entity_count', 0),
                    'original_length': result.get('original_length', 0),
                    'cleaned_length': result.get('cleaned_length', 0)
                })
                results['processed_count'] += 1
            else:
                results['error_count'] += 1
                error_msg = f"Error processing {transcript_file}: {result.get('error', 'Unknown error')}"
                results['errors'].append(error_msg)
                
        except Exception as e:
            results['error_count'] += 1
            error_msg = f"Error processing {transcript_file}: {str(e)}"
            results['errors'].append(error_msg)
            if status_placeholder:
                status_placeholder.text(error_msg)
    
    # Generate summary report
    if results['processed_files']:
        total_statements = sum(f['statements_count'] for f in results['processed_files'])
        total_entities = sum(f['entities_count'] for f in results['processed_files'])
        total_original_chars = sum(f['original_length'] for f in results['processed_files'])
        total_cleaned_chars = sum(f['cleaned_length'] for f in results['processed_files'])
        
        results['summary'] = {
            'total_transcripts': len(transcript_files),
            'successful_transcripts': results['processed_count'],
            'failed_transcripts': results['error_count'],
            'total_statements': total_statements,
            'total_entities': total_entities,
            'total_original_characters': total_original_chars,
            'total_cleaned_characters': total_cleaned_chars,
            'compression_ratio': total_cleaned_chars / total_original_chars if total_original_chars > 0 else 0
        }
    
    return results

# Main application
st.title("YouTube ID & Transcript Processor")

# Create tabs for different functionalities
tab1, tab2, tab3, tab4 = st.tabs(["📺 YouTube Extractor", "📝 Transcript Processor", "📊 Analysis Results", "💊 Prescriptive Insights"])

# Tab 1: YouTube ID Extractor (based on original app)
with tab1:
    st.header("YouTube Video Extractor")
    
    # Add sidebar with reset button
    with st.sidebar:
        st.header("App Controls")
        
        # Reset App button
        if st.button("🔄 Reset App", help="Clear all data and reset the application"):
            # Clear all session state
            st.session_state.clear()
            
            # Force widget reset by setting a special flag
            st.session_state._reset_triggered = True
            
            # Rerun the app to start fresh
            st.rerun()
        
        st.divider()
    
    # Check if reset was triggered and handle it before any other initialization
    if st.session_state.get('_reset_triggered', False):
        # Clear the reset flag
        del st.session_state._reset_triggered
        
        # Clear any remaining widget state
        for key in list(st.session_state.keys()):
            if key.startswith('_') or key in ['df', 'form_df', 'last_date_range', 'date_range_filter']:
                if key in st.session_state:
                    del st.session_state[key]
        
        # Explicitly clear the URL input field
        st.session_state.url_input = ""
        
        # Reset without showing a success message
        st.rerun()
    
    # Initialize session state only if not reset
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'last_date_range' not in st.session_state:
        st.session_state.last_date_range = None
    
    # URL input with key for state management
    url = st.text_input("Enter YouTube URL", key="url_input")
    
    # Fetch button
    if st.button("Fetch Videos"):
        if url:
            try:
                df = fetch_metadata(url)
                st.session_state.df = df
                
                # Check if DataFrame is not empty and has upload_date column
                if not df.empty and 'upload_date' in df.columns:
                    # Initialize date range to full range when new data is fetched
                    valid_dates = df['upload_date'].dropna()
                    if len(valid_dates) > 0:
                        min_date = valid_dates.min()
                        max_date = valid_dates.max()
                        
                        # Convert to datetime.date if they're pandas Timestamp
                        if hasattr(min_date, 'date'):
                            min_date = min_date.date()
                        if hasattr(max_date, 'date'):
                            max_date = max_date.date()
                        
                        st.session_state.last_date_range = (min_date, max_date)
                    else:
                        st.session_state.last_date_range = None
                else:
                    st.session_state.last_date_range = None
                    
                st.success(f"Found {len(df)} videos")
            except Exception as e:
                st.error(f"Error fetching videos: {str(e)}")
        else:
            st.warning("Please enter a URL")
    
    # Display data if available
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Date range filter
        # Check if upload_date column exists before trying to access it
        if not df.empty and 'upload_date' in df.columns:
            # Filter out None values to get valid dates
            valid_dates = df['upload_date'].dropna()
        else:
            valid_dates = pd.Series(dtype='object')  # Empty series if column doesn't exist
        
        if len(valid_dates) > 0:
            min_date = valid_dates.min()
            max_date = valid_dates.max()
            
            # Ensure we have valid date objects
            if pd.notna(min_date) and pd.notna(max_date):
                # Convert to datetime.date if they're pandas Timestamp
                if hasattr(min_date, 'date'):
                    min_date = min_date.date()
                if hasattr(max_date, 'date'):
                    max_date = max_date.date()
                
                # Display full date range as static info in UK format
                st.info(f"Full date range: {min_date.strftime('%d/%m/%Y')} to {max_date.strftime('%d/%m/%Y')}")
                    
                # Initialize date range from session state if available
                if st.session_state.last_date_range:
                    default_range = st.session_state.last_date_range
                else:
                    default_range = [min_date, max_date]
                    
                date_range = st.date_input(
                    "Filter by upload date range",
                    value=default_range,
                    min_value=min_date,
                    max_value=max_date,
                    format="DD/MM/YYYY",
                    key="date_range_filter"
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                else:
                    start_date = end_date = date_range[0]
                    
                # Check if date range has changed and update selections accordingly
                current_range = (start_date, end_date)
                if st.session_state.last_date_range != current_range:
                    # Update the Selected column based on the new date range
                    df_updated = df.copy()
                    for idx, row in df_updated.iterrows():
                        if pd.notna(row['upload_date']):
                            # Convert to datetime.date if needed
                            upload_date = row['upload_date']
                            if hasattr(upload_date, 'date'):
                                upload_date = upload_date.date()
                            
                            # Check if the video's upload date is within the selected range
                            if start_date <= upload_date <= end_date:
                                df_updated.at[idx, 'Selected'] = True
                            else:
                                df_updated.at[idx, 'Selected'] = False
                    
                    # Re-sort by upload_date in descending order to maintain "Latest to Earliest" order if column exists
                    if 'upload_date' in df_updated.columns:
                        df_updated = df_updated.sort_values(by='upload_date', ascending=False, na_position='last')
                    
                    # Update session state for both the main df and form df
                    st.session_state.df = df_updated
                    st.session_state.form_df = df_updated.copy()
                    st.session_state.last_date_range = current_range
                    st.rerun()
            else:
                start_date = end_date = None
                st.info("No valid upload dates found in data")
        else:
            start_date = end_date = None
            st.info("No valid upload dates found in data")
        
        # Select/Deselect all buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All", key="select_all_btn"):
                # Update both the main df and form df
                df_updated = df.copy()
                df_updated['Selected'] = True
                # Re-sort to maintain "Latest to Earliest" order if upload_date column exists
                if 'upload_date' in df_updated.columns:
                    df_updated = df_updated.sort_values(by='upload_date', ascending=False, na_position='last')
                st.session_state.df = df_updated
                st.session_state.form_df = df_updated.copy()
                st.rerun()
        with col2:
            if st.button("Deselect All", key="deselect_all_btn"):
                # Update both the main df and form df
                df_updated = df.copy()
                df_updated['Selected'] = False
                # Re-sort to maintain "Latest to Earliest" order if upload_date column exists
                if 'upload_date' in df_updated.columns:
                    df_updated = df_updated.sort_values(by='upload_date', ascending=False, na_position='last')
                st.session_state.df = df_updated
                st.session_state.form_df = df_updated.copy()
                st.rerun()
        
        # Reorder columns to put 'Selected' first for better UX
        column_order = ['Selected', 'title', 'upload_date', 'id', 'full_url', 'webpage_url']
        df = df[column_order]
        
        # Display editable dataframe
        st.subheader("Video Selection")
        
        # FORM-BASED APPROACH: Wrap data_editor in a form to prevent scroll reset
        # When widgets are inside a form, Streamlit doesn't rerun on interaction
        # It only reruns when the submit button is pressed
        
        # Initialize session state for form data if needed
        if 'form_df' not in st.session_state:
            st.session_state.form_df = None
        
        # Update form dataframe if main dataframe has changed
        if (st.session_state.form_df is None or
            st.session_state.form_df.shape != df.shape or
            not st.session_state.form_df.drop(columns=['Selected'], errors='ignore').equals(
                df.drop(columns=['Selected'], errors='ignore'))):
            st.session_state.form_df = df.copy()
        
        # Create a form for the data editor
        with st.form("video_selection_form"):
            st.info("📝 **Selection Mode**: Check/uncheck videos below and click 'Apply Selection Changes' when done")
            
            # Use the form dataframe for the data_editor
            edited_df = st.data_editor(
                st.session_state.form_df,
                column_config={
                    "Selected": st.column_config.CheckboxColumn(
                        "Select",
                        help="Select videos to include in the output",
                        default=True,
                        width="small"
                    ),
                    "title": st.column_config.TextColumn("Title", disabled=True, width="large"),
                    "upload_date": st.column_config.DateColumn("Upload Date", format="DD/MM/YYYY", disabled=True, width="medium"),
                    "id": st.column_config.TextColumn("Video ID", disabled=True, width="small"),
                    "full_url": st.column_config.TextColumn("Full URL", disabled=True, width="large"),
                    "webpage_url": st.column_config.TextColumn("Webpage URL", disabled=True, width="large"),
                },
                hide_index=True,
                num_rows="dynamic",
                width='stretch',
                use_container_width=True
            )
            
            # Add submit button for the form
            submitted = st.form_submit_button("Apply Selection Changes")
            
            # Only process selection changes when the form is submitted
            if submitted:
                # Update the form dataframe with the edited values
                st.session_state.form_df = edited_df.copy()
                
                # Also update the main df to keep it in sync
                df['Selected'] = edited_df['Selected']
                st.session_state.df = df
                
                st.success(f"Selection updated: {len(edited_df[edited_df['Selected'] == True])} videos selected")
                st.rerun()
        
        # Filter and output results
        # Use form dataframe for filtering (it contains the latest submitted selections)
        filtered_df = st.session_state.form_df[st.session_state.form_df['Selected'] == True]
        
        # Display results
        st.subheader("Output")
        st.write(f"Selected videos: {len(filtered_df)}")
        
        if len(filtered_df) > 0:
            video_urls = "\n".join(filtered_df['full_url'].tolist())
            st.text_area("YouTube URLs (copy this)", value=video_urls, height=200)
            
            # Add download subtitles section with delay controls
            st.subheader("Download Controls")
            
            # Delay controls in columns
            delay_col1, delay_col2 = st.columns(2)
            with delay_col1:
                min_delay = st.number_input(
                    "Min Delay (seconds)",
                    min_value=0.0,
                    max_value=60.0,
                    value=0.0,
                    step=0.5,
                    help="Minimum delay between video downloads to respect rate limits",
                    key="min_delay_input"
                )
            with delay_col2:
                max_delay = st.number_input(
                    "Max Delay (seconds)",
                    min_value=0.0,
                    max_value=60.0,
                    value=0.0,
                    step=0.5,
                    help="Maximum delay between video downloads (random delay between min and max)",
                    key="max_delay_input"
                )
            
            # Validate delay values
            if max_delay < min_delay:
                st.warning("Max delay should be >= min delay. Using min delay for both.")
                max_delay = min_delay
            
            # Show delay info if enabled
            if min_delay > 0 or max_delay > 0:
                st.info(f"⏱️ Rate limiting enabled: Random delay of {min_delay:.1f}s - {max_delay:.1f}s between downloads")
            
            # Add download subtitles button
            col1, col2 = st.columns(2)
            with col1:
                download_subtitles = st.button("Download Subtitles as TXT", key="download_subtitles_btn")
            
            if download_subtitles:
                # Create a placeholder for status updates
                status_placeholder = st.empty()
                
                # Create a progress bar
                progress_bar = st.progress(0)
                
                # Get the list of selected video URLs
                selected_video_urls = filtered_df['full_url'].tolist()
                
                # Download subtitles with progress bar and delay settings
                results = download_subtitles_for_videos(
                    selected_video_urls,
                    output_dir="subtitles",
                    status_placeholder=status_placeholder,
                    progress_bar=progress_bar,
                    min_delay=min_delay,
                    max_delay=max_delay
                )
                
                # Display results
                status_placeholder.empty()
                progress_bar.empty()
                
                # Summary section
                st.subheader("Download Summary")
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                with summary_col1:
                    if results['success_count'] > 0:
                        st.success(f"✅ Success: {results['success_count']}")
                    else:
                        st.metric("Success", results['success_count'])
                with summary_col2:
                    if results.get('skipped_count', 0) > 0:
                        st.info(f"⏭️ Skipped: {results['skipped_count']}")
                    else:
                        st.metric("Skipped", results.get('skipped_count', 0))
                with summary_col3:
                    if results['error_count'] > 0:
                        st.error(f"❌ Failed: {results['error_count']}")
                    else:
                        st.metric("Failed", results['error_count'])
                
                # Detailed download log expander for errors
                if results['error_count'] > 0:
                    st.warning(f"⚠️ {results['error_count']} video(s) failed. See 'Detailed download log' below for diagnostics.")
                    
                    with st.expander("📋 Detailed download log", expanded=False):
                        st.markdown("### Failed Downloads Diagnostics")
                        st.markdown("---")
                        
                        for idx, error in enumerate(results['errors'], 1):
                            if isinstance(error, dict):
                                # Structured error entry
                                st.markdown(f"**Video {idx}:** `{error.get('url', 'Unknown URL')}`")
                                
                                # Create diagnostics table
                                diag_data = {
                                    "Field": ["Exit Code", "STDOUT (trimmed)", "STDERR (trimmed)"],
                                    "Value": [
                                        str(error.get('exit_code', 'N/A')),
                                        (error.get('stdout', 'N/A') or 'N/A')[:200] + ('...' if error.get('stdout') and len(error.get('stdout', '')) > 200 else ''),
                                        (error.get('stderr', 'N/A') or 'N/A')[:200] + ('...' if error.get('stderr') and len(error.get('stderr', '')) > 200 else '')
                                    ]
                                }
                                st.table(diag_data)
                                
                                # Show full command in a code block
                                with st.expander(f"Full command for video {idx}"):
                                    st.code(error.get('command', 'N/A'), language="bash")
                                
                                # Full output text areas
                                if error.get('stdout'):
                                    st.text_area(f"Full STDOUT (Video {idx})", value=error['stdout'], height=100, key=f"stdout_{idx}")
                                if error.get('stderr'):
                                    st.text_area(f"Full STDERR (Video {idx})", value=error['stderr'], height=100, key=f"stderr_{idx}")
                                
                                st.markdown("---")
                            else:
                                # Legacy string error format
                                st.error(f"**Video {idx}:** {error}")
                                st.markdown("---")
                
                # Show download link for subtitles directory
                if os.path.exists("subtitles") and os.listdir("subtitles"):
                    total_files = len([f for f in os.listdir("subtitles") if f.endswith('.txt')])
                    st.info(f"📁 Subtitle files saved to 'subtitles' directory ({total_files} total TXT files)")
                    
                    # Add resume info
                    if results.get('skipped_count', 0) > 0:
                        st.info("💡 **Resume Feature**: Already downloaded files were automatically skipped. You can safely restart the download process if it gets interrupted.")
        else:
            st.info("No videos selected matching the criteria")

# Tab 2: Transcript Processor
with tab2:
    st.header("Transcript Processing Pipeline")
    
    st.info("""
    **📝 Transcript Processing with spaCy**
    
    This tab allows you to process downloaded transcript files using advanced NLP techniques:
    - Text cleaning and normalization
    - Statement extraction with importance scoring
    - Named entity recognition
    - Sentiment analysis
    - Embedding generation for semantic search
    """)
    
    # Check if subtitles directory exists and has files
    subtitles_dir = Path("subtitles")
    if subtitles_dir.exists():
        transcript_files = list(subtitles_dir.glob("*.txt"))
        
        if transcript_files:
            # Display info panel summarizing discovered transcripts
            st.subheader("Transcript Processing Summary")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Transcripts Found", len(transcript_files))
            with col2:
                st.metric("Output Directory", "processed_transcripts/")
            
            st.info(f"All {len(transcript_files)} discovered transcripts in 'subtitles/' will be processed in one go.")
            
            # Process all transcripts button
            if st.button("Run Transcript Pipeline", key="run_pipeline_btn"):
                # Create placeholders for status and progress
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                # Process all transcripts
                transcript_file_paths = [str(f) for f in transcript_files]
                results = process_transcripts_with_pipeline(
                    transcript_file_paths,
                    status_placeholder=status_placeholder,
                    progress_bar=progress_bar
                )
                
                # Clear placeholders
                status_placeholder.empty()
                progress_bar.empty()
                
                # Display results
                if results['processed_count'] > 0:
                    st.success(f"Successfully processed {results['processed_count']} transcripts")
                    
                    # Display summary statistics
                    if 'summary' in results:
                        summary = results['summary']
                        st.subheader("Processing Summary")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Transcripts", summary['total_transcripts'])
                            st.metric("Successful", summary['successful_transcripts'])
                            st.metric("Failed", summary['failed_transcripts'])
                        
                        with col2:
                            st.metric("Total Statements", summary['total_statements'])
                            st.metric("Total Entities", summary['total_entities'])
                        
                        with col3:
                            st.metric("Original Characters", summary['total_original_characters'])
                            st.metric("Cleaned Characters", summary['total_cleaned_characters'])
                            st.metric("Compression Ratio", f"{summary['compression_ratio']:.2%}")
                
                if results['error_count'] > 0:
                    st.error(f"Failed to process {results['error_count']} transcripts")
                    with st.expander("Error Details"):
                        for error in results['errors']:
                            st.error(error)
                
                # Show download link for processed transcripts
                if os.path.exists("processed_transcripts") and os.listdir("processed_transcripts"):
                    st.info("Processed transcripts have been saved to the 'processed_transcripts' directory")
        else:
            st.warning("No transcript files found in the 'subtitles' directory. Please download subtitles first using the YouTube Extractor tab.")
    else:
        st.warning("No 'subtitles' directory found. Please download subtitles first using the YouTube Extractor tab.")

# Tab 3: Analysis Results
with tab3:
    st.header("Analysis Results")
    
    st.info("""
    **📊 Analysis of Processed Transcripts**
    
    This tab displays analysis results from processed transcripts, including:
    - Entity extraction results
    - Statement importance scores
    - Sentiment analysis
    - Embedding similarities
    """)
    
    # Check if processed_transcripts directory exists and has files
    processed_dir = Path("processed_transcripts")
    if processed_dir.exists():
        processed_files = list(processed_dir.glob("*_processed.json"))
        
        if processed_files:
            st.success(f"Found {len(processed_files)} processed transcript files")
            
            # Select a file to analyze
            selected_file = st.selectbox(
                "Select a processed transcript to analyze:",
                options=processed_files,
                format_func=lambda x: x.name.replace("_processed.json", "")
            )
            
            if selected_file:
                try:
                    # Load and display the processed data
                    with open(selected_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Display basic information
                    st.subheader(f"Analysis for: {selected_file.stem.replace('_processed', '')}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Original Length", data.get('original_length', 0))
                        st.metric("Cleaned Length", data.get('cleaned_length', 0))
                        st.metric("Statements", data.get('statement_count', 0))
                    
                    with col2:
                        st.metric("Entities", data.get('entity_count', 0))
                        if 'embedding_dim' in data:
                            st.metric("Embedding Dimension", data['embedding_dim'])
                    
                    # Display statements
                    if data.get('statements'):
                        st.subheader("Extracted Statements")
                        
                        for i, statement in enumerate(data['statements'][:10], 1):  # Show top 10
                            with st.expander(f"Statement {i}: Importance {statement.get('importance_score', 0):.3f}"):
                                st.write(statement.get('text', ''))
                                
                                # Display entities
                                if statement.get('entities'):
                                    st.write("**Entities:**")
                                    for entity in statement['entities']:
                                        st.write(f"- {entity.get('text', '')} ({entity.get('label', '')})")
                                
                                # Display sentiment
                                if statement.get('sentiment'):
                                    sentiment = statement['sentiment']
                                    st.write(f"**Sentiment:** Polarity: {sentiment.get('polarity', 0):.3f}, Subjectivity: {sentiment.get('subjectivity', 0):.3f}")
                    
                    # Display full text
                    if data.get('cleaned_text'):
                        st.subheader("Cleaned Transcript")
                        st.text_area("Full cleaned text:", value=data['cleaned_text'], height=300)
                    
                    # Download button
                    if st.button(f"Download {selected_file.name}"):
                        with open(selected_file, 'r', encoding='utf-8') as f:
                            st.download_button(
                                label="Download Processed Data",
                                data=f.read(),
                                file_name=selected_file.name,
                                mime="application/json"
                            )
                
                except Exception as e:
                    st.error(f"Error loading processed file: {str(e)}")
        else:
            st.warning("No processed transcript files found. Please process transcripts first using the Transcript Processor tab.")
    else:
        st.warning("No 'processed_transcripts' directory found. Please process transcripts first using the Transcript Processor tab.")

# Tab 4: Prescriptive Insights
with tab4:
    st.header("Prescriptive Insights Generator")
    
    st.info("""
    **💊 Prescriptive Insights with AI**
    
    This tab generates personalized health plans by analyzing processed transcripts:
    - Select topics of interest from available categories
    - Customize persona tone and keywords
    - Generate evidence-based prescriptive plans with citations
    - Download results as Markdown or view evidence chunks
    """)
    
    # Initialize session state for prescriptive insights
    if 'insights_generated' not in st.session_state:
        st.session_state.insights_generated = False
    if 'current_insights' not in st.session_state:
        st.session_state.current_insights = None
    if 'retrieved_chunks' not in st.session_state:
        st.session_state.retrieved_chunks = []
    
    # Check if processed transcripts exist
    processed_dir = Path("processed_transcripts")
    if not processed_dir.exists() or not list(processed_dir.glob("*_processed.json")):
        st.warning("No processed transcripts found. Please process transcripts first using the Transcript Processor tab.")
        st.stop()
    
    # Check if chunk index exists
    chunk_index_path = config.INSIGHTS_CHUNK_DIR / "master_chunks.parquet"
    if not chunk_index_path.exists():
        st.warning("No chunk index found. Please build chunk index first.")
        if st.button("Build Chunk Index", key="build_chunk_index_btn"):
            with st.spinner("Building chunk index from processed transcripts..."):
                try:
                    chunk_builder = ChunkBuilder()
                    stats = chunk_builder.build_chunks()
                    st.success(f"Successfully built chunk index with {stats['total_chunks']} chunks from {stats['processed_transcripts']} transcripts.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error building chunk index: {str(e)}")
    else:
        # Display chunk index statistics
        try:
            retrieval_engine = RetrievalEngine()
            stats = retrieval_engine.get_statistics()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Chunks", stats['total_chunks'])
                st.metric("Source Transcripts", stats['total_transcripts'])
            with col2:
                st.metric("Avg Chunk Length", f"{stats['avg_chunk_length']:.0f} chars")
                if 'chunks_with_embeddings' in stats:
                    st.metric("Chunks with Embeddings", stats['chunks_with_embeddings'])
            with col3:
                st.metric("Topics Available", len(stats['topics_available']))
                if 'embedding_coverage' in stats:
                    st.metric("Embedding Coverage", f"{stats['embedding_coverage']:.1%}")
        except Exception as e:
            st.error(f"Error loading chunk index: {str(e)}")
            st.info("Please try rebuilding the chunk index.")
            if st.button("Rebuild Chunk Index", key="rebuild_chunk_index_btn"):
                with st.spinner("Rebuilding chunk index..."):
                    try:
                        chunk_builder = ChunkBuilder()
                        stats = chunk_builder.build_chunks(force_rebuild=True)
                        st.success(f"Successfully rebuilt chunk index with {stats['total_chunks']} chunks.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error rebuilding chunk index: {str(e)}")
    
    # Check Ollama connection
    try:
        llm_client = LLMClient()
        health = llm_client.health_check()
        
        if health["status"] != "healthy":
            st.error("Ollama server is not running or unhealthy. Please start Ollama and pull a model.")
            st.info("See [OLLAMA_GUIDE.md](OLLAMA_GUIDE.md) for setup instructions.")
            st.stop()
    except Exception as e:
        st.error(f"Error connecting to Ollama: {str(e)}")
        st.info("See [OLLAMA_GUIDE.md](OLLAMA_GUIDE.md) for setup instructions.")
        st.stop()
    
    # Prescriptive Insights UI Controls
    st.subheader("Generate Prescriptive Plan")
    
    # Topic selection
    topic_registry = TopicRegistry()
    available_topics = topic_registry.get_all_topics()
    topic_options = {topic_id: topic.name for topic_id, topic in available_topics.items()}
    
    selected_topics = st.multiselect(
        "Select Topics",
        options=list(topic_options.keys()),
        format_func=lambda x: topic_options[x],
        default=["start_plan", "supplements", "holistic_view"],
        help="Select topics you want to include in the prescriptive plan"
    )
    
    # Persona and customization options
    col1, col2 = st.columns(2)
    with col1:
        persona_toggle = st.toggle(
            "Medical/Empathetic Persona",
            value=True,
            help="Enable medical authority with patient empathy tone"
        )
        evidence_preview = st.toggle(
            "Show Evidence Preview",
            value=False,
            help="Show preview of retrieved chunks before generating plan"
        )
    
    with col2:
        chunk_limit = st.slider(
            "Chunk Limit per Topic",
            min_value=5,
            max_value=50,
            value=15,
            step=5,
            help="Maximum number of chunks to retrieve per topic"
        )
        use_semantic_search = st.toggle(
            "Semantic Search",
            value=True,
            help="Use semantic similarity for better chunk ranking"
        )
    
    # Keyword overrides and patient context
    with st.expander("Advanced Options"):
        keyword_overrides = {}
        
        for topic_id in selected_topics:
            topic = topic_registry.get_topic(topic_id)
            default_keywords = ", ".join(topic.keywords[:3])  # Show first 3 keywords as example
            custom_keywords = st.text_input(
                f"Keywords for {topic.name}",
                value="",
                placeholder=f"Default: {default_keywords}",
                key=f"keywords_{topic_id}",
                help=f"Override default keywords for {topic.name}. Leave empty to use defaults."
            )
            
            if custom_keywords.strip():
                keyword_overrides[topic_id] = [k.strip() for k in custom_keywords.split(",")]
        
        patient_context = st.text_area(
            "Patient Context (Optional)",
            value="",
            placeholder="e.g., 45-year-old female with insulin resistance, looking to start keto diet",
            help="Provide specific patient context for personalized recommendations"
        )
        
        custom_persona_additions = st.text_area(
            "Custom Persona Additions (Optional)",
            value="",
            placeholder="Additional instructions for the AI persona",
            help="Add custom instructions to the AI persona"
        )
    
    # Generate button
    generate_button = st.button(
        "Generate Prescriptive Plan",
        type="primary",
        disabled=not selected_topics,
        help="Generate a prescriptive plan based on selected topics and options"
    )
    
    if generate_button and selected_topics:
        with st.spinner("Retrieving relevant chunks and generating insights..."):
            try:
                # Initialize retrieval engine
                retrieval_engine = RetrievalEngine()
                
                # Retrieve chunks for each selected topic
                all_chunks = []
                for topic_id in selected_topics:
                    keywords = keyword_overrides.get(topic_id)
                    topic_chunks = retrieval_engine.search_by_topic(
                        topic_id=topic_id,
                        keywords=keywords,
                        top_k=chunk_limit,
                        use_semantic=use_semantic_search
                    )
                    all_chunks.extend(topic_chunks)
                
                # Remove duplicates by chunk_id
                seen_ids = set()
                unique_chunks = []
                for chunk in all_chunks:
                    if chunk["chunk_id"] not in seen_ids:
                        seen_ids.add(chunk["chunk_id"])
                        unique_chunks.append(chunk)
                
                # Sort by importance and limit total
                unique_chunks.sort(
                    key=lambda x: x.get("max_importance_score", 0), 
                    reverse=True
                )
                
                total_chunks = unique_chunks[:chunk_limit * len(selected_topics)]
                st.session_state.retrieved_chunks = total_chunks
                
                if evidence_preview:
                    st.subheader("Evidence Preview")
                    st.info(f"Found {len(total_chunks)} relevant chunks. Showing top 10:")
                    
                    for i, chunk in enumerate(total_chunks[:10], 1):
                        with st.expander(f"Chunk {i}: {chunk['chunk_id']} (Score: {chunk.get('max_importance_score', 0):.3f})"):
                            st.write(chunk["text"][:500] + "..." if len(chunk["text"]) > 500 else chunk["text"])
                            st.write(f"**Topics:** {', '.join(chunk.get('matching_topics', []))}")
                            st.write(f"**Source:** {chunk['transcript_id']}")
                
                # Generate insights
                orchestrator = create_insights_orchestrator(
                    retrieval_engine=retrieval_engine,
                    llm_client=llm_client
                )
                
                # Create insights request
                request = InsightsRequest(
                    topics=selected_topics,
                    persona_toggle=persona_toggle,
                    keyword_overrides=keyword_overrides,
                    chunk_limit=chunk_limit,
                    include_supplements="supplements" in selected_topics,
                    patient_context=patient_context,
                    custom_persona_additions=custom_persona_additions,
                    use_semantic_search=use_semantic_search
                )
                
                # Generate insights
                insights_response = orchestrator.generate_insights(request)
                st.session_state.current_insights = insights_response
                st.session_state.insights_generated = True
                
                st.success("Prescriptive plan generated successfully!")
                
            except Exception as e:
                st.error(f"Error generating prescriptive plan: {str(e)}")
                st.info("Please check your Ollama connection and try again.")
    
    # Display generated insights
    if st.session_state.insights_generated and st.session_state.current_insights:
        insights = st.session_state.current_insights
        
        st.subheader("Generated Prescriptive Plan")
        
        # Display sections
        for section in insights.sections:
            st.markdown(f"### {section.title}")
            st.write(section.content)
            
            # Show citations if available
            if section.citations:
                with st.expander(f"Citations for {section.title}"):
                    for citation in section.citations:
                        st.code(f"[{citation}]")
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            # Markdown download
            markdown_content = f"# Prescriptive Health Plan\n\n"
            markdown_content += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            markdown_content += f"**Topics:** {', '.join([topic_registry.get_topic(t).name for t in insights.generation_metadata.get('topics_requested', [])])}\n\n"
            markdown_content += f"**Chunks Used:** {len(insights.chunks_used)}\n\n"
            markdown_content += "---\n\n"
            
            for section in insights.sections:
                markdown_content += f"## {section.title}\n\n"
                markdown_content += f"{section.content}\n\n"
                if section.citations:
                    markdown_content += f"**Citations:** {', '.join([f'[{c}]' for c in section.citations])}\n\n"
            
            b64 = base64.b64encode(markdown_content.encode()).decode()
            href = f'<a href="data:file/markdown;base64,{b64}" download="prescriptive_plan.md">Download Prescriptive Plan (Markdown)</a>'
            st.markdown(href, unsafe_allow_html=True)
        
        with col2:
            # Evidence table
            if st.button("Show Evidence Table"):
                st.subheader("Evidence Chunks")
                
                evidence_data = []
                for chunk_id in insights.chunks_used:
                    chunk = retrieval_engine.get_chunk_by_id(chunk_id)
                    if chunk:
                        evidence_data.append({
                            "Chunk ID": chunk_id,
                            "Transcript": chunk.get("transcript_id", ""),
                            "Topics": ", ".join(chunk.get("matching_topics", [])),
                            "Importance": f"{chunk.get('max_importance_score', 0):.3f}",
                            "Preview": chunk["text"][:100] + "..." if len(chunk["text"]) > 100 else chunk["text"]
                        })
                
                if evidence_data:
                    df = pd.DataFrame(evidence_data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No evidence chunks found.")