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

# Import transcript processing modules
import sys
sys.path.append('.')
from text_processor import TextProcessor
from transcript_processor import TranscriptProcessor
from embedding_generator import EmbeddingGenerator
import config

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
        status_placeholder.text(f"Getting video list... (Elapsed: {elapsed:.1f}s)")
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
        
        # Include elapsed time if start_time is provided
        if start_time:
            elapsed = time.time() - start_time
            status_placeholder.text(f"Getting upload dates: videos {start_video}-{end_video} of {total_videos} (batch {batch_num}/{total_batches})... (Elapsed: {elapsed:.1f}s)")
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
        status_placeholder.text(f"Finalizing results... (Total elapsed: {elapsed:.1f}s)")
        
        # Create DataFrame and sort by upload_date (newest first)
        df = pd.DataFrame(videos)
        
        # Ensure upload_date is properly converted to datetime for sorting
        # This handles both string dates and already-converted datetime objects
        df['upload_date'] = pd.to_datetime(df['upload_date'], errors='coerce').dt.date
        
        # Sort by upload_date in descending order (newest first)
        # Put videos with no upload_date at the end
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

def download_subtitles_for_videos(video_urls, output_dir="subtitles", status_placeholder=None, progress_bar=None):
    """
    Download subtitles for a list of YouTube videos in TXT format
    
    Args:
        video_urls (list): List of YouTube video URLs
        output_dir (str): Directory to save subtitles
        status_placeholder: Streamlit placeholder for status updates
        progress_bar: Streamlit progress bar for visual progress tracking
        
    Returns:
        dict: Results with success count, error count, and error details
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = {
        'success_count': 0,
        'error_count': 0,
        'errors': []
    }
    
    # Download subtitles for each video
    for i, video_url in enumerate(video_urls, 1):
        if status_placeholder:
            status_placeholder.text(f"Processing video {i}/{len(video_urls)}: {video_url}")
        
        # Update progress bar if provided
        if progress_bar:
            progress_bar.progress(i / len(video_urls))
        
        # Use yt-dlp to download subtitles as SRT first (most reliable format)
        cmd = [
            "yt-dlp",
            "--write-sub",
            "--write-auto-sub",
            "--sub-lang",
            "en",
            "--skip-download",
            "--sub-format",
            "srt",
            "--output",
            f"{output_dir}/%(title)s.%(ext)s",
            video_url
        ]
        
        try:
            # Download subtitles as SRT
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Find downloaded SRT file and convert it to TXT
            import glob
            srt_files = glob.glob(f"{output_dir}/*.en.srt")
            
            for srt_file in srt_files:
                # Create corresponding TXT filename
                txt_file = srt_file.replace('.srt', '.txt')
                
                # Convert SRT to TXT
                if convert_srt_to_txt(srt_file, txt_file):
                    if status_placeholder:
                        status_placeholder.text(f"Successfully downloaded and converted subtitles for video {i} to TXT format")
                else:
                    # If conversion fails, keep the SRT file
                    if status_placeholder:
                        status_placeholder.text(f"Downloaded subtitles for video {i} in SRT format (conversion to TXT failed)")
            
            results['success_count'] += 1
            
        except subprocess.CalledProcessError as e:
            results['error_count'] += 1
            error_msg = f"Error downloading subtitles for video {i}: {e}"
            results['errors'].append(error_msg)
            if status_placeholder:
                status_placeholder.text(error_msg)
    
    return results

def process_transcripts_with_pipeline(transcript_files, status_placeholder=None, progress_bar=None):
    """
    Process transcript files using the spaCy pipeline
    
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
tab1, tab2, tab3 = st.tabs(["📺 YouTube Extractor", "📝 Transcript Processor", "📊 Analysis Results"])

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
                
                # Download subtitles with progress bar
                results = download_subtitles_for_videos(
                    selected_video_urls,
                    output_dir="subtitles",
                    status_placeholder=status_placeholder,
                    progress_bar=progress_bar
                )
                
                # Display results
                status_placeholder.empty()
                progress_bar.empty()
                
                if results['success_count'] > 0:
                    st.success(f"Successfully downloaded subtitles for {results['success_count']} videos as TXT files")
                
                if results['error_count'] > 0:
                    st.error(f"Failed to download subtitles for {results['error_count']} videos")
                    with st.expander("Error Details"):
                        for error in results['errors']:
                            st.error(error)
                
                # Show download link for subtitles directory
                if os.path.exists("subtitles") and os.listdir("subtitles"):
                    st.info("Subtitle files have been saved to the 'subtitles' directory as TXT files")
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
            st.success(f"Found {len(transcript_files)} transcript files in the 'subtitles' directory")
            
            # File selection
            st.subheader("Select Transcript Files to Process")
            
            # Display file selection checkboxes
            selected_files = []
            for file_path in transcript_files:
                if st.checkbox(file_path.name, key=f"file_{file_path.stem}"):
                    selected_files.append(str(file_path))
            
            if selected_files:
                st.write(f"Selected {len(selected_files)} files for processing")
                
                # Process button
                if st.button("Process Selected Transcripts", key="process_transcripts_btn"):
                    # Create placeholders for status and progress
                    status_placeholder = st.empty()
                    progress_bar = st.progress(0)
                    
                    # Process transcripts
                    results = process_transcripts_with_pipeline(
                        selected_files,
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
                st.warning("Please select at least one transcript file to process")
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