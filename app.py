import streamlit as st
import os
import glob
import datetime
import cv2
import torch
import pandas as pd
import time
from PIL import Image
from utils import extract_sharpest_frame, resize_frame, process_videos_parallel
from ultralytics import YOLO
import multiprocessing
import platform
import psutil

# Check GPU
gpu_info = "No GPU detected"
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
    gpu_info = f"GPU: {gpu_name} with {gpu_memory:.2f} GB VRAM"

# System info
system_info = f"OS: {platform.system()} {platform.version()}"
cpu_info = f"CPU: {platform.processor()} ({multiprocessing.cpu_count()} cores)"
ram_info = f"RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB"

# Streamlit app title
st.title("Video Frame Extraction")
st.write("Extract the sharpest frame from videos using GPU acceleration")

# Show system info in expandable section
with st.expander("System Information"):
    st.write(system_info)
    st.write(cpu_info)
    st.write(ram_info)
    st.write(gpu_info)

# Directory selection
video_dir = st.text_input("Enter directory containing video files:", "D:\\Axial\\ONC\\mothra_videos")
output_dir = st.text_input("Enter output directory for frames:", "D:\\Axial\\ONC\\extracted_frames")

if video_dir:
    # Check if the directory exists
    if not os.path.exists(video_dir):
        st.error("Directory does not exist.")
    else:
        # Check if the output directory exists, create if it doesn't
        if not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                st.error(f"Error creating output directory: {e}")
                output_dir = None
        
        if output_dir:
            # Options
            extract_option = st.checkbox("Extract Frames", value=True)
            
            col1, col2 = st.columns(2)
            with col1:
                output_raw = st.checkbox("Output Raw Frames", value=True)
            with col2:
                resize_option = st.checkbox("Output Resized Frames (1024x1024)", value=True)
            
            # GPU option
            use_gpu = st.checkbox("Use GPU Acceleration", value=torch.cuda.is_available())
            
            # Advanced options in an expandable section
            with st.expander("Advanced Options"):
                batch_size = st.number_input("Batch size (videos to process at once)", min_value=1, max_value=256, value=32)
                
                # Performance options
                col1, col2 = st.columns(2)
                with col1:
                    sample_rate = st.slider("Sample rate (analyze every Nth frame)", min_value=1, max_value=60, value=15)
                with col2:
                    num_workers = st.slider("Number of workers", min_value=1, max_value=max(16, multiprocessing.cpu_count()), 
                                           value=min(4, multiprocessing.cpu_count()))
                
                skip_existing = st.checkbox("Skip existing outputs", value=True)
                show_detailed_stats = st.checkbox("Show detailed statistics", value=True)

            # Process button
            if st.button("Process Videos"):
                start_time = time.time()
                
                # Get all video files in the directory
                video_files = glob.glob(os.path.join(video_dir, "*.mp4"))  # Assuming .mp4 files
                
                # If no videos found, also check for other video formats
                if not video_files:
                    for ext in [".avi", ".mov", ".mkv", ".flv", ".wmv"]:
                        video_files.extend(glob.glob(os.path.join(video_dir, f"*{ext}")))

                if not video_files:
                    st.warning(f"No video files found in {video_dir}")
                else:
                    # Display total number of videos found
                    st.write(f"Found {len(video_files)} videos. Processing in batches of {batch_size} using {num_workers} workers.")
                    
                    # Create a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    stats_text = st.empty()
                    
                    # Process in batches
                    total_batches = (len(video_files) + batch_size - 1) // batch_size
                    processed_count = 0
                    skipped_count = 0
                    error_count = 0
                    
                    # Create the subdirectories for outputs
                    if output_raw:
                        os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
                    if resize_option:
                        os.makedirs(os.path.join(output_dir, "resized"), exist_ok=True)
                    
                    for batch_idx in range(total_batches):
                        batch_start_time = time.time()
                        
                        start_idx = batch_idx * batch_size
                        end_idx = min((batch_idx + 1) * batch_size, len(video_files))
                        batch_files = video_files[start_idx:end_idx]
                        
                        status_text.text(f"Processing batch {batch_idx+1}/{total_batches} ({start_idx+1}-{end_idx} of {len(video_files)} videos)")
                        
                        # Filter files to skip existing outputs if requested
                        if skip_existing:
                            filtered_batch_files = []
                            for video_file in batch_files:
                                filename_without_extension = os.path.splitext(os.path.basename(video_file))[0]
                                raw_exists = os.path.exists(os.path.join(output_dir, "raw", filename_without_extension + ".jpg"))
                                resized_exists = os.path.exists(os.path.join(output_dir, "resized", filename_without_extension + ".jpg"))
                                
                                # Skip if all requested outputs already exist
                                if (not output_raw or raw_exists) and (not resize_option or resized_exists):
                                    skipped_count += 1
                                else:
                                    filtered_batch_files.append(video_file)
                            
                            batch_files = filtered_batch_files
                        
                        # Process videos in parallel
                        results = process_videos_parallel(batch_files, output_dir, output_raw, resize_option, num_workers, use_gpu)
                        
                        # Count results
                        for result in results:
                            if "Processed" in result:
                                processed_count += 1
                            elif "Error" in result:
                                error_count += 1
                            elif "Skipped" in result:
                                skipped_count += 1
                        
                        # Update progress
                        current_progress = (batch_idx + 1) / total_batches
                        progress_bar.progress(current_progress)
                        
                        # Calculate and display stats
                        batch_time = time.time() - batch_start_time
                        elapsed_time = time.time() - start_time
                        videos_per_second = (processed_count + skipped_count) / elapsed_time if elapsed_time > 0 else 0
                        estimated_time_remaining = (len(video_files) - (processed_count + skipped_count)) / videos_per_second if videos_per_second > 0 else 0
                        
                        if show_detailed_stats:
                            stats_text.text(
                                f"Stats: {processed_count} processed, {skipped_count} skipped, {error_count} errors | "
                                f"Speed: {videos_per_second:.2f} videos/sec | "
                                f"Batch time: {batch_time:.1f}s | "
                                f"Elapsed: {elapsed_time:.1f}s | "
                                f"Est. remaining: {estimated_time_remaining:.1f}s"
                            )
                        else:
                            stats_text.text(f"Processed: {processed_count} | Skipped: {skipped_count} | Errors: {error_count}")
                    
                    # Complete the progress bar
                    progress_bar.progress(1.0)
                    total_time = time.time() - start_time
                    status_text.text(
                        f"Processing complete! Processed {processed_count} videos, skipped {skipped_count}, "
                        f"had {error_count} errors in {total_time:.1f} seconds. "
                        f"Output saved to {output_dir}"
                    )
