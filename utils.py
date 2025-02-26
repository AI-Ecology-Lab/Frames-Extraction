import cv2
import datetime
import numpy as np
from PIL import Image
import os
import concurrent.futures
import torch

# Check for GPU availability
HAS_CUDA = torch.cuda.is_available()
if HAS_CUDA:
    # Set device to use your RTX 3090
    torch.cuda.set_device(0)
    # Get GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
    print(f"Using GPU: {gpu_name} with {gpu_memory:.2f} GB VRAM")

def extract_sharpest_frame(video_path, sample_rate=30, block_size=8, use_gpu=True):
    """
    Extracts the sharpest frame from a video file using Laplacian variance.
    Uses frame sampling and multi-block analysis for efficiency.
    
    Args:
        video_path: Path to the video file
        sample_rate: Only analyze every Nth frame (higher = faster but less accurate)
        block_size: Number of blocks to divide the image into for faster calculation
        use_gpu: Whether to use GPU for computation (currently using PyTorch for CUDA detection only)
        
    Returns:
        Sharpest frame from the video
    """
    # Check if there's a cached frame first
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(video_path)), "frame_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a simple hash of the video path for cache filename
    cache_filename = os.path.join(cache_dir, os.path.basename(video_path).replace('.mp4', '.jpg'))
    
    # If a cached frame exists, return it
    if os.path.exists(cache_filename):
        return cv2.imread(cache_filename)
    
    max_variance = 0
    sharpest_frame = None
    video = cv2.VideoCapture(video_path)
    
    # Get video parameters
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # If very short video, analyze all frames
    if total_frames < 100:
        sample_rate = 1
    
    # Using OpenCV's GPU support if available
    if use_gpu and HAS_CUDA and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        # OpenCV CUDA acceleration
        gpu_frame_count = 0
        while True:
            success, frame = video.read()
            if not success:
                break
                
            # Only process every Nth frame
            if gpu_frame_count % sample_rate != 0:
                gpu_frame_count += 1
                continue
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use batch processing to leverage GPU
            if gray.size > 1024*1024:  # For large images, downscale for faster processing
                gray = cv2.resize(gray, (1024, 1024))
                
            # Focus on center region (often the most important part)
            h, w = gray.shape
            center_gray = gray[h//4:3*h//4, w//4:3*w//4]
            
            # Calculate Laplacian using OpenCV
            gpu_mat = cv2.cuda_GpuMat(center_gray)
            gpu_laplacian = cv2.cuda.createLaplacian(cv2.CV_8UC1, 1)
            laplacian_result = gpu_laplacian.apply(gpu_mat)
            
            # Download result and calculate variance
            laplacian = laplacian_result.download()
            variance = np.var(laplacian)
            
            # Update if this frame is sharper
            if variance > max_variance:
                max_variance = variance
                sharpest_frame = frame.copy()
                
            gpu_frame_count += 1
    else:
        # CPU-based processing
        frame_count = 0
        while True:
            success, frame = video.read()
            if not success:
                break
                
            # Only process every Nth frame
            if frame_count % sample_rate != 0:
                frame_count += 1
                continue
                
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Downsample for faster processing if image is large
            height, width = gray.shape
            if width > 1024 or height > 1024:
                scale_factor = 0.5
                gray = cv2.resize(gray, (int(width * scale_factor), int(height * scale_factor)))
            
            # Focus on center region (often the most important part)
            center_offset_h = gray.shape[0] // 4
            center_offset_w = gray.shape[1] // 4
            center_gray = gray[center_offset_h:-center_offset_h, center_offset_w:-center_offset_w]
            
            # Calculate Laplacian variance
            laplacian = cv2.Laplacian(center_gray, cv2.CV_64F)
            variance = laplacian.var()
            
            # Update if this frame is sharper
            if variance > max_variance:
                max_variance = variance
                sharpest_frame = frame.copy()
            
            frame_count += 1
    
    video.release()
    
    # Cache the result
    if sharpest_frame is not None:
        cv2.imwrite(cache_filename, sharpest_frame)
    
    return sharpest_frame

def resize_frame(frame, width, height, interpolation=cv2.INTER_CUBIC):
    """Resizes a frame to the specified width and height using bicubic interpolation."""
    # Check if resizing is really needed (don't waste time if dimensions are very close)
    h, w = frame.shape[:2]
    if abs(w - width) < 5 and abs(h - height) < 5:
        return frame
        
    return cv2.resize(frame, (width, height), interpolation=interpolation)

def process_video_file(video_file, output_dir, output_raw=True, resize_option=True, width=1024, height=1024, use_gpu=True):
    """Process a single video file - extract frame, resize, and save it."""
    try:
        # Extract filename without extension
        filename_without_extension = os.path.splitext(os.path.basename(video_file))[0]
        
        # Output path for the frame (raw and resized)
        raw_frame_path = os.path.join(output_dir, "raw", filename_without_extension + ".jpg")
        resized_frame_path = os.path.join(output_dir, "resized", filename_without_extension + ".jpg")
        
        # Create subdirectories if needed
        os.makedirs(os.path.join(output_dir, "raw"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "resized"), exist_ok=True)
        
        # Skip if output already exists
        if (output_raw and os.path.exists(raw_frame_path)) or (resize_option and os.path.exists(resized_frame_path)):
            return f"Skipped existing: {os.path.basename(video_file)}"
            
        # Extract frame
        frame = extract_sharpest_frame(video_file, use_gpu=use_gpu)
        if frame is not None:
            # Save raw frame if requested
            if output_raw:
                # Convert frame to PIL Image and save
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img.save(raw_frame_path)
            
            # Resize and save frame if requested
            if resize_option:
                resized_frame = resize_frame(frame, width, height)
                img_resized = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
                img_resized.save(resized_frame_path)
                
            return f"Processed: {os.path.basename(video_file)}"
        else:
            return f"Failed to extract frame: {os.path.basename(video_file)}"

    except Exception as e:
        return f"Error processing {os.path.basename(video_file)}: {str(e)}"

def process_videos_parallel(video_files, output_dir, output_raw=True, resize_option=True, max_workers=4, use_gpu=True):
    """Process multiple video files in parallel."""
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_video_file, video, output_dir, output_raw, resize_option, 1024, 1024, use_gpu): 
                  video for video in video_files}
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
    return results

def run_inference(model, frame):
    """Runs YOLOv11 inference on a frame using ultralytics."""
    # Convert the frame to PIL Image
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)
    
    # Perform inference
    results = model(pil_image)
    return results

def parse_filename(filename):
    """Parses the filename to extract the timestamp."""
    try:
        # DRAGONFISHSUBC13113_20201009T160014.000Z.mp4
        parts = filename.split('_')
        datetime_str = parts[1].split('.')[0]  # Remove .000Z
        datetime_object = datetime.datetime.strptime(datetime_str, '%Y%m%dT%H%M%S')
        return datetime_object.strftime('%m/%d/%Y %H:%M')
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None
