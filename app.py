import streamlit as st
import os
import glob
import datetime
import cv2
import torch
import pandas as pd
from PIL import Image
from utils import extract_sharpest_frame, resize_frame, run_inference, parse_filename
from ultralytics import YOLO

# Streamlit app title
st.title("Video and Image Processing App")

# Directory selection
video_dir = st.text_input("Enter directory containing video files:")

if video_dir:
    # Check if the directory exists
    if not os.path.exists(video_dir):
        st.error("Directory does not exist.")
    else:
        # Load YOLOv11 model
        try:
            model = YOLO('https://huggingface.co/atticus-carter/YOLOV11_ONC_Mothra/resolve/main/weights/best.pt?download=true')
            st.success("YOLOv11 model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading YOLOv11 model: {e}")
            model = None

        # Options
        extract_option = st.checkbox("Extract Frames", value=True)
        resize_option = st.checkbox("Resize Frames to 1024x1024", value=True)
        run_inference_option = st.checkbox("Run YOLOv11 Inference", value=True)
        create_csv_option = st.checkbox("Create CSV with Timestamps and Inference Results", value=True)

        # Process button
        if st.button("Process"):
            if model is not None:
                # Get all video files in the directory
                video_files = glob.glob(os.path.join(video_dir, "*.mp4"))  # Assuming .mp4 files

                if not video_files:
                    st.warning("No video files found in the directory.")
                else:
                    all_results = []
                    for video_file in video_files:
                        try:
                            # Extract filename without extension
                            filename_without_extension = os.path.splitext(os.path.basename(video_file))[0]
                            
                            # Output directory for images
                            output_dir = os.path.join(video_dir, filename_without_extension + "_frames")
                            os.makedirs(output_dir, exist_ok=True)

                            # Extract frames
                            if extract_option:
                                frames = extract_sharpest_frame(video_file)
                                st.write(f"Extracted {len(frames)} frames from {video_file}")

                                # Save and process frames
                                for i, frame in enumerate(frames):
                                    frame_filename = f"{filename_without_extension}_frame_{i:04d}.jpg"
                                    frame_path = os.path.join(output_dir, frame_filename)
                                    
                                    # Resize frame
                                    if resize_option:
                                        frame = resize_frame(frame, 1024, 1024)
                                    
                                    # Convert frame to PIL Image and save
                                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                                    img.save(frame_path)

                                    # Run inference
                                    if run_inference_option:
                                        results = run_inference(model, frame)
                                        
                                        # Parse filename to get timestamp
                                        timestamp = parse_filename(os.path.basename(video_file))
                                        
                                        # Format inference results
                                        result_row = {
                                            'File': frame_filename,
                                            'Timestamp': timestamp
                                        }
                                        
                                        # Add classname counts to the result row
                                        for class_name in model.names:
                                            class_count = 0
                                            for *xyxy, conf, cls in results.boxes.data:
                                                if model.names[int(cls)] == class_name:
                                                    class_count += 1
                                            result_row[class_name] = class_count
                                        
                                        all_results.append(result_row)

                        except Exception as e:
                            st.error(f"Error processing {video_file}: {e}")

                    # Create CSV
                    if create_csv_option and all_results:
                        df = pd.DataFrame(all_results)
                        csv_path = os.path.join(video_dir, "results.csv")
                        df.to_csv(csv_path, index=False)
                        st.success(f"Results saved to {csv_path}")
            else:
                st.warning("Please load the YOLOv11 model first.")
