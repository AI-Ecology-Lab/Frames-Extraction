import cv2
import datetime
from PIL import Image
def extract_sharpest_frame(video_path):
    """Extracts the sharpest frame from a video file using Laplacian variance."""
    max_variance = 0
    sharpest_frame = None
    video = cv2.VideoCapture(video_path)
    
    while True:
        success, frame = video.read()
        if not success:
            break
            
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Calculate Laplacian variance
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        
        # Update if this frame is sharper
        if variance > max_variance:
            max_variance = variance
            sharpest_frame = frame.copy()
    
    video.release()
    return sharpest_frame

def resize_frame(frame, width, height, interpolation=cv2.INTER_CUBIC):
    """Resizes a frame to the specified width and height using bicubic interpolation."""
    return cv2.resize(frame, (width, height), interpolation=interpolation)

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
