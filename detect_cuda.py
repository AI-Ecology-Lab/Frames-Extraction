import torch
import os
import subprocess
import sys
import platform

def get_cuda_info():
    """Detect CUDA version and display relevant information"""
    print("=== CUDA Detection Tool ===")
    
    # Check if CUDA is available via PyTorch
    if torch.cuda.is_available():
        print(f"PyTorch CUDA available: Yes")
        print(f"CUDA version (PyTorch): {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"CUDA capabilities: {torch.cuda.get_device_capability(0)}")
        
        # Get driver version
        if platform.system() == 'Windows':
            try:
                # For Windows, try using nvidia-smi
                result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.returncode == 0:
                    print(f"NVIDIA driver version: {result.stdout.strip()}")
            except:
                print("Could not determine NVIDIA driver version")
        
        # Recommend cupy version
        cuda_version = torch.version.cuda
        if cuda_version.startswith('11'):
            print("\nRecommended cupy installation:")
            print("pip install cupy-cuda11x")
        elif cuda_version.startswith('12'):
            print("\nRecommended cupy installation:")
            print("pip install cupy-cuda12x")
        else:
            print("\nRecommended cupy installation:")
            print(f"pip install cupy (let pip determine the best version)")
            
    else:
        print("PyTorch CUDA available: No")
        print("\nNo CUDA-capable device detected or CUDA not properly installed")
        print("To use GPU acceleration, please install the appropriate NVIDIA drivers and CUDA toolkit")
    
if __name__ == "__main__":
    get_cuda_info()
