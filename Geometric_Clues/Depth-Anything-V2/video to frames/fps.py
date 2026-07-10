import cv2
import os

# Folder containing your .mp4 files
video_folder = r"C:\Users\Admin\OneDrive\Desktop\my acadamics\sem 7\research\Git\GenAIVideoDetection\Geometric_Clues\Depth-Anything-V2\dataset\train-val\real"

# List all .mp4 files
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]

print(f"Found {len(video_files)} video files.\n")

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_file}")
        continue
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"{video_file}: {fps} FPS")
    
    cap.release()