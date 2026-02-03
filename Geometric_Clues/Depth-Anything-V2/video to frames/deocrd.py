import decord
from decord import VideoReader, cpu
import os
import cv2  # used for saving frames as images

def video_to_frames(video_path, output_dir):
    # Create output directory if not exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load video
    vr = VideoReader(video_path, ctx=cpu(0))

    # Total number of frames
    num_frames = len(vr)
    print(f"Total frames in video: {num_frames}")

    # Extract and save frames
    for i in range(num_frames):
        frame = vr[i].asnumpy()  # Convert to numpy array
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB → BGR for saving
        frame_path = os.path.join(output_dir, f"frame_{i:05d}.jpg")
        cv2.imwrite(frame_path, frame_bgr)

    print(f"Frames saved to: {output_dir}")


if __name__ == "__main__":
    video_path = "generated-english.mp4"   # replace with your video file
    output_dir = "frames_output"
    video_to_frames(video_path, output_dir)
