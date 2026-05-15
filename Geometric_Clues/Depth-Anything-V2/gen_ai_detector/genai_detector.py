from process_video import process_video
from depth_gen import make_depthmaps
from predict import predict_video_folder
import shutil
import os

# Process video and generate depth maps
process_video("10.mp4",5,4,"output")
make_depthmaps("output/frames","output/depthmaps")

# Make prediction
pred,prob=predict_video_folder("../trained_detection_models/best_r3d18_depthmaps_full.pt","output/depthmaps")
print(f"\nPrediction: {'Gen AI' if pred == 1 else 'Real'}")
print(f"Confidence: Real={prob[0]:.2%}, Gen AI={prob[1]:.2%}")

# Clean up output folder after prediction
print("\n   Cleaning up output folder...")
if os.path.exists("output/frames"):
    shutil.rmtree("output/frames")
    os.makedirs("output/frames")
    print("✓ Cleared output/frames")

if os.path.exists("output/depthmaps"):
    shutil.rmtree("output/depthmaps")
    os.makedirs("output/depthmaps")
    print("✓ Cleared output/depthmaps")

print("   Cleanup complete! Ready for next video.")




