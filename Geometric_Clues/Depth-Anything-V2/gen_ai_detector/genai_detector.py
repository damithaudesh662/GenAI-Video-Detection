from process_video import process_video
from depth_gen import make_depthmaps
from predict import predict_video_folder

process_video("2.mp4",7,8,"output")
make_depthmaps("output/frames","output/depthmaps")
pred,prob=predict_video_folder("best_r3d18_depthmaps_full.pt","output/depthmaps")
print(pred,prob)




