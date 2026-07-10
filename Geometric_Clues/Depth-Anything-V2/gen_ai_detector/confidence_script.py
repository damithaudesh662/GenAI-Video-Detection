# from pathlib import Path
# from predict import predict_video_folder
# import shutil
# import os
# import statistics

# # ---------------- CONFIG ----------------
# MODEL_PATH = "best_r3d18_depthmaps_full.pt"
# BASE_PATH_GEN_AI = Path(
#     r"C:\Users\Admin\OneDrive\Desktop\my acadamics\sem 7\research\Git\GenAIVideoDetection\Geometric_Clues\Depth-Anything-V2\dataset\depthmaps\gen_ai"
# )

# TOTAL_VIDEOS = 500
# # ---------------------------------------

# TP = 0  # correctly predicted AI
# FN = 0  # predicted REAL but actually AI

# correct_confidences = []
# incorrect_confidences = []

# for i in range(1, TOTAL_VIDEOS + 1):
#     folder = BASE_PATH_GEN_AI / str(i)

#     pred, conf = predict_video_folder(MODEL_PATH, str(folder))
    

#     # convert numpy confidence to python float
#     conf = float(conf[1]) * 100


#     # Cleanup
#     if os.path.exists("output/frames"):
#         shutil.rmtree("output/frames")
#         os.makedirs("output/frames")

#     if os.path.exists("output/depthmaps"):
#         shutil.rmtree("output/depthmaps")
#         os.makedirs("output/depthmaps")

#     # Metrics
#     if pred == 1:
#         TP += 1
#         correct_confidences.append(conf)
#     else:
#         FN += 1
#         incorrect_confidences.append(conf)

# # ---------------- STAT FUNCTIONS ----------------
# def stats(conf_list):
#     return {
#         "mean": statistics.mean(conf_list),
#         "median": statistics.median(conf_list),
#         "min": min(conf_list),
#         "max": max(conf_list),
#     }

# correct_stats = stats(correct_confidences) if correct_confidences else None
# incorrect_stats = stats(incorrect_confidences) if incorrect_confidences else None

# # ---------------- RESULTS ----------------
# print("\n========== RESULTS (GEN-AI VIDEOS) ==========")
# print(f"Total videos tested : {TOTAL_VIDEOS}")
# print(f"Correct (Detected as AI)   : {TP}")
# print(f"Incorrect (Detected REAL)  : {FN}")

# if correct_stats:
#     print("\nCorrect Predictions:")
#     print(f"Mean confidence:    {correct_stats['mean']:.2f}%")
#     print(f"Median confidence:  {correct_stats['median']:.2f}%")
#     print(f"Min confidence:     {correct_stats['min']:.2f}%")
#     print(f"Max confidence:     {correct_stats['max']:.2f}%")

# if incorrect_stats:
#     print("\nIncorrect Predictions:")
#     print(f"Mean confidence:    {incorrect_stats['mean']:.2f}%")
#     print(f"Median confidence:  {incorrect_stats['median']:.2f}%")
#     print(f"Min confidence:     {incorrect_stats['min']:.2f}%")
#     print(f"Max confidence:     {incorrect_stats['max']:.2f}%")

# print("============================================")



from pathlib import Path
from predict import predict_video_folder
import shutil
import os
import statistics

# ---------------- CONFIG ----------------
MODEL_PATH = "best_r3d18_depthmaps_full.pt"
BASE_PATH_REAL = Path(
    r"C:\Users\Admin\OneDrive\Desktop\my acadamics\sem 7\research\Git\GenAIVideoDetection\Geometric_Clues\Depth-Anything-V2\gen_ai_detector\depthmaps\real_extra_4fps_frames"
)

TOTAL_VIDEOS = 500
# ---------------------------------------

FP = 0  # predicted AI, actually REAL
TN = 0  # predicted REAL, actually REAL

correct_confidences = []    # TN
incorrect_confidences = []  # FP

for i in range(1000, TOTAL_VIDEOS + 1001):
    folder = BASE_PATH_REAL / str(i)
    print(folder)

    pred, conf = predict_video_folder(MODEL_PATH, str(folder))

    # convert numpy confidence to python float (%)
    conf = float(conf[0]) * 100

    # -------- Cleanup --------
    if os.path.exists("output/frames"):
        shutil.rmtree("output/frames")
        os.makedirs("output/frames")

    if os.path.exists("output/depthmaps"):
        shutil.rmtree("output/depthmaps")
        os.makedirs("output/depthmaps")

    # -------- Evaluation --------
    if pred == 0:
        TN += 1
        correct_confidences.append(conf)
    else:
        FP += 1
        incorrect_confidences.append(conf)

# ---------------- STAT FUNCTION ----------------
def stats(conf_list):
    return {
        "mean": statistics.mean(conf_list),
        "median": statistics.median(conf_list),
        "min": min(conf_list),
        "max": max(conf_list),
    }

correct_stats = stats(correct_confidences) if correct_confidences else None
incorrect_stats = stats(incorrect_confidences) if incorrect_confidences else None

# ---------------- METRICS ----------------
accuracy = TN / TOTAL_VIDEOS
false_positive_rate = FP / TOTAL_VIDEOS

# ---------------- RESULTS ----------------
print("\n========== RESULTS (REAL VIDEOS) ==========")
print(f"Total videos tested : {TOTAL_VIDEOS}")
print(f"Correct (Detected REAL) : {TN}")
print(f"Incorrect (Detected AI) : {FP}")

print("\nAccuracy            :", round(accuracy, 4))
print("False Positive Rate :", round(false_positive_rate, 4))

if correct_stats:
    print("\nCorrect Predictions (REAL → REAL):")
    print(f"Mean confidence:    {correct_stats['mean']:.2f}%")
    print(f"Median confidence:  {correct_stats['median']:.2f}%")
    print(f"Min confidence:     {correct_stats['min']:.2f}%")
    print(f"Max confidence:     {correct_stats['max']:.2f}%")

if incorrect_stats:
    print("\nIncorrect Predictions (REAL → AI):")
    print(f"Mean confidence:    {incorrect_stats['mean']:.2f}%")
    print(f"Median confidence:  {incorrect_stats['median']:.2f}%")
    print(f"Min confidence:     {incorrect_stats['min']:.2f}%")
    print(f"Max confidence:     {incorrect_stats['max']:.2f}%")

print("===========================================")
