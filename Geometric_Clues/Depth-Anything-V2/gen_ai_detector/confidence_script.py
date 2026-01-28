

# from pathlib import Path
# from predict import predict_video_folder

# # ---------------- CONFIG ----------------
# MODEL_PATH = "best_r3d18_depthmaps_full.pt"
# BASE_PATH_GEN_AI = Path(
#     r"C:\Users\Admin\OneDrive\Desktop\my acadamics\sem 7\research\Git\GenAIVideoDetection\Geometric_Clues\Depth-Anything-V2\dataset\depthmaps\gen_ai"
# )

# BASE_PATH_REAL = Path(
#     r"C:\Users\Admin\OneDrive\Desktop\my acadamics\sem 7\research\Git\GenAIVideoDetection\Geometric_Clues\Depth-Anything-V2\dataset\depthmaps\real"
# )

# TOTAL_VIDEOS = 920
# # ---------------------------------------

# TP = 0  # predicted AI, actually AI
# FP = 0  # predicted AI, actually REAL
# TN = 0  # predicted REAL, actually REAL
# FN = 0  # predicted REAL, actually AI (will stay 0)

# for i in range(1, TOTAL_VIDEOS + 1):
#     folder = BASE_PATH_REAL / str(i)
#     pred, _ = predict_video_folder(MODEL_PATH, str(folder))

#     if pred == 1:
#         FP += 1   # REAL video detected as AI
#     else:
#         TN += 1   # REAL video detected as REAL

# # ---------------- METRICS ----------------
# accuracy = (TP + TN) / TOTAL_VIDEOS

# precision = TP / (TP + FP) if (TP + FP) > 0 else 0
# recall = TP / (TP + FN) if (TP + FN) > 0 else 0
# f1_score = (
#     2 * precision * recall / (precision + recall)
#     if (precision + recall) > 0
#     else 0
# )

# # ---------------- RESULTS ----------------
# print("\n========== RESULTS ==========")
# print(f"Total videos tested : {TOTAL_VIDEOS}")
# print(f"Detected as AI      : {FP}")
# print(f"Detected as REAL    : {TN}")

# print("\nAccuracy  :", round(accuracy, 4))
# print("F1 Score  :", round(f1_score, 4))
# print("============================")

from pathlib import Path
from predict import predict_video_folder

# ---------------- CONFIG ----------------
MODEL_PATH = "best_r3d18_depthmaps_full.pt"
BASE_PATH_GEN_AI = Path(
    r"C:\Users\Admin\OneDrive\Desktop\my acadamics\sem 7\research\Git\GenAIVideoDetection\Geometric_Clues\Depth-Anything-V2\dataset\depthmaps\gen_ai"
)
TOTAL_VIDEOS = 920
# ---------------------------------------

TP = 0  # predicted AI, actually AI
FN = 0  # predicted REAL, actually AI
FP = 0
TN = 0

for i in range(1, TOTAL_VIDEOS + 1):
    folder = BASE_PATH_GEN_AI / str(i)
    pred, _ = predict_video_folder(MODEL_PATH, str(folder))

    if pred == 1:
        TP += 1   # correctly detected as AI
    else:
        FN += 1   # missed AI (detected as REAL)

# ---------------- METRICS ----------------
accuracy = (TP + TN) / TOTAL_VIDEOS

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = (
    2 * precision * recall / (precision + recall)
    if (precision + recall) > 0
    else 0
)

# ---------------- RESULTS ----------------
print("\n========== RESULTS (GEN-AI VIDEOS) ==========")
print(f"Total videos tested : {TOTAL_VIDEOS}")
print(f"Detected as AI      : {TP}")
print(f"Detected as REAL    : {FN}")

print("\nAccuracy :", round(accuracy, 4))
print("F1 Score :", round(f1_score, 4))
print("===========================================")
