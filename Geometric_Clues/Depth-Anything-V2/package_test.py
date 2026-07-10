import torch
print(torch.__version__)              # should show +cuXXX, not +cpu [web:121]
print(torch.cuda.is_available())      # should be True [web:128]
print(torch.version.cuda)             # e.g., 12.1 if CUDA 12.1 wheel installed [web:121]
print(torch.cuda.device_count())      # >=1 [web:128]
print(torch.cuda.get_device_name(0))  # GPU name if available [web:128]
