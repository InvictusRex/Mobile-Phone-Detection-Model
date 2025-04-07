import torch

print("CUDA Version:", torch.version.cuda)
print("cuDNN Version:", torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "None")
print("PyTorch Built With CUDA:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
