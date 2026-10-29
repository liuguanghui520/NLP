#!/usr/bin/env python3
import sys

import torch

print("python", sys.version)
print("python_executable", sys.executable)
print("torch", torch.__version__)
print("torch_git", torch.version.git_version)
print("torch_cuda_build", torch.version.cuda)
print("torch_cudnn_build", torch.backends.cudnn.version())
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
print("cuda_bf16_supported", torch.cuda.is_bf16_supported())
if torch.cuda.is_available():
    print("cuda_device_0", torch.cuda.get_device_name(0))
    print("cuda_capability_0", torch.cuda.get_device_capability(0))
    tensor = torch.ones((2, 2), device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    print("bf16_forward_ok", tensor.dtype, tuple(tensor.shape), tensor.sum().item())
