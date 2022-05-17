import torch

devices_num = torch._C._cuda_getDeviceCount()

print(devices_num)
