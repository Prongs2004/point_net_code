import torch
print(torch.version.cuda)   # 应该显示 13.0
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
