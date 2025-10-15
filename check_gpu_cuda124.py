
# 当前PyTorch稳定版（截至2025年5月）仅支持到CUDA 12.4，直接使用cu125索引会报错，可先使用CUDA 12.4，向下兼容
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# pip install transformers


import torch
# 检查cuda是否可用
print(torch.cuda.is_available())  # 应返回True
print(torch.version.cuda)        # 应输出12.4

import torch
# 检查GPU数量
print(torch.cuda.device_count())

LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(LLM_DEVICE)


# # 定期使用，手动释放内存
# import torch, gc
# gc.collect()
# torch.cuda.empty_cache()  # 如果使用了GPU


