import torch
import sys
import os

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from yolov12 import Yolo12

# --- 自定义参数 ---
scales = "m"
batch_size = 2
H = 320
W = 320
# -----------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")
model = Yolo12(scales=scales, num_cls=3).to(device)
img = torch.randn(batch_size, 3, H, W).to(device)

#运行并测算
if device.type == 'cuda':
    torch.cuda.reset_peak_memory_stats()

out = model(img)

if device.type == 'cuda':
    mem = torch.cuda.max_memory_allocated(device) / 1024**2
    print(f"Model:{scales} (batch_size = {batch_size}; 分辨率 {H} * {W})\n显存峰值占用: {mem:.2f} MB")

print(f"Output Shape: {list(out.shape)}")