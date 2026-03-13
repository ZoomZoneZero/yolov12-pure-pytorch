import torch
import sys
import os
import time

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from yolov12 import Yolo12

# --- 自定义测算参数 ---
scales = "l"
batch_size = 1
H = 640
W = 640
warmup_iters = 20  # 预热次数，稳定GPU频率
test_iters = 100   # 测算次数，取平均更准确
# --------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")
if device.type == 'cuda':
    print(f"显卡型号: {torch.cuda.get_device_name(0)}")

model = Yolo12(scales=scales, num_cls=3).to(device)
model.eval()  
img = torch.randn(batch_size, 3, H, W).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

try:
    from thop import profile
    macs, _ = profile(model, inputs=(img, ), verbose=False)
    gflops = (macs * 2) / 1e9
    flops_info = f"{gflops:.2f} GFLOPs"
except ImportError:
    flops_info = "请安装 thop (pip install thop) 以查看计算量"

print("正在预热 GPU...")
with torch.no_grad():
    for _ in range(warmup_iters):
        _ = model(img)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()  
        torch.cuda.reset_peak_memory_stats()  

    start_time = time.time()
    for _ in range(test_iters):
        out = model(img)
    
    if device.type == 'cuda':
        torch.cuda.synchronize() 
    end_time = time.time()

total_time = end_time - start_time
avg_latency_batch = (total_time / test_iters) * 1000  
avg_latency_img = avg_latency_batch / batch_size      
fps = (test_iters * batch_size) / total_time          

if device.type == 'cuda':
    mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
else:
    mem_allocated = 0.0
    mem_reserved = 0.0

if isinstance(out, torch.Tensor):
    out_shapes = list(out.shape)
elif isinstance(out, (list, tuple)):
    out_shapes =[list(o.shape) if isinstance(o, torch.Tensor) else type(o) for o in out]
else:
    out_shapes = type(out)

print("\n" + "="*12 + " 性能测算结果 " + "="*12)
print(f"模型规格       : YOLOv12-{scales}")
print(f"输入尺寸       : Batch={batch_size}, C=3, H={H}, W={W}")
print(f"模型参数量     : {total_params / 1e6:.2f} M (可训练: {trainable_params / 1e6:.2f} M)")
print(f"计算量 (FLOPs) : {flops_info}")
print("-" * 38)
print(f"平均批次耗时   : {avg_latency_batch:.2f} ms")
print(f"单张图片耗时   : {avg_latency_img:.2f} ms")
print(f"模型吞吐量     : {fps:.2f} FPS (imgs/s)")
if device.type == 'cuda':
    print(f"峰值显存分配   : {mem_allocated:.2f} MB (模型真实占用)")
    print(f"峰值显存预留   : {mem_reserved:.2f} MB")
print("-" * 38)
print(f"Output Shape   : {out_shapes}")
print("=" * 38)