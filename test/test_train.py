import torch
import time
import sys
import os
import torch.optim as optim

# 确保能导入同级目录下的模块
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from yolov12 import Yolo12_train
from loss import Yolo12_Loss

def test_training_performance():
    # --- 自定义参数 ---
    scales = "x"          
    batch_size = 2      
    H, W = 640, 640       
    num_cls = 10          
    num_boxes = 30 
    # -----------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True # 开启自动寻找最优卷积算法
    
    print(f"当前运行设备: {device}")
    print(f"Model:{scales}, batch_size={batch_size}, 单图最高物体数 {num_boxes}, 分辨率 {H} * {W}")

    model = Yolo12_train(scales=scales, num_cls=num_cls).to(device)
    model.train() 
    
    # 动态获取 reg_max 防止维度错乱
    if hasattr(model, 'detect'):
        real_reg_max = model.detect.reg_max
    elif hasattr(model, 'model') and hasattr(model.model[-1], 'reg_max'):
        real_reg_max = model.model[-1].reg_max
    else:
        real_reg_max = 16 
        
    criterion = Yolo12_Loss(num_cls=num_cls, reg_max=real_reg_max).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.937)
    
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    if device.type == 'cuda': torch.cuda.synchronize()
    t_data_start = time.time()

    imgs = torch.randn(batch_size, 3, H, W, device=device)

    targets = torch.zeros(batch_size, num_boxes, 5, device=device)
    targets[:, :, 0] = torch.rand(batch_size, num_boxes, device=device) * 0.6 + 0.2  # cx
    targets[:, :, 1] = torch.rand(batch_size, num_boxes, device=device) * 0.6 + 0.2  # cy
    targets[:, :, 2] = torch.rand(batch_size, num_boxes, device=device) * 0.3 + 0.1  # w
    targets[:, :, 3] = torch.rand(batch_size, num_boxes, device=device) * 0.3 + 0.1  # h
    targets[:, :, 4] = torch.randint(0, num_cls, (batch_size, num_boxes), device=device).float() # cls

    if device.type == 'cuda': torch.cuda.synchronize()
    t_data_end = time.time()

    print("正在预热 GPU...")
    if device.type == 'cuda':
        with torch.amp.autocast('cuda'): 
            warmup_preds = model(imgs)
            warmup_loss = criterion(warmup_preds, targets, imgs)
        scaler.scale(warmup_loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
    else:
        warmup_preds = model(imgs)
        warmup_loss = criterion(warmup_preds, targets, imgs)
        warmup_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if device.type == 'cuda': 
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    t_train_start = time.time()

    # 前向传播
    if device.type == 'cuda':
        with torch.amp.autocast('cuda'):
            preds = model(imgs)
    else:
        preds = model(imgs)
    
    if device.type == 'cuda': torch.cuda.synchronize()
    t_forward = time.time()

    # Loss 计算
    if device.type == 'cuda':
        with torch.amp.autocast('cuda'):
            loss = criterion(preds, targets, imgs)
    else:
        loss = criterion(preds, targets, imgs)
        
    if device.type == 'cuda': torch.cuda.synchronize()
    t_loss = time.time()

    # 反向传播
    if device.type == 'cuda':
        scaler.scale(loss).backward()
    else:
        loss.backward()
        
    if device.type == 'cuda': torch.cuda.synchronize()
    t_backward = time.time()

    # 参数更新
    if device.type == 'cuda':
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad()
    
    if device.type == 'cuda': torch.cuda.synchronize()
    t_train_end = time.time()

    # 提取与打印数据
    print(f"Total Loss : {loss.item():.4f}")
    
    box_outs, cls_outs, _ = preds
    print("\n --- Shape验证 ---")
    print(f"预测框 Shape    : {list(box_outs.shape)}   # [Batch, 4, _]")
    print(f"分类输出 Shape  : {list(cls_outs.shape)}  # [Batch, nc, _]")
    
    print("\n --- 耗时统计 (单次 Batch) ---")
    print(f"数据加载耗时 : {(t_data_end - t_data_start) * 1000:.2f} ms")
    print(f"前向传播 : {(t_forward - t_train_start) * 1000:.2f} ms")
    print(f"Loss计算 : {(t_loss - t_forward) * 1000:.2f} ms")
    print(f"反向传播 : {(t_backward - t_loss) * 1000:.2f} ms")
    print(f"参数更新 : {(t_train_end - t_backward) * 1000:.2f} ms")
    total_time = (t_train_end - t_train_start) * 1000
    print(f"单轮训练耗时 : {total_time:.2f} ms (FPS: {batch_size / (total_time/1000):.1f} imgs/s)")

    print("\n --- 显存峰值统计 ---")
    if device.type == 'cuda':
        mem_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"显存峰值占用    : {mem_mb:.2f} MB")
        reserved_mb = torch.cuda.max_memory_reserved(device) / 1024**2
        print(f"PyTorch保留显存 : {reserved_mb:.2f} MB")

if __name__ == "__main__":
    test_training_performance()