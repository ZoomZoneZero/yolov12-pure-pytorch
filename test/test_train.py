import torch
import sys
import os
import time
import torch.optim as optim

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)

from yolov12 import Yolo12_train
from loss import Yolo12_Loss

def test_training_performance():
    # --- 自定义训练测算参数 ---
    scales = "n"          
    batch_size = 2      
    H, W = 640, 640       
    num_cls = 3          
    num_boxes = 30     # 单图最大物体数（估计）
    warmup_iters = 5   # 预热次数，稳定GPU频率
    test_iters = 20    # 测算次数，取平均更准确 
    # --------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True  

    print(f"========== 测算环境 ==========")
    print(f"当前运行设备: {device}")
    if device.type == 'cuda':
        print(f"显卡型号: {torch.cuda.get_device_name(0)}")

    model = Yolo12_train(scales=scales, num_cls=num_cls).to(device)
    model.train() 
    
    if hasattr(model, 'detect'):
        real_reg_max = model.detect.reg_max
    elif hasattr(model, 'model') and hasattr(model.model[-1], 'reg_max'):
        real_reg_max = model.model[-1].reg_max
    else:
        real_reg_max = 16 
        
    criterion = Yolo12_Loss(num_cls=num_cls, reg_max=real_reg_max).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.937)
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    imgs = torch.randn(batch_size, 3, H, W, device=device)
    targets = torch.zeros(batch_size, num_boxes, 5, device=device)
    targets[:, :, 0] = torch.rand(batch_size, num_boxes, device=device) * 0.6 + 0.2  # cx
    targets[:, :, 1] = torch.rand(batch_size, num_boxes, device=device) * 0.6 + 0.2  # cy
    targets[:, :, 2] = torch.rand(batch_size, num_boxes, device=device) * 0.3 + 0.1  # w
    targets[:, :, 3] = torch.rand(batch_size, num_boxes, device=device) * 0.3 + 0.1  # h
    targets[:, :, 4] = torch.randint(0, num_cls, (batch_size, num_boxes), device=device).float() # cls
    if device.type == 'cuda': torch.cuda.synchronize()

    try:
        from thop import profile
        macs, _ = profile(model, inputs=(imgs, ), verbose=False)
        flops_info = f"{(macs * 2) / 1e9:.2f} GFLOPs (仅代表前向)"
    except ImportError:
        flops_info = "请安装 thop 以查看计算量"

    print("正在预热 GPU...")
    for _ in range(warmup_iters):
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'): 
                warmup_preds = model(imgs)
                warmup_loss = criterion(warmup_preds, targets, imgs)
            scaler.scale(warmup_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            warmup_preds = model(imgs)
            warmup_loss = criterion(warmup_preds, targets, imgs)
            warmup_loss.backward()
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
    if device.type == 'cuda': 
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    time_fwd, time_loss, time_bwd, time_step = 0.0, 0.0, 0.0, 0.0

    for _ in range(test_iters):
        t0 = time.time()
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
        else:
            preds = model(imgs)
        if device.type == 'cuda': torch.cuda.synchronize()
        t1 = time.time()
        time_fwd += (t1 - t0)

        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                loss = criterion(preds, targets, imgs)
        else:
            loss = criterion(preds, targets, imgs)
        if device.type == 'cuda': torch.cuda.synchronize()
        t2 = time.time()
        time_loss += (t2 - t1)

        if device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if device.type == 'cuda': torch.cuda.synchronize()
        t3 = time.time()
        time_bwd += (t3 - t2)

        if device.type == 'cuda':
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if device.type == 'cuda': torch.cuda.synchronize()
        t4 = time.time()
        time_step += (t4 - t3)

    avg_fwd = (time_fwd / test_iters) * 1000
    avg_loss = (time_loss / test_iters) * 1000
    avg_bwd = (time_bwd / test_iters) * 1000
    avg_step = (time_step / test_iters) * 1000
    
    total_time_batch = avg_fwd + avg_loss + avg_bwd + avg_step
    total_time_img = total_time_batch / batch_size
    fps = (1000.0 / total_time_batch) * batch_size

    if device.type == 'cuda':
        mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        mem_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
    else:
        mem_allocated, mem_reserved = 0.0, 0.0

    box_outs, cls_outs = preds[0], preds[1]

    with torch.no_grad():
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                loss_ = criterion(Yolo12_train(scales=scales, num_cls=num_cls).to(device)(imgs), targets, imgs)
        else:
            loss_ = criterion(Yolo12_train(scales=scales, num_cls=num_cls).to(device)(imgs), targets, imgs)  

    print("\n" + "="*12 + " 性能测算结果 " + "="*12)
    print(f"模型规格       : YOLOv12-{scales} (Train Mode)")
    print(f"输入尺寸       : Batch={batch_size}, C=3, H={H}, W={W}")
    print(f"目标设定       : 单图最高物体数={num_boxes}, 类别数={num_cls}")
    print(f"模型参数量     : {total_params / 1e6:.2f} M (可训练: {trainable_params / 1e6:.2f} M)")
    print(f"计算量 (FLOPs) : {flops_info}")
    print("-" * 42)
    print(f"耗时明细 (单次 Batch 平均):")
    print(f"  - 前向传播   : {avg_fwd:.2f} ms")
    print(f"  - Loss计算   : {avg_loss:.2f} ms")
    print(f"  - 反向传播   : {avg_bwd:.2f} ms")
    print(f"  - 参数更新   : {avg_step:.2f} ms")
    print("-" * 42)
    print(f"总批次平均耗时 : {total_time_batch:.2f} ms")
    print(f"单张图片总耗时 : {total_time_img:.2f} ms")
    print(f"训练吞吐量     : {fps:.2f} FPS (imgs/s)")
    if device.type == 'cuda':
        print(f"峰值显存分配   : {mem_allocated:.2f} MB (模型真实占用)")
        print(f"峰值显存预留   : {mem_reserved:.2f} MB")
    print("-" * 42)
    print(f"Loss 样例值    : {loss_.item():.4f}")
    print(f"预测框 Shape   : {list(box_outs.shape)}")
    print(f"分类输出 Shape : {list(cls_outs.shape)}")
    print("=" * 42)

if __name__ == "__main__":
    test_training_performance()