import datetime
import os
import math
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
from torch.utils.data import DataLoader
from yolov12 import Yolo12_train
from loss import Yolo12_Loss
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes, seed_everything
from utils.utils_fit import fit_one_epoch
from utils.utils_annotation import auto_annotation
from config import *

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

class ModelEMA:
    """ 
    指数移动平均 (Exponential Moving Average)
    在训练时维护一个参数平滑版的模型，能有效提高模型在测试集上的鲁棒性。
    """
    def __init__(self, model, decay=0.9999, tau=2000):
        self.ema = deepcopy(model.module if hasattr(model, 'module') else model).eval()
        self.updates = 0
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        # ema_weight = decay * ema_weight + (1 - decay) * model_weight
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)
            msd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

def get_lr_scheduler(lr_decay_type, lr_limit_max, lr_limit_min, total_iters, warmup_iters_ratio=0.05, warmup_lr_ratio=0.1):
    """
    YOLOv12 学习率调度器
    lr_limit_max: 设定的最高学习率 (如 0.01)
    lr_limit_min: 设定的最低学习率 (如 0.0001)
    """
    def lr_fn(epoch):
        warmup_total_iters = max(total_iters * warmup_iters_ratio, 5)
        
        # --- Warmup (线性爬坡) ---
        if epoch < warmup_total_iters:
            alpha = epoch / warmup_total_iters
            factor = warmup_lr_ratio + (1 - warmup_lr_ratio) * alpha
            return lr_limit_max * factor

        # --- 主训练阶段 (下坡) ---
        if lr_decay_type == "cos":
            import math
            epoch_rel = epoch - warmup_total_iters
            iter_rel = total_iters - warmup_total_iters
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * epoch_rel / iter_rel))
            return lr_limit_min + (lr_limit_max - lr_limit_min) * cos_factor
        
        # 默认线性下降
        return lr_limit_min + (lr_limit_max - lr_limit_min) * (1.0 - epoch / total_iters)

    return lr_fn


if __name__ == "__main__":
    # --- 全局硬件与环境参数 ---
    Cuda        = CUDA
    seed        = SEED
    fp16        = FP16      # 建议开启，节省显存并加速
    distributed = DISTRIBUTED     # Linux下设为True开启多卡DDP
    sync_bn     = SYNC_BN     # 多卡同步BatchNorm

    save_dir     = SAVE_DIR    
    save_period  = SAVE_PERIOD        
    mosaic       = MOSAIC 

    mosaic_prob         = MOSAIC_PROB
    mixup               = MIXUP
    mixup_prob          = MIXUP_PROB

    # --- 模型与数据配置 ---
    scales          = SCALES      # n, s, m, l, x
    classes_path = CLASSES_PATH
    # 预训练权重路径，若想从头练则设为 ''
    model_path   = MODEL_PATH_ 
    input_shape  = INPUT_SHAPE
    # --- 读取数据集索引 ---
    train_annotation_path = TRAIN_TXT
    val_annotation_path = VAL_TXT

    # --- 训练计划 ---
    Init_Epoch          = INIT_EPOCH
    Freeze_Epoch        = FREEZE_EPOCH    # 冻结训练轮数
    Freeze_batch_size   = FREEZE_BATCH_SIZE    # 冻结时可开大点
    UnFreeze_Epoch      = UNFREEZE_EPOCH   # 总轮数
    Unfreeze_batch_size = UNFREEZE_BATCH_SIZE     # 解冻后建议减小 batch
    Freeze_Train        = FREEZE_TRAIN  # 是否进行冻结主干训练

    # --- 优化器与学习率 ---
    Init_lr             = INIT_LR  # SGD建议1e-2, Adam建议1e-3
    Min_lr              = MIN_LR
    optimizer_type      = OPTIMIZER_TYPE # "sgd" 或 "adam"
    momentum            = MOMENTUM
    weight_decay        = WEIGHT_DECAY
    lr_decay_type       = LR_DECAY_TYPE # 学习率下降方式

    # --- 初始化设备环境 ---
    seed_everything(seed)
    if distributed:
        if os.name != 'nt': # Linux NCCL
            dist.init_process_group(backend="nccl")
            local_rank = int(os.environ["LOCAL_RANK"])
            device = torch.device("cuda", local_rank)
        else:
            print("Windows 不支持 DDP，自动降级为单卡模式")
            distributed = False
            local_rank, device = 0, torch.device('cuda')
    else:
        local_rank = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    if device.type == 'cuda':      
        torch.backends.cudnn.benchmark = True 
        torch.backends.cudnn.deterministic = False

    class_names, num_cls = get_classes(classes_path)
    model = Yolo12_train(scales=scales, num_cls=num_cls)

    if model_path != '':
        if local_rank == 0: print(f'Load weights from {model_path}...')
        model_dict = model.state_dict()
        # map_location 确保在没有 GPU 的机器上也能读权重
        pretrained_dict = torch.load(model_path, map_location='cpu')
        
        if "model" in pretrained_dict:
            pretrained_dict = pretrained_dict["model"]

        load_key, temp_dict = [], {}
        for k, v in pretrained_dict.items():
            # 自动映射 backbone 前缀逻辑
            rk = k if k in model_dict else f"backbone.{k}" if f"backbone.{k}" in model_dict else None
            # 跳过不同类别的 Detect Head
            if rk and np.shape(model_dict[rk]) == np.shape(v):
                temp_dict[rk] = v
                load_key.append(rk)

        model.load_state_dict(temp_dict, strict=False)
        if local_rank == 0:
            print(f"Successfully loaded {len(load_key)} / {len(model_dict)} keys.")
            if len(load_key) < (len(model_dict) * 0.8):
                print("警告: 加载比例过低，请检查模型结构与权重是否匹配！")
    # --- 损失函数与记录器 ---
    # 输入: model(images) 吐出的 (bboxes, logits, dist) 元组
    # 输出: total_loss 标量
    real_reg_max = model.module.detect.reg_max if distributed or isinstance(model, nn.DataParallel) else model.detect.reg_max
    yolo_loss = Yolo12_Loss(num_cls=num_cls, reg_max=real_reg_max)
    
    if local_rank == 0:
        time_str = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # --- 混合精度训练 (AMP) 设置 ---
    scaler = torch.amp.GradScaler('cuda') if fp16 else None

    # --- 优化器权重分组 ---
    # 卷积层的 Weight 需要正则化(Weight Decay)来防止过拟合
    # 但 Bias(偏置) 和 BN 层的参数如果加了正则化，反而会抑制模型收敛
    pg0, pg1, pg2 = [], [], []  
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)    # 组2: 所有的偏置 (no decay)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)  # 组0: 所有的归一化权重 (no decay)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # 组1: 所有的卷积权重 (apply decay)

    if optimizer_type == "sgd":
        optimizer = optim.SGD(pg0, Init_lr, momentum=momentum, nesterov=True)
    else:
        optimizer = optim.Adam(pg0, Init_lr, betas=(momentum, 0.999))

    optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
    optimizer.add_param_group({"params": pg2}) 
    del pg0, pg1, pg2

    # ---多卡/单卡模型 ---
    model_train = model.train()
    if Cuda:
        if distributed:
            # Linux 环境下的多卡同步逻辑
            if sync_bn:
                model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train, device_ids=[local_rank], find_unused_parameters=True
            )
        else:
            # Windows/单卡模式，使用 DataParallel 或直接转 cuda
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True # 开启卷积算子自动优化
            model_train = model_train.cuda()

    # --- EMA 实例化 ---
    # 验证集评估(mAP)时应使用 ema.ema 模型，其结果比 model 稳
    ema = ModelEMA(model_train) if local_rank == 0 else None

    try:
        f = open(train_annotation_path)
        f.close()
    except:
        auto_annotation(DATASET_NAME, classes_path)

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train, num_val = len(train_lines), len(val_lines)

    # --- 学习率调度器初始化 ---
    # total_iters 设为最大 Epoch 数
    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr, Min_lr, UnFreeze_Epoch)

    # --- 冻结阶段：构建 DataLoader ---
    train_dataset = YoloDataset(train_lines, input_shape, num_cls, 
                                epoch_length=UnFreeze_Epoch, mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=True)
    val_dataset = YoloDataset(val_lines, input_shape, num_cls, mixup=False, mosaic_prob=0, mixup_prob=0, 
                              epoch_length=UnFreeze_Epoch, mosaic=False, train=False)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
        shuffle = False
    else:
        train_sampler, val_sampler = None, None
        shuffle = True

    # num_workers 在 Windows 下若报错可设为 0
    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=Freeze_batch_size, 
                     num_workers=4, pin_memory=True, drop_last=True, 
                     collate_fn=yolo_dataset_collate, sampler=train_sampler)
    gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=Freeze_batch_size, 
                         num_workers=4, pin_memory=True, drop_last=True, 
                         collate_fn=yolo_dataset_collate, sampler=val_sampler)

    # --- 实时 mAP 评估回调 ---
    if local_rank == 0:
        # EvalCallback 内部执行推理 -> NMS -> 算 AP
        eval_callback = EvalCallback(model, input_shape, class_names, num_cls, 
                                     val_lines, log_dir, Cuda, eval_flag=True, period=EVAL_PERIOD)
        eval_callback.ema = ema
    else:
        eval_callback = None

    UnFreeze_flag = False
    # --- 初始计算步长 (针对冻结阶段) ---
    batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size
    
    # 冻结阶段 (Freeze): 只练 Detect Head，保护预训练权重不被破坏
    # 解冻阶段 (Unfreeze): 全网参数参与梯度下降
    
    for epoch in range(Init_Epoch, UnFreeze_Epoch):
        if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
            if local_rank == 0:
                print("\n>>> Unfreezing all layers for full-network fine-tuning...")
            
            for param in model.parameters():
                param.requires_grad = True

            batch_size = Unfreeze_batch_size
            
            gen = DataLoader(train_dataset, shuffle=(train_sampler is None), batch_size=batch_size, 
                             num_workers=4, pin_memory=True, drop_last=True, 
                             collate_fn=yolo_dataset_collate, sampler=train_sampler)
            gen_val = DataLoader(val_dataset, shuffle=(val_sampler is None), batch_size=batch_size, 
                                 num_workers=4, pin_memory=True, drop_last=True, 
                                 collate_fn=yolo_dataset_collate, sampler=val_sampler)
            
            epoch_step = num_train // batch_size
            epoch_step_val = num_val // batch_size
            
            if epoch_step == 0: raise ValueError("Batch size too large for current dataset.")
            
            UnFreeze_flag = True
            model_train.train()

        # --- 收尾阶段 (关闭 Mosaic 增强) ---
        # 让模型在真实的图像分布下进行收敛，修正 BN 层统计信息
        if epoch >= UnFreeze_Epoch - 15:
            gen.dataset.mosaic = False
            if local_rank == 0: print(f"--- Epoch {epoch}: Mosaic augmentation disabled ---")

        # --- 准备本轮学习率 ---
        # 传入当前 epoch，获取由 Warmup + Cosine 衰减计算后的 lr
        lr = lr_scheduler_func(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # --- 训练环节 ---
        # model_train: 包装后的模型 (DP/DDP)
        # yolo_loss:   接收 (bboxes, logits, dist) 元组的损失类
        # targets:     由 gen 吐出的 [B, M, 5] 形状 Tensor
        
        if local_rank == 0:
            m = model_train.module if hasattr(model_train, 'module') else model_train
            print(f'\nEpoch {epoch + 1}/{UnFreeze_Epoch}, Learning Rate: {lr:.6f}')
        
        # images -> model -> loss
        fit_one_epoch(
            model_train, model, ema, yolo_loss, loss_history, eval_callback, 
            optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
            UnFreeze_Epoch, Cuda, fp16, scaler, save_period, save_dir, local_rank
        )

        # --- 分布式同步 ---
        if distributed:
            dist.barrier()

    # --- 训练结束，收尾工作 ---
    if local_rank == 0:
        loss_history.writer.close()