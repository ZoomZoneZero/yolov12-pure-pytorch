# ======================================================
#     YOLOv12 中央配置文件 - 修改此处即可适配新数据集     #
# ======================================================
import os
# ======================================================
# --- 模型通用参数区 ---

CLASSES_PATH = "v_class.txt" # 类别文件路径

INPUT_SHAPE = [640, 640]  # 图片尺寸
SCALES              = 'l'

# ======================================================


# ======================================================
# --- 训练模型参数区 ---

# 数据集设置
DATASET_NAME = "BCCD"  # 数据集文件夹名称 
MOSAIC       = True
MOSAIC_PROB         = 1.0
MIXUP        = True
MIXUP_PROB          = 0.15

# 权重初始化设置、保存设置
MODEL_PATH_         = ''
SEED                = 11
SAVE_PERIOD  = 10       # 保存间隔，不宜过小
SAVE_DIR     = f"logs_{DATASET_NAME}_{SCALES}"   # 也可直接设为logs,目前命名方式便于区分不同数据集

# 设备类设置
CUDA        = True
FP16        = True      # 混合精度训练，节省显存
DISTRIBUTED = False     # Linux下设为True开启多卡DDP
SYNC_BN     = False     # 多卡同步BatchNorm

# 训练计划   
INIT_EPOCH          = 0
FREEZE_EPOCH        = 0
FREEZE_BATCH_SIZE   = 2
UNFREEZE_EPOCH      = 500
UNFREEZE_BATCH_SIZE = 2
FREEZE_TRAIN        = True  # 是否进行冻结主干训练
EVAL_PERIOD         = 10     # 每隔几轮进行评估（一般为10）

# 优化器与学习率 
INIT_LR             = 1e-2
MIN_LR              = INIT_LR * 0.01
OPTIMIZER_TYPE      = "sgd"
MOMENTUM            = 0.937
WEIGHT_DECAY        = 1e-2
LR_DECAY_TYPE       = "cos"

#以下参数自动生成，无需设置
INDEX_DIR = f"{DATASET_NAME}_index"
TRAIN_TXT = os.path.join(INDEX_DIR, "train.txt")
VAL_TXT   = os.path.join(INDEX_DIR, "val.txt")
# ======================================================


# ======================================================
# --- 模型预测参数区 --- (想用get_map计算mAP 在这配置前两项)
MODEL_PATH      = "model_data/yolov12_l.pth"    # 预测时权重路径，不可为空
CUDA            = True

CONFIDENCE      = 0.2   # 只有得分大于置信度的预测框会被保留下来     
NMS_IOU         = 0.5   # 非极大抑制所用到的nms_iou大小
LETTERBOX_IMAGE = True  # 该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize
CROP            = False # 指定了是否在单张图片预测后对目标进行截取
COUNT           = False # 指定了是否进行目标的计数

# =======================================================