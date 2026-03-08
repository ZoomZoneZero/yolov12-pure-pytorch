#!/bin/bash

CONFIG_FILE="config.py"
JOB_LIST="train_linux/jobs.txt"
LOG_DIR="logs"
mkdir -p $LOG_DIR

DEFAULT_OPTIMIZER="sgd"
DEFAULT_LR="1e-2"
DEFAULT_DECAY="5e-4"
DEFAULT_MOMENTUM="0.937"
DEFAULT_LR_TYPE="cos"
DEFAULT_MIN_LR_RATIO="0.01"
DEFAULT_WEIGHTS=""  
DEFAULT_MOSAIC="0.5"
DEFAULT_MIXUP="0.15"
DEFAULT_SEED="11"

# 获取参数值，如果是 "-" 或空则返回默认值
get_param() {
    local val=$1
    local default=$2
    if [[ -z "$val" || "$val" == "-" ]]; then
        echo "$default"
    else
        echo "$val"
    fi
}

echo ">>> 读取 $JOB_LIST ..."

NEXT_IS_JOB=false

# 过滤注释行和空行
grep -v "^\s*#" "$JOB_LIST" | grep -v "^\s*$" | while read -r line; do
    if [[ "$line" =~ ^[[:space:]]*\$\$ ]]; then
        NEXT_IS_JOB=true
        continue
    fi

    if [ "$NEXT_IS_JOB" = false ]; then
        continue
    fi

    params=($line)
    if [[ -z "${params[0]}" || -z "${params[1]}" ]]; then
        continue
    fi
    
    START_TIME=${params[0]}
    SCALE=${params[1]}
    EPOCHS=${params[2]}
    UNFREEZE_BS=${params[3]}
    FREEZE_EP=${params[4]}
    FREEZE_BS=${params[5]}

    OPTIMIZER=$(get_param "${params[6]}" "$DEFAULT_OPTIMIZER")
    INIT_LR=$(get_param "${params[7]}" "$DEFAULT_LR")
    WEIGHT_DECAY=$(get_param "${params[8]}" "$DEFAULT_DECAY")
    MOMENTUM=$(get_param "${params[9]}" "$DEFAULT_MOMENTUM")
    LR_TYPE=$(get_param "${params[10]}" "$DEFAULT_LR_TYPE")
    MIN_LR_RATIO=$(get_param "${params[11]}" "$DEFAULT_MIN_LR_RATIO")
    WEIGHTS=$(get_param "${params[12]}" "$DEFAULT_WEIGHTS")
    MOSAIC_PROB=$(get_param "${params[13]}" "$DEFAULT_MOSAIC")
    MIXUP_PROB=$(get_param "${params[14]}" "$DEFAULT_MIXUP")
    MOSAIC_BOOL=$(awk "BEGIN {print ($MOSAIC_PROB <= 0) ? \"False\" : \"True\"}")
    MIXUP_BOOL=$(awk "BEGIN {print ($MIXUP_PROB <= 0) ? \"False\" : \"True\"}")
    SEED=$(get_param "${params[15]}" "$DEFAULT_SEED")

    echo "=========================================================="
    echo "任务准备: Scale=$SCALE, Epochs=$EPOCHS, Opt=$OPTIMIZER, LR=$INIT_LR"
    echo "         Decay=$WEIGHT_DECAY, Seed=$SEED"
    echo "=========================================================="

    if [ "$START_TIME" != "now" ]; then
        CURRENT_EPOCH=$(date +%s)
        TARGET_EPOCH=$(date -d "$START_TIME" +%s)
        if [ $TARGET_EPOCH -lt $CURRENT_EPOCH ]; then
            TARGET_EPOCH=$(date -d "tomorrow $START_TIME" +%s)
        fi
        SLEEP_SEC=$((TARGET_EPOCH - CURRENT_EPOCH))
        echo ">>> 等待启动... 目标时间: $START_TIME (还剩 ${SLEEP_SEC}秒)"
        sleep $SLEEP_SEC
    fi
    
    sed -i "s/^SCALES *=.*/SCALES              = '${SCALE}'/" $CONFIG_FILE
    sed -i "s/^UNFREEZE_EPOCH *=.*/UNFREEZE_EPOCH      = ${EPOCHS}/" $CONFIG_FILE
    sed -i "s/^UNFREEZE_BATCH_SIZE *=.*/UNFREEZE_BATCH_SIZE = ${UNFREEZE_BS}/" $CONFIG_FILE
    sed -i "s/^FREEZE_EPOCH *=.*/FREEZE_EPOCH        = ${FREEZE_EP}/" $CONFIG_FILE
    sed -i "s/^FREEZE_BATCH_SIZE *=.*/FREEZE_BATCH_SIZE   = ${FREEZE_BS}/" $CONFIG_FILE
    sed -i "s/^OPTIMIZER_TYPE *=.*/OPTIMIZER_TYPE      = \"${OPTIMIZER}\"/" $CONFIG_FILE
    sed -i "s/^INIT_LR *=.*/INIT_LR             = ${INIT_LR}/" $CONFIG_FILE
    sed -i "s/^WEIGHT_DECAY *=.*/WEIGHT_DECAY        = ${WEIGHT_DECAY}/" $CONFIG_FILE
    sed -i "s/^MOMENTUM *=.*/MOMENTUM            = ${MOMENTUM}/" $CONFIG_FILE
    sed -i "s/^LR_DECAY_TYPE *=.*/LR_DECAY_TYPE       = \"${LR_TYPE}\"/" $CONFIG_FILE
    sed -i "s/^MIN_LR *=.*/MIN_LR              = INIT_LR * ${MIN_LR_RATIO}/" $CONFIG_FILE
    sed -i "s|^MODEL_PATH_ *=.*|MODEL_PATH_         = '${WEIGHTS}'|" $CONFIG_FILE
    sed -i "s/^MOSAIC *=.*/MOSAIC       = ${MOSAIC_BOOL}/" $CONFIG_FILE
    sed -i "s/^MOSAIC_PROB *=.*/MOSAIC_PROB         = ${MOSAIC_PROB}/" $CONFIG_FILE
    sed -i "s/^MIXUP *=.*/MIXUP        = ${MIXUP_BOOL}/" $CONFIG_FILE
    sed -i "s/^MIXUP_PROB *=.*/MIXUP_PROB          = ${MIXUP_PROB}/" $CONFIG_FILE
    sed -i "s/^SEED *=.*/SEED                = ${SEED}/" $CONFIG_FILE

    TASK_NAME="train_${SCALE}_${OPTIMIZER}_$(date +%Y%m%d_%H%M)"
    echo ">>> 正在运行... 日志: $LOG_DIR/$TASK_NAME.log"
    
    python train.py > "$LOG_DIR/${TASK_NAME}_console.log" 2>&1 < /dev/null

    echo ">>> 任务完成。"
    echo ""
    sleep 5 # 缓冲时间
done

echo "所有任务执行完毕！"