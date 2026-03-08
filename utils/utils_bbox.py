import torch
import torchvision

def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    # 输入均为 Tensor，且在同一设备
    device = box_xy.device
    
    # 将 [x, y] 翻转为 [y, x] 以适配 VOC 习惯
    box_xy = torch.flip(box_xy, dims=[-1])
    box_wh = torch.flip(box_wh, dims=[-1])       

    input_shape = torch.tensor(input_shape, device=device, dtype=torch.float32)
    image_shape = torch.tensor(image_shape, device=device, dtype=torch.float32)

    if letterbox_image:
        # 处理黑边缩放
        scale = torch.min(input_shape / image_shape)
        new_shape = torch.round(image_shape * scale)
        offset = (input_shape - new_shape) / 2. / input_shape
        box_xy = (box_xy - offset) * (input_shape / new_shape)
        box_wh *= (input_shape / new_shape)

    # 重点：输出顺序为 ymin, xmin, ymax, xmax
    box_mins  = box_xy - (box_wh / 2.)
    box_maxes = box_xy + (box_wh / 2.)
    boxes = torch.cat([box_mins, box_maxes], dim=-1)
    
    # 乘回原图像素
    boxes *= torch.cat([image_shape, image_shape], dim=-1)
    return boxes


def decode_outputs(outputs, input_shape):
    # 如果 outputs 已经是一个 Tensor 且是 3 维的 [B, 7, 8400]
    if isinstance(outputs, (list, tuple)) and len(outputs) == 5:
        p_box = outputs[0]
        p_cls = outputs[1].sigmoid() # 评估必须转概率

        # 用分类分数的最大值作为伪置信度
        conf, _ = torch.max(p_cls, dim=1, keepdim=True) # [B, 1, 8400]
        
        # 拼装成 [B, 4 + 1 + nc, 8400]
        final = torch.cat([p_box, conf, p_cls], dim=1)
        
        # 最后转置为 [B, 8400, 5 + nc] 返回
        return final.permute(0, 2, 1)
    
    grids   = []
    strides = []
    hw      = [x.shape[-2:] for x in outputs]
    #---------------------------------------------------#
    #   outputs输入前代表每个特征层的预测结果
    #   batch_size, 4 + 1 + num_classes, 80, 80 => batch_size, 4 + 1 + num_classes, 6400
    #   batch_size, 5 + num_classes, 40, 40
    #   batch_size, 5 + num_classes, 20, 20
    #   batch_size, 4 + 1 + num_classes, 6400 + 1600 + 400 -> batch_size, 4 + 1 + num_classes, 8400
    #   堆叠后为batch_size, 8400, 5 + num_classes
    #---------------------------------------------------#
    outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
    #---------------------------------------------------#
    #   获得每一个特征点属于每一个种类的概率
    #---------------------------------------------------#
    outputs[:, :, 4:] = torch.sigmoid(outputs[:, :, 4:])
    for h, w in hw:
        #---------------------------#
        #   根据特征层的高宽生成网格点
        #---------------------------#   
        grid_y, grid_x  = torch.meshgrid([torch.arange(h), torch.arange(w)])
        #---------------------------#
        #   1, 6400, 2
        #   1, 1600, 2
        #   1, 400, 2
        #---------------------------#   
        grid            = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
        shape           = grid.shape[:2]

        grids.append(grid)
        strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))
    #---------------------------#
    #   将网格点堆叠到一起
    #   1, 6400, 2
    #   1, 1600, 2
    #   1, 400, 2
    #
    #   1, 8400, 2
    #---------------------------#
    grids               = torch.cat(grids, dim=1).type(outputs.type())
    strides             = torch.cat(strides, dim=1).type(outputs.type())
    #------------------------#
    #   根据网格点进行解码
    #------------------------#
    outputs[..., :2]    = (outputs[..., :2] + grids) * strides
    outputs[..., 2:4]   = torch.exp(outputs[..., 2:4]) * strides
    #-----------------#
    #   归一化
    #-----------------#
    outputs[..., [0,2]] = outputs[..., [0,2]] / input_shape[1]
    outputs[..., [1,3]] = outputs[..., [1,3]] / input_shape[0]
    return outputs

def non_max_suppression(prediction, num_classes, input_shape, image_shape, letterbox_image, conf_thres=0.5, nms_thres=0.4):
    """
    严谨版 NMS，适配 YOLOv12 无 objectness 分支的情况
    参数：
    prediction: [B, 8400, 4 + nc]
    image_shape: 原图的高宽 [h, w]
    """
    bs = prediction.shape[0]
    # 坐标转换：从 [x1, y1, x2, y2] 转为 [center_x, center_y, w, h] 
    # 因为后续的 yolo_correct_boxes 习惯处理中心点格式
    xc = (prediction[..., 0:1] + prediction[..., 2:3]) / 2
    yc = (prediction[..., 1:2] + prediction[..., 3:4]) / 2
    w = prediction[..., 2:3] - prediction[..., 0:1]
    h = prediction[..., 3:4] - prediction[..., 1:2]
    prediction[..., 0] = xc.squeeze(-1)
    prediction[..., 1] = yc.squeeze(-1)
    prediction[..., 2] = w.squeeze(-1)
    prediction[..., 3] = h.squeeze(-1)

    output = [None] * bs
    for i, image_pred in enumerate(prediction):
        # 提取分类分和索引
        class_conf, class_pred = torch.max(image_pred[:, 4 : 4 + num_classes], 1, keepdim=True)
        
        # 过滤
        conf_mask = (class_conf.squeeze() >= conf_thres)
        if not conf_mask.any(): continue 
        
        # 筛选数据
        image_pred = image_pred[conf_mask]
        class_conf = class_conf[conf_mask]
        class_pred = class_pred[conf_mask]

        # 坐标还原 [N, 4] -> (ymin, xmin, ymax, xmax)
        detections = yolo_correct_boxes(image_pred[:, :2], image_pred[:, 2:4], input_shape, image_shape, letterbox_image)

        # 拼装成 7 列格式
        # 列0-3: 坐标
        # 列4: 伪置信度 (全填 1.0)
        # 列5: 分类分数
        # 列6: 类别ID
        conf_ones = torch.ones_like(class_conf)
        curr_boxes = torch.cat([detections, conf_ones, class_conf, class_pred.float()], 1)

        # 执行 NMS
        # 注意：这里算 score 的时候用 curr_boxes[:, 4] * curr_boxes[:, 5] 
        keep = torchvision.ops.batched_nms(
            curr_boxes[:, :4], 
            curr_boxes[:, 4] * curr_boxes[:, 5], 
            curr_boxes[:, 6], 
            nms_thres
        )
        
        output[i] = curr_boxes[keep].detach().cpu().numpy()

    return output


def encode(bboxes, anchor_points):
    """
    bboxes: [N, 4] (x1, y1, x2, y2) 像素级
    anchor_points: [N, 2] (cx, cy) 像素级
    """
    x1y1, x2y2 = torch.split(bboxes, 2, dim=-1)
    # 距离 = 锚点 - 左上 / 右下 - 锚点
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    # 返回 [N, 4] (l, t, r, b)
    return torch.cat((lt, rb), dim=-1)