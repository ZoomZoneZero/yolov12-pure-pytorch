import os
import torch
from tqdm import tqdm
from utils.utils import get_lr

def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss        = 0
    val_loss    = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    
    model_train.train() # 确保开启 Train 模式
    
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                # 确保 targets 是一个整体 Tensor [B, M, 5] 传给 TAL
                targets = torch.stack([torch.from_numpy(ann).cuda(local_rank) for ann in targets]) if isinstance(targets, list) else targets.cuda(local_rank)

        optimizer.zero_grad()
        
        # --- 前向传播与多输出处理 ---
        if not fp16:
            # outputs 此时是 (bboxes, cls_logits, reg_dist) 元组
            outputs = model_train(images)
            # yolo_loss 内部会处理这个元组
            loss_value = yolo_loss(outputs, targets, images)
            loss_value.backward()
            optimizer.step()
        else:
            with torch.amp.autocast('cuda'):
                outputs = model_train(images)
                loss_value = yolo_loss(outputs, targets, images)
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()

        # --- EMA 更新 ---
        if ema:
            ema.update(model_train)

        loss += loss_value.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'loss'  : loss / (iteration + 1), 
                                'lr'    : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    # --- 验证阶段模式切换  ---
    model_train.eval() 
    
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            if cuda:
                images  = images.cuda(local_rank)
                targets = torch.stack([torch.from_numpy(ann).cuda(local_rank) for ann in targets]) if isinstance(targets, list) else targets.cuda(local_rank)

            outputs = model_train(images)
            loss_value = yolo_loss(outputs, targets, images)

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        
        # 记录日志与 mAP
        if loss_history:
            loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        
        # mAP 评估：使用 EMA 后的模型评估，分数更高
        if eval_callback:
            eval_model = ema.ema if ema else model
            eval_callback.on_epoch_end(epoch + 1, eval_model)
            
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))