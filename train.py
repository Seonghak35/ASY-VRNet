# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import datetime
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.efficient_vrnet import EfficientVRNet
from nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)
from utils.callbacks import LossHistory, EvalCallback
from utils_seg.callbacks import LossHistory as LossHistory_seg
from utils_seg.callbacks import EvalCallback as EvalCallback_seg
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes, show_config
from utils.utils_fit import fit_one_epoch
from utils_seg.utils import show_config as show_config_seg
from utils_seg.dataloader import DeeplabDataset, deeplab_dataset_collate
from utils_seg.utils_fit import fit_one_epoch as fit_one_epoch_seg
from utils_seg.callbacks import LossHistory as LossHistory_seg


if __name__ == "__main__":

    Cuda = False
    distributed = False
    sync_bn = False
    fp16 = True
    classes_path = 'model_data/waterscenes.txt'
    model_path = ''

    input_shape = [320, 320]
    phi = 'nano'

    mosaic = False
    mosaic_prob = 0.5
    mixup = False
    mixup_prob = 0.5
    special_aug_ratio = 0.6

    Init_Epoch = 0
    Freeze_Epoch = 1
    Freeze_batch_size = 2

    UnFreeze_Epoch = 100
    Unfreeze_batch_size = 8

    Freeze_Train = False

    Init_lr = 1e-3
    Min_lr = Init_lr * 0.01

    optimizer_type = "adam"
    momentum = 0.9
    weight_decay = 0
    lr_decay_type = "cos"

    save_period = 10
    save_dir = 'logs'

    eval_flag = True
    eval_period = 10

    num_workers = 0

    radar_file_path = "/workspaces/ASY-VRNet/VOCdevkit/VOC2007/VOCradar"
    train_annotation_path = '2007_train.txt'
    val_annotation_path = '2007_val.txt'

    # ============================ segmentation hyperparameters ============================= #

    VOCdevkit_path = '/workspaces/ASY-VRNet/VOCdevkit/VOC2007'
    num_classes_seg = 8 # waterscenes dataset have 7 classes.

    dice_loss = True
    focal_loss = True

    cls_weights = np.ones([num_classes_seg], np.float32)

    save_dir_seg = 'logs_seg'

    # ======================================================================================= #

    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        local_rank = 1
        rank = 1

    class_names, num_classes = get_classes(classes_path)

    # model = EfficientVRNet(num_classes=num_classes, num_seg_classes=num_classes_seg, phi=phi).cuda(local_rank)
    model = EfficientVRNet(num_classes=num_classes, num_seg_classes=num_classes_seg, phi=phi)
    weights_init(model)
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0 or local_rank == 1:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    yolo_loss = YOLOLoss(num_classes, fp16)
    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    log_dir_seg = os.path.join(save_dir_seg, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)
    loss_history_seg = LossHistory_seg(log_dir_seg, model, input_shape=input_shape)

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler

        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()

    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank],
                                                                    find_unused_parameters=True)
        else:
            # model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.to(device)

    ema = ModelEMA(model_train)

    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(
        classes_path=classes_path, model_path=model_path, input_shape=input_shape, \
        Init_Epoch=Init_Epoch, Freeze_Epoch=Freeze_Epoch, UnFreeze_Epoch=UnFreeze_Epoch,
        Freeze_batch_size=Freeze_batch_size, Unfreeze_batch_size=Unfreeze_batch_size, Freeze_Train=Freeze_Train, \
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum,
        lr_decay_type=lr_decay_type, \
        save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
    )
 
    wanted_step = 5e4 if optimizer_type == "sgd" else 1.5e4
    total_step = num_train // Unfreeze_batch_size * UnFreeze_Epoch
    if total_step <= wanted_step:
        if num_train // Unfreeze_batch_size == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')
        wanted_epoch = wanted_step // (num_train // Unfreeze_batch_size) + 1
        print("\n\033[1;33;44m[Warning] When using the %s optimizer, it is recommended to set the total training steps to at least %d.\033[0m" % (optimizer_type, wanted_step))
        print("\033[1;33;44m[Warning] In this run, the total number of training samples is %d, the Unfreeze_batch_size is %d, and the total number of training epochs is %d, resulting in %d total training steps.\033[0m" % (
            num_train, Unfreeze_batch_size, UnFreeze_Epoch, total_step))
        print("\033[1;33;44m[Warning] Since the total training steps is %d, which is less than the recommended %d steps, it is suggested to set the number of epochs to %d.\033[0m" % (
            total_step, wanted_step, wanted_epoch))

    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            for param in model.backbone.backbone.parameters():
                param.requires_grad = False

        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        pg0, pg1, pg2 = [], [], []
        for k, v in model.named_modules():
            if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                pg2.append(v.bias)
            if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                pg0.append(v.weight)
            elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                pg1.append(v.weight)
        optimizer = {
            'adam': optim.Adam(pg0, Init_lr_fit, betas=(momentum, 0.999)),
            'sgd': optim.SGD(pg0, Init_lr_fit, momentum=momentum, nesterov=True)
        }[optimizer_type]
        optimizer.add_param_group({"params": pg1, "weight_decay": weight_decay})
        optimizer.add_param_group({"params": pg2})

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        if ema:
            ema.updates = epoch_step * Init_Epoch

        train_dataset = YoloDataset(annotation_lines=train_lines, input_shape=input_shape, num_classes=num_classes,
                                    epoch_length=UnFreeze_Epoch, \
                                    mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0,
                                    train=False, special_aug_ratio=0, radar_root=radar_file_path,
                                    num_classes_seg=num_classes_seg, seg_dataset_path=VOCdevkit_path)

        val_dataset = YoloDataset(annotation_lines=val_lines, input_shape=input_shape, num_classes=num_classes,
                                  epoch_length=UnFreeze_Epoch, \
                                  mosaic=False, mixup=False, mosaic_prob=0, mixup_prob=0, train=False,
                                  special_aug_ratio=0, radar_root=radar_file_path,
                                  num_classes_seg=num_classes_seg, seg_dataset_path=VOCdevkit_path)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True, )
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False, )
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                         pin_memory=True,
                         drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
        gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

        eval_callback = EvalCallback(model, input_shape, class_names, num_classes, val_lines, log_dir, Cuda, \
                                     eval_flag=eval_flag, period=eval_period, radar_path=radar_file_path, local_rank=local_rank)
        eval_callback_seg = EvalCallback_seg(model, input_shape, num_classes_seg, val_lines, VOCdevkit_path,
                                             log_dir_seg, Cuda, eval_flag=eval_flag, period=eval_period,
                                             radar_path=radar_file_path, local_rank=local_rank)

        train_index = 0
        for epoch in range(Init_Epoch, UnFreeze_Epoch):
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size

                nbs = 64
                lr_limit_max = 1e-3 if optimizer_type == 'adam' else 5e-2
                lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                for param in model.backbone.backbone.parameters():
                    param.requires_grad = True

                epoch_step = num_train // batch_size
                epoch_step_val = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                if ema:
                    ema.updates = epoch_step * epoch

                gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                 pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)
                gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                                     pin_memory=True,
                                     drop_last=True, collate_fn=yolo_dataset_collate, sampler=val_sampler)

                UnFreeze_flag = True

            gen.dataset.epoch_now = epoch
            gen_val.dataset.epoch_now = epoch
            if distributed:
                train_sampler.set_epoch(epoch)

            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, loss_history_seg, eval_callback,
                          eval_callback_seg, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch,
                          Cuda, fp16, scaler, save_period, save_dir,
                          dice_loss, focal_loss, cls_weights, num_classes_seg, local_rank)


            if distributed:
                dist.barrier()

        if local_rank >= 1:
            loss_history.writer.close()
            loss_history_seg.writer.close()
