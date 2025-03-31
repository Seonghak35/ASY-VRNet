import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from nets.efficient_vrnet import EfficientVRNet
from utils.utils import get_classes
from utils.callbacks import EvalCallback
from utils_seg.callbacks import EvalCallback as EvalCallback_seg
from nets.yolo_training import (ModelEMA, YOLOLoss, get_lr_scheduler,
                                set_optimizer_lr, weights_init)

# ---------------------------------------------------
# 사용자 설정
# ---------------------------------------------------
model_path = '/workspaces/ASY-VRNet/logs/last_epoch_weights.pth'
classes_path = 'model_data/waterscenes.txt'

phi = 'nano'

VOCdevkit_path = '/workspaces/ASY-VRNet/VOCdevkit/VOC2007'
num_classes_seg = 8 # waterscenes dataset have 7 classes

val_annotation_path = '2007_val.txt'
radar_data_dir = "/workspaces/ASY-VRNet/VOCdevkit/VOC2007/VOCradar"
map_log_dir = 'map_logs'
miou_log_dir = 'miou_logs'
map_out_path = '.temp_map_out'
miou_out_path = '.temp_miou_out'

Cuda = False
input_shape = [320, 320]

# ---------------------------------------------------

# 클래스 이름 불러오기
class_names, num_classes = get_classes(classes_path)

# 모델 정의 및 weight 로드
model = EfficientVRNet(num_classes=num_classes, num_seg_classes=num_classes_seg, phi=phi)
weights_init(model)

print('Load weights {}.'.format(model_path))
model_dict = model.state_dict()
pretrained_dict = torch.load(model_path, map_location='cpu')
load_key, no_load_key, temp_dict = [], [], {}
for k, v in pretrained_dict.items():
    if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
        temp_dict[k] = v
        load_key.append(k)
    else:
        no_load_key.append(k)
model_dict.update(temp_dict)
model.load_state_dict(model_dict)
print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

# 검증 데이터 라인 불러오기
with open(val_annotation_path, encoding='utf-8') as f:
    val_lines = f.readlines()
num_val = len(val_lines)

# EvalCallback 생성
evaluator = EvalCallback(
    net=model,
    input_shape=input_shape,
    class_names=class_names,
    num_classes=num_classes,
    val_lines=val_lines,
    log_dir=map_log_dir,
    cuda=Cuda,
    local_rank=1,
    radar_path=radar_data_dir,
    map_out_path=map_out_path,
    eval_flag=True,
    period=1
)

evaluator_seg = EvalCallback_seg(
    net=model,
    input_shape=input_shape,
    num_classes=num_classes_seg,
    image_ids=val_lines,
    dataset_path=VOCdevkit_path,
    log_dir=miou_log_dir,
    cuda=Cuda,
    local_rank=1,
    radar_path=radar_data_dir,
    miou_out_path=map_out_path,
    eval_flag=True,
    period=1
)

model.eval()
# 실제 평가 수행
evaluator.on_epoch_end(epoch=0, model_eval=model)
evaluator_seg.on_epoch_end(epoch=0, model_eval=model)
