import os
import sys

root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_path not in sys.path:
    sys.path.append(root_path)

import io
import shutil
from tqdm import tqdm
from contextlib import redirect_stdout
from PIL import Image
from yolo4use import YOLO
from utils.utils import get_classes
from utils.callbacks import EvalCallback
from config import *
from utils.utils_map import get_coco_map

model_path  = MODEL_PATH 
scales      = SCALES
class_names, num_classes = get_classes(CLASSES_PATH)
map_out_path = ".temp_eval"

yolo = YOLO(model_path=model_path, scales=scales)

with open(VAL_TXT, 'r', encoding='utf-8') as f:
    val_lines = f.readlines()

if os.path.exists(map_out_path): shutil.rmtree(map_out_path)
os.makedirs(os.path.join(map_out_path, "ground-truth"))
os.makedirs(os.path.join(map_out_path, "detection-results"))

eval_callback = EvalCallback(yolo.net, INPUT_SHAPE, class_names, num_classes, val_lines, map_out_path, CUDA)

for annotation_line in tqdm(val_lines):
    line        = annotation_line.split()
    image_id    = os.path.basename(line[0]).split('.')[0]
    image       = Image.open(line[0])
    gt_boxes    = [box.split(',') for box in line[1:]]
    
    # 生成预测 TXT
    eval_callback.get_map_txt(image_id, image, class_names, map_out_path)
    
    # 生成真值 TXT
    with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
        for box in gt_boxes:
            left, top, right, bottom, obj = box
            new_f.write("%s %s %s %s %s\n" % (class_names[int(obj)], left, top, right, bottom))


f = io.StringIO()
with redirect_stdout(f):
    get_coco_map(class_names = class_names, path = map_out_path)

full_output = f.getvalue()

filtered_lines = []
for line in full_output.split('\n'):
    if line.strip().startswith("Average Precision") or line.strip().startswith("Average Recall"):
        filtered_lines.append(line)

final_txt_name = f"map_{DATASET_NAME}_{scales}.txt"
print("")
with open(final_txt_name, "w", encoding="utf-8") as f_out:
    f_out.write("\n".join(filtered_lines))

shutil.rmtree(map_out_path)

for l in filtered_lines: print(l)

print("Get map done.")

