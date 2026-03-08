import os
import json
import xml.etree.ElementTree as ET
from tqdm import tqdm

# ================= 配置区域 =================
voc_root      = "MyBloodData"     # VOC文件夹
classes_path  = "v_class.txt"     # 类别文件
save_root     = "COCO_data"       # 生成的coco文件夹命名

# 在标准 voc文件夹下，以下配置自动生成，无需修改
xml_dir = os.path.join(voc_root, "Annotations")
img_dir = os.path.join(voc_root, "JPEGImages")

import os
import json
import xml.etree.ElementTree as ET
import random
import shutil
from tqdm import tqdm

class VOC2COCOConverter:
    def __init__(self, xml_dir, img_dir, save_root, classes_path, train_ratio=0.9):
        """
        xml_dir: 原始原始XML文件夹
        img_dir: 原始图片文件夹
        save_root: 想要生成的标准COCO根目录名 (如 'COCO_Blood')
        classes_path: v_class.txt 路径
        """
        self.xml_dir = xml_dir
        self.img_dir = img_dir
        self.save_root = save_root
        self.train_ratio = train_ratio
        
        # 加载类别
        with open(classes_path, 'r', encoding='utf-8') as f:
            self.classes = [line.strip() for line in f.readlines() if line.strip()]
        self.category_map = {name: i for i, name in enumerate(self.classes)}

        # 创建标准COCO目录结构
        self.train_img_dir = os.path.join(save_root, 'train2017')
        self.val_img_dir = os.path.join(save_root, 'val2017')
        self.ann_dir = os.path.join(save_root, 'annotations')
        
        for d in [self.train_img_dir, self.val_img_dir, self.ann_dir]:
            os.makedirs(d, exist_ok=True)

    def _get_base_json(self):
        return {
            "info": {"description": "Standard COCO2017 format converted from VOC"},
            "images": [],
            "annotations": [],
            "categories": [{"id": i, "name": name, "supercategory": "none"} for i, name in enumerate(self.classes)]
        }

    def convert(self):
        # 获取所有XML
        xml_files = [f for f in os.listdir(self.xml_dir) if f.endswith('.xml')]
        random.seed(42)
        random.shuffle(xml_files)
        
        # 划分训练/验证
        num_train = int(len(xml_files) * self.train_ratio)
        train_list = xml_files[:num_train]
        val_list = xml_files[num_train:]

        self.process_subset(train_list, "train")
        self.process_subset(val_list, "val")

    def process_subset(self, file_list, subset_name):
        print(f"正在处理 {subset_name} 子集...")
        coco_data = self._get_base_json()
        ann_id = 1
        img_id = 1
        
        target_img_dir = self.train_img_dir if subset_name == "train" else self.val_img_dir

        for xml_name in tqdm(file_list):
            xml_path = os.path.join(self.xml_dir, xml_name)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # 图片信息处理
            filename = root.find('filename').text
            size = root.find('size')
            w, h = int(size.find('width').text), int(size.find('height').text)

            # 物理复制图片到标准文件夹
            src_img = os.path.join(self.img_dir, filename)
            if os.path.exists(src_img):
                shutil.copy(src_img, target_img_dir)
            else:
                # 处理XML内记录的文件名与实际不符的情况
                potential_img = os.path.join(self.img_dir, xml_name.replace('.xml', '.jpg'))
                if os.path.exists(potential_img):
                    shutil.copy(potential_img, target_img_dir)
                    filename = os.path.basename(potential_img)

            coco_data["images"].append({
                "id": img_id,
                "file_name": filename,
                "width": w, "height": h
            })

            # 标注处理
            for obj in root.iter('object'):
                cls_name = obj.find('name').text
                if cls_name not in self.category_map: continue
                
                xmlbox = obj.find('bndbox')
                xmin, ymin = float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text)
                xmax, ymax = float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text)
                
                bw, bh = xmax - xmin, ymax - ymin
                
                coco_data["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": self.category_map[cls_name],
                    "bbox": [xmin, ymin, bw, bh],
                    "area": bw * bh,
                    "iscrowd": 0,
                    "segmentation": []
                })
                ann_id += 1
            img_id += 1

        # 保存JSON
        json_name = f"instances_{subset_name}2017.json"
        with open(os.path.join(self.ann_dir, json_name), 'w') as f:
            json.dump(coco_data, f, indent=2)

if __name__ == "__main__":
    converter = VOC2COCOConverter(
        xml_dir = "MyBlood/Annotations",
        img_dir = "MyBlood/JPEGImages",
        save_root = save_root, 
        classes_path = classes_path
    )
    converter.convert()