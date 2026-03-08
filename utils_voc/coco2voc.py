import os
import json
from xml.etree.ElementTree import Element, SubElement, ElementTree, tostring
from xml.dom import minidom

# ================= 配置区域 =================
# 请修改这里的路径
json_file_path = 'train\_annotations.coco.json' 
xml_save_path = 'VOCdevkit\Annotations'  #注意保留后面的 Annotations
# ===========================================

def coco_to_voc(json_path, output_dir):
    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 读取 JSON 文件
    print(f"正在读取 {json_path} ...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {json_path}，请检查路径是否正确。")
        return

    # 构建类别映射 (id -> name)
    category_map = {}
    if 'categories' in data:
        for cat in data['categories']:
            category_map[cat['id']] = cat['name']
    else:
        print("错误: JSON文件中没有 'categories' 字段。")
        return
    
    print(f"检测到的类别: {category_map}")

    # 构建图片索引 (id -> image_info)
    images_dict = {}
    if 'images' in data:
        for img in data['images']:
            images_dict[img['id']] = img
    else:
        print("错误: JSON文件中没有 'images' 字段。")
        return

    # 将标注按图片 ID 分组
    annotations_dict = {}
    if 'annotations' in data:
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in annotations_dict:
                annotations_dict[img_id] = []
            annotations_dict[img_id].append(ann)
    else:
        print("警告: JSON文件中没有找到 'annotations' 字段，将只生成不带标注的XML。")

    # 遍历所有图片并生成 XML
    count = 0
    for img_id, img_info in images_dict.items():
        filename = img_info['file_name']
        height = img_info['height']
        width = img_info['width']
        
        # 创建 XML 根节点
        root = Element('annotation')
        
        # 文件夹名
        SubElement(root, 'folder').text = 'VOC'
        SubElement(root, 'filename').text = filename
        
        # 图片大小
        size = SubElement(root, 'size')
        SubElement(size, 'width').text = str(width)
        SubElement(size, 'height').text = str(height)
        SubElement(size, 'depth').text = '3'

        # 处理该图片的所有标注
        if img_id in annotations_dict:
            for ann in annotations_dict[img_id]:
                # 获取类别名称
                cat_id = ann['category_id']
                if cat_id not in category_map:
                    continue 
                class_name = category_map[cat_id]

                # COCO bbox: [x_min, y_min, width, height]
                bbox = ann['bbox']
                xmin = bbox[0]
                ymin = bbox[1]
                w = bbox[2]
                h = bbox[3]

                # VOC bbox: [xmin, ymin, xmax, ymax]
                xmax = xmin + w
                ymax = ymin + h

                # 创建 object 节点
                obj = SubElement(root, 'object')
                SubElement(obj, 'name').text = class_name
                SubElement(obj, 'pose').text = 'Unspecified'
                SubElement(obj, 'truncated').text = '0'
                SubElement(obj, 'difficult').text = '0'

                bndbox = SubElement(obj, 'bndbox')
                SubElement(bndbox, 'xmin').text = str(int(xmin))
                SubElement(bndbox, 'ymin').text = str(int(ymin))
                SubElement(bndbox, 'xmax').text = str(int(xmax))
                SubElement(bndbox, 'ymax').text = str(int(ymax))

        # 格式化 XML 字符串
        xml_str = minidom.parseString(
            tostring(root, 'utf-8')
        ).toprettyxml(indent="    ")

        # 保存 XML 文件
        # 去掉原文件后缀，加上 .xml
        file_name_without_ext = os.path.splitext(filename)[0]
        xml_filename = f"{file_name_without_ext}.xml"
        save_path = os.path.join(output_dir, xml_filename)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(xml_str)
        
        count += 1
        if count % 100 == 0:
            print(f"已处理 {count} 张图片...")

    print(f"转换完成！共生成 {count} 个 XML 文件保存在 {output_dir}")

if __name__ == '__main__':
    coco_to_voc(json_file_path, xml_save_path)