# 用于区分 coco 与 voc 数据集
def auto_annotation(dataset_path, classes_path):
    import os
    
    try:
        dir_contents = os.listdir(dataset_path)
    except FileNotFoundError:
        print(f"错误：找不到数据集路径 {dataset_path}")
        return None

    # 匹配 VOC 格式的特征
    if 'Annotations' in dir_contents:
        print("检测到 VOC 格式数据集 (包含大写 Annotations)")
        from utils_voc.voc_annotation import run_voc_annotation
        run_voc_annotation(dataset_path, classes_path)
        return "VOC"
    
    # 严格匹配 COCO 格式的特征
    elif 'annotations' in dir_contents:
        print("检测到 COCO 格式数据集 (包含小写 annotations)")
        from utils_coco.coco_annotation import run_coco_annotation
        run_coco_annotation(dataset_path)
        return "COCO"
    
    print(f"未能识别数据集格式，{dataset_path} 目录下没有 Annotations 或 annotations")
    return None