def auto_annotation(dataset_path, classes_path):
    import os
    if os.path.exists(os.path.join(dataset_path, 'Annotations')):
        from utils_voc.voc_annotation import run_voc_annotation
        run_voc_annotation(dataset_path, classes_path)
        return "VOC"
    
    elif os.path.exists(os.path.join(dataset_path, 'annotations')):
        from utils_coco.coco_annotation import run_coco_annotation
        run_coco_annotation(dataset_path)
        return "COCO"
    
    return None