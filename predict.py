from PIL import Image
from yolo4use import YOLO
from config import *

if __name__ == "__main__":

    yolo = YOLO(
        model_path = MODEL_PATH,
        classes_path = CLASSES_PATH,
        input_shape = INPUT_SHAPE,
        scales = SCALES,
        confidence = CONFIDENCE,
        nms_iou = NMS_IOU,
        letterbox_image = LETTERBOX_IMAGE,
        cuda = CUDA
    )

    while True:
        img = input('Input image filename:')
        try:
            image = Image.open(img)
        except:
            print('Open Error! Try again!')
            continue
        else:
            r_image = yolo.detect_image(image, crop = CROP, count = COUNT)
            r_image.show()