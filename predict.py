import os
import json
from data_gen_and_train.yolo import YOLO
from PIL import Image
from tqdm import tqdm


# 获取图像路径
test_path = "./input/test_images/"
img_names = os.listdir(test_path)
print(img_names)

# 创建图像和结果保存目录
target_path = "./data_gen_and_train/results/predicts/"
if not os.path.exists("./data_gen_and_train/results/"):
    os.mkdir("./data_gen_and_train/results/")
if not os.path.exists(target_path):
    os.mkdir(target_path)

# 实例化检测器
detector = YOLO(img_size=416,
                model_path="./data_gen_and_train/ckpt_jyz/Epoch232-Total_Loss1.5223-Val_Loss3.6240.pth",
                cls_path="./data_gen_and_train/param_files/classes.name",
                anchors_path="./data_gen_and_train/param_files/yolo_anchors.txt",
                font_path="./data_gen_and_train/param_files/simhei.ttf",
                confidence=0.6)

# 推理结果
for img_name in tqdm(img_names):
    img_file = test_path + img_name
    image = Image.open(img_file)
    tar_img, boxes, labels, _ = detector.detect_image_draw(image)

    # 保存可视化图像
    tar_file = target_path + img_name
    tar_img.save(tar_file)
print("Done...")
