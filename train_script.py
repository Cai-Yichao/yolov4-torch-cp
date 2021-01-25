# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

"""
(1)转换VOC数据格式，方便数据生成以及模型训练时读取加载数据:
"""
from data_gen_and_train.utils.voc2yolo4 import *
from data_gen_and_train.utils.voc_annotation import *

xmlfilepath = r'./input/dataset/labels/'
saveBasePath = r"./input/dataset/"
anno_file = "./data_gen_and_train/param_files/init_train.txt"

voc2yolo(xmlfilepath, saveBasePath)
gen_label_txt(xmlfilepath, saveBasePath, anno_file)

print("Finished format transformation.")

# 如果准备了评价数据
saveBasePath = r"./input/test/"
xmlfilepath = r'./input/test/labels/'
if os.path.exists(xmlfilepath):
    anno_file = "./data_gen_and_train/param_files/test.txt"
    voc2yolo(xmlfilepath, saveBasePath)
    gen_label_txt(xmlfilepath, saveBasePath, anno_file)

print("Finished format transformation.")


"""
(2)生成数据，扩充数据集
"""
from data_gen_and_train.gen_data.generator import Generator
input_file = "./data_gen_and_train/param_files/init_train.txt"
output_file = "./data_gen_and_train/param_files/expanded_train.txt"
gen = Generator(input_file)
gen.gen_hsv_tuning(output_file)          # 色域变换，扩充基础数量
# gen.gen_fancyPCA_tuning(output_file)     # facny PCA
# gen.gen_scale_transform(output_file)     # 尺度变换
gen.gen_bright_transform(output_file)    # 亮度变换
# gen.gen_flip_transform(output_file)      # 翻转位置变换
gen.gen_rotate_transform(output_file)    # 旋转位置变换

print("Finished data_generation.")


"""
(3) 模型训练
"""
from data_gen_and_train.train import Trainer

if __name__ == "__main__":
    """
    训练检测模型
    """
    model = Trainer(
        annotation_path="./data_gen_and_train/param_files/expanded_train.txt",
        classes_path="./data_gen_and_train/param_files/classes.name",
        ckpt_path="./data_gen_and_train/ckpt/",
        freeze_bn_size=32,          
        freeze_epoch=50,            # 冻结权重epoch
        freeze_learning_rate=1e-3,          # 冻结时初始学习率
        bn_size=8,  
        total_epoch=250,            # 总的训练epoch
        learning_rate=5e-4,         # 解冻后初始学习率
        cosine_lr=False,    # 是否使用余弦学习率，默认False
        mosaic=True,      # 是否使用mosaic增强，默认True
        smooth_label=0,       # 是否使用标签平滑，默认0
        input_size=416          # 输入尺寸，默认416
    )

    model.train()
    print("Finished training model .")

