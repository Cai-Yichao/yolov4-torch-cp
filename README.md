# 拷贝版YOLOv4
模型核心代码拷贝于： https://github.com/bubbliiiing/yolov4-pytorch
____

## 1 数据准备

**Step 1** 利用LabelImg标注数据

**Step 2** 在./input/dataset路径下新建images和labels文件夹，将图像拷贝到images目录，将相应的xml文件拷贝到labels目录。

**Step 3** 下载预训练权重（链接来自 https://github.com/bubbliiiing/yolov4-pytorch）

链接: https://pan.baidu.com/s/1VNSYi39AaqjHVbdNpW_7sw 提取码: q2iv

（yolo4_weights.pth是coco数据集的权重。
yolo4_voc_weights.pth是voc数据集的权重。）

将下载好的文件保存到，./data_gen_and_train/param_files目录。

**Step 4** 将数据集标签类别，写到./data_gen_and_train/param_files/classes.name文件中。

**Step 5** 修改./data_gen_and_train/utils/voc_annotation.py文件，第7行的classes列表为数据集的标签类别。

____
## 2 训练

运行train_script.py 脚本：

```shell
nohup python -u train_script.py &
```
**Note 1:** 脚本第(2)部分增加了一些数据集扩充的方法。

如果需要使用，将调用的第1个生成方法中的这一行注释取消，否则会训练时会丢失原始数据（若第1个方法是gen_hsv_tuning时，不用修改）。
```python
# w_f.writelines(self.line_list)
```
如果不需要就将脚本第(2) 部分全部注释掉，并且把第47行的expanded_train.txt改为init_train.txt。

**Note 2:** 第(3)部分的模型训练参数控制训练过程：
``` python
 annotation_path   # 训练数据文件，根据Note1决定是否修改
 classes_path    # 标签类别文件，无需修改
 ckpt_path      # 权重保存文件，默认ckpt，可修改。
 freeze_bn_size  # 冻结时批大小，根据显存修改
 freeze_epoch    # 冻结训练epoch
 freeze_learning_rate  # 冻结时初始学习率
 bn_size         # 解冻后批大小，若冻结批大小拉满显存，这里大小要为小于其一半
 total_epoch        # 总的训练epoch(包含冻结epoch)
 learning_rate       # 解冻后初始学习率
cosine_lr    # 是否使用余弦学习率，默认False
mosaic      # 是否使用mosaic增强，默认True
smooth_label       # 是否使用标签平滑，默认0
 input_size           # 网络输入大小，416或608, 默认416
```

____
## 3 绘制LOSS曲线
在训练结束后或者训练过程中，可以绘制Loss曲线：
```shell
cd ./draw_loss
python draw_loss.py
输入权重文件目录(如ckpt): ckpt
```
loss曲线会保存至./data_gen_and_train目录下面。

为了显示效果，绘制从第15个epoch开始，如需调整，修改./draw_loss/draw_loss.py的27, 28, 29行。 

## 4 评价
评价代码在./validation下面。

**Step 1** 修改脚本

（1）若训练时未使用数据扩充，修改1_get_val_txt.py中第9行expanded_train.txt为init_train.txt

（2）修改3_get_val_predict.py中23行一下的变量：
```shell
imgs_path   # 数据集图像保存路径
model_path  # 评价的目标权重路径
image_size # 网络输入大小，和训练保持一致
```
**Step 2** 依次运行：
```shell
cd ./validation
python 1_get_val_txt.py
python 2_get_val_gt.py
python 3_get_val_predict.py
python 4_get_map.py
```
完毕后，在./validation/results下面会生成相应的指标可视化图像。

## 5 推理
（1）修改predict.py第9行路径，改为待测试图像的存放路径
（2）修改predict.py第21行的实例化参数：
```shell
image_size # 网络输入大小，和训练保持一致
model_path  # 评价的目标权重路径
confidence  # 识别输出的置信阈值
```
（3）运行predict.py

执行完毕后，在./data_gen_and_train/predicts/results中会生成推理结果的可视化图像，文件名和原始图像一一对应。

## 声明
本项目的算法核心代码来源于 https://github.com/bubbliiiing/yolov4-pytorch
我只是调整和添加了一些代码。如有不合适之处，联系我删除。