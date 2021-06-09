# PyTorch YOLOv4
The core codes are reproduced from https://github.com/bubbliiiing/yolov4-pytorch , based on which some changes are implemented to make it easier to use.

____
## 1 Data preparing

**Step 1** 

Label you data with LabelImg in VOC fromat, then divide them into two parts, training set and evaluating set.

**Step 2** 

「training set」： Make directories in **./input/dataset** and name them with **images** and **labels**, put the images into **images** and the xml files into **labels**.

「evaluating set」： Make directories in **./input/dataset** and name them with **images** and **labels**, and put in data the same way of making traning set. 

**Step 3** 

Download pretrained checkpoint from https://pan.baidu.com/s/1VNSYi39AaqjHVbdNpW_7sw   [password: q2iv] （the url is from https://github.com/bubbliiiing/yolov4-pytorch）

（N.B. yolo4_weights.pth was trained with COCO， yolo4_voc_weights.pth was trained with VOC.）

Save the chekpoint into **./data_gen_and_train/param_files**.

**Step 4** 

Write the classes's names into **./data_gen_and_train/param_files/classes.name**.

**Step 5** 

Modify the elements in the list named **classes** with class names of your dataset, you can find it on **line 7** in **./data_gen_and_train/utils/voc_annotation.py**.

____
## 2 Training

Run **train_script.py** with command：

```shell
nohup python -u train_script.py &
```
**N.B. 1** Section (2) of the script are some methods for data argumentation.

If you want to use some of those，cancel the annotation before this line (see below) of the first method you'll use, other wise the original data won't be involved in the training process.If the first method is **gen_hsv_tuning** as it is，you don't have to to this.
```python
# w_f.writelines(self.line_list)
```
If you don't want to do data argumentation，juest simply annote all codes of section (2), and assign **""init_train.txt""** to **annotation_path** on line 75.

**N.B. 2** The parameters to init Trainer are defined as bellow：
``` python
 annotation_path      # training data, see N.B. 1
 classes_path         # path of classes.name, keep it unchanged
 ckpt_path            # path to save ckpoints
 freeze_bn_size       # batch size when the model partially frozen 
 freeze_epoch         # epochs of training with the model partially frozen 
 freeze_learning_rate # initial learning rate
 bn_size              # batch size when training the whole model. 
 total_epoch          # total epoch (including freeze_epoch)
 learning_rate        # initial learning rate of unfrozen training
 cosine_lr            # if use consine learning trick, True or False
 mosaic               # if use mosaic trick, True or False
 smooth_label         # if use label smoothing, 0 or 1
 input_size           # input size，416 or 608; 416 by default
```

Normally, **bn_size** should to smaller than **freeze_bn_size** to avoid OOM. Before training, you can set **freeze_epoch = 0** to figure out a felicitous **bn_size**.

____
## 3 Use mlflow tracking your training process

To record parameters like learning rate, bn_size and dataset path, use:
``` python
mlflow.log_param("key", value)
```
To track dynamic information like loss, accuracy, use:
``` python
mlflow.log_param("key", value, iteration)
```

To save data and result to database, use:
``` python
mlflow.log_artifacts("path")
```

____
## 4 Evaluating
codes under **./validation**

**Step 1** 

Change variants below, which can be find in **2_get_val_predict.py, line 23**：
```python
model_path  # model path to be evaluated
image_size  # inputsize, same with the training configuration
```
**Step 2** 

Run these scripts：
```shell
cd ./validation
python 1_get_val_gt.py
python 2_get_val_predict.py
python 3_get_map.py
```
After which，results can be find in **./validation/results**.

____
## 5 Predict
（1）Modify the path of images to be infered **on line 10 in predict.py**.

（2）Change variants below, in **predict.py, line 22**:
```python
image_size    # inputsize, same with the training configuration
model_path    # model path
confidence    # confidence threshold
```
（3）run predict.py

The results will be written into **./data_gen_and_train/predicts/results**.
