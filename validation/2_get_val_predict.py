"""
step 2:
获取评价的检测结果
"""
import sys
sys.path.append("..")
import cv2
import numpy as np
import colorsys
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data_gen_and_train.yolo import YOLO
from data_gen_and_train.nets.yolo4 import YoloBody
from PIL import Image,ImageFont, ImageDraw
from data_gen_and_train.utils.utils import non_max_suppression, \
    bbox_iou, DecodeBox,letterbox_image,yolo_correct_boxes
from tqdm import tqdm

# global values
imgs_path = "../input/test/images/"
model_path = "../data_gen_and_train/ckpt_no_mosaic/Epoch246-Total_Loss0.0925-Val_Loss0.2497.pth"
cls_path = "../data_gen_and_train/param_files/classes.name"
anchors_path = "../data_gen_and_train/param_files/yolo_anchors.txt"
font_path = "../data_gen_and_train/param_files/simhei.ttf"
image_size = 416

class mAP_Yolo(YOLO):
    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#

    def detect_image(self,image_id,image):
        self.confidence = 0.01
        f = open("detection-results/"+image_id+".txt","w")
        image_shape = np.array(np.shape(image)[0:2])

        crop_img = np.array(letterbox_image(image, (self.model_image_size[0],self.model_image_size[1])))
        photo = np.array(crop_img,dtype = np.float32)
        photo /= 255.0
        photo = np.transpose(photo, (2, 0, 1))
        photo = photo.astype(np.float32)
        images = []
        images.append(photo)
        images = np.asarray(images)

        with torch.no_grad():
            images = torch.from_numpy(images)
            if self.cuda:
                images = images.cuda()
            outputs = self.net(images)

        output_list = []
        for i in range(3):
            output_list.append(self.yolo_decodes[i](outputs[i]))
        output = torch.cat(output_list, 1)
        batch_detections = non_max_suppression(output, len(self.class_names),
                                                conf_thres=self.confidence,
                                                nms_thres=self.iou)

        try:
            batch_detections = batch_detections[0].cpu().numpy()
        except:
            return image

        top_index = batch_detections[:,4]*batch_detections[:,5] > self.confidence
        top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]
        top_label = np.array(batch_detections[top_index,-1],np.int32)
        top_bboxes = np.array(batch_detections[top_index,:4])
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

        # 去掉灰条
        boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]),image_shape)

        for i, c in enumerate(top_label):
            predicted_class = self.class_names[c]
            score = str(top_conf[i])

            top, left, bottom, right = boxes[i]
            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return

"""
实例化检测器
"""
yolo = mAP_Yolo(
    img_size = image_size,
    model_path =model_path,
    cls_path =cls_path,
    anchors_path =anchors_path,
    font_path = font_path,
    confidence = 0.56
)

img_ids = []
with open('../input/test/trainval.txt', 'r') as file_r:
    for line in file_r.readlines():
        img_path = line.strip('\n').split(' ')[0]
        img_id = img_path.split('/')[-1].strip('.jpg')
        img_ids.append(img_id)
    file_r.close()

if not os.path.exists("detection-results"):
    os.makedirs("detection-results")

for image_id in tqdm(img_ids):
    image_path = imgs_path + image_id+".jpg"
    image = Image.open(image_path)
    yolo.detect_image(image_id,image)
print("Conversion completed!")

