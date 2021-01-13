# -*- coding: UTF-8 -*-
"""
2020_11_03
"""

import cv2 as cv
# from gen_data.hsv_tuning import hsv_tuning
from PIL import Image
import numpy as np
import math
from data_gen_and_train.gen_data.hsv_tuning import hsv_tuning
from data_gen_and_train.gen_data.fancy_pca import fancy_pca
from data_gen_and_train.gen_data.imageprocess import *


def parse_info_line(in_str):
    """
    读取文件中一行的信息
    :param in_str: 文件的一行
    :return: 图像路径，绝缘子标注
    """
    info_list = in_str.split(' ')
    img_path = info_list[0]
    jyz_boxes = []
    zb_boxes = []

    for i in range(1, len(info_list)):
        list_tmp = info_list[i].split(',')
        box = [int(list_tmp[0]), int(list_tmp[1]), int(list_tmp[2]),
               int(list_tmp[3])]
        if list_tmp[-1] == '0':
            jyz_boxes.append(box)
        else:
            zb_boxes.append(box)

    return img_path, jyz_boxes, zb_boxes


def angle2point(angle, image_shape, x, y):
    angle_pi = angle * math.pi / 180
    # 改变后的坐标宽高需要调换
    h = image_shape[0]
    w = image_shape[1]
    # 注意这里转换后得到的是整个翻转后未去除边缘的地方，也就是其余部分填充即最大边的矩形。
    # 所以要减掉这一部分才是本文所用方法的坐标位置。
    if w >= h:
        #print('w>=h')
        x_rotate = int(int(x) * math.cos(angle_pi) - int(y) * math.sin(angle_pi) -
                       0.5 * w * math.cos(angle_pi) + 0.5 * h * math.sin(angle_pi) + 0.5 * w - abs(
            math.ceil((w - h) / 2)))
        y_rotate = int(int(x) * math.sin(angle_pi) + int(y) * math.cos(angle_pi) -
                       0.5 * w * math.sin(angle_pi) - 0.5 * h * math.cos(angle_pi) + 0.5 * h + abs(
            math.ceil((w - h) / 2)))
    else:
        #print('w<h')
        x_rotate = int(int(x) * math.cos(angle_pi) - int(y) * math.sin(angle_pi) -
                       0.5 * w * math.cos(angle_pi) + 0.5 * h * math.sin(angle_pi) + 0.5 * w + abs(
            math.ceil((w - h) / 2)))
        y_rotate = int(int(x) * math.sin(angle_pi) + int(y) * math.cos(angle_pi) -
                       0.5 * w * math.sin(angle_pi) - 0.5 * h * math.cos(angle_pi) + 0.5 * h - abs(
            math.ceil((w - h) / 2)))
    if x_rotate < 0:
        x_rotate = 0
    if y_rotate < 0:
        y_rotate = 0
    return int(x_rotate), int(y_rotate)


class Generator:
    """
    图像生成
    """

    def __init__(self, info_file):
        self.info_file = info_file
        self.img_infos, self.line_list = self.__loader()

    def __loader(self):
        images = []
        with open(self.info_file, 'r') as f_r:
            line_list = f_r.readlines()
            for line in line_list:
                line = line.strip("\n")
                tmp_dict = {"path": parse_info_line(line)[0], "jyz": parse_info_line(line)[1],
                            "zb": parse_info_line(line)[2]}
                images.append(tmp_dict)
        return images, line_list

    @staticmethod
    def string2write(img_path, jyz, zb):
        str_tmp = img_path + " "
        for i in jyz:
            str_tmp += str(int(i[0])) + "," + str(int(i[1])) \
                       + "," + str(int(i[2])) + "," + str(int(i[3])) + ",0 "
        for i in zb:
            str_tmp += str(int(i[0])) + "," + str(int(i[1])) \
                       + "," + str(int(i[2])) + "," + str(int(i[3])) + ",1 "
        str_tmp = str_tmp.strip(" ") + "\n"
        return str_tmp

    @staticmethod
    def change_coordinate(transform_type, param, jyz, zb, image_shape=None):
        if transform_type == 'scale':
            new_jyz = []
            new_zb = []
            for i in jyz:
                new_jyz.append([round(x * param) for x in i])
            for i in zb:
                new_zb.append([round(x * param) for x in i])
        elif transform_type == 'flip':
            new_jyz = []
            new_zb = []
            if param == -1:  # 对角镜像翻转
                for i in jyz:
                    y_start = image_shape[0] - i[1] - 1
                    y_end = image_shape[0] - i[3] - 1
                    x_start = image_shape[1] - i[0] - 1
                    x_end = image_shape[1] - i[2] - 1
                    if x_start < 0:
                        x_start = 0
                    if x_end < 0:
                        x_end = 0
                    if y_start < 0:
                        y_start = 0
                    if y_end < 0:
                        y_end = 0
                    if x_start > x_end:
                        temp = x_start
                        x_start = x_end
                        x_end = temp
                        temp = y_start
                        y_start = y_end
                        y_end = temp
                    new_jyz.append([x_start, y_start, x_end, y_end])
                for i in zb:
                    y_start = image_shape[0] - i[1] - 1
                    y_end = image_shape[0] - i[3] - 1
                    x_start = image_shape[1] - i[0] - 1
                    x_end = image_shape[1] - i[2] - 1
                    if x_start < 0:
                        x_start = 0
                    if x_end < 0:
                        x_end = 0
                    if y_start < 0:
                        y_start = 0
                    if y_end < 0:
                        y_end = 0
                    if x_start > x_end:
                        temp = x_start
                        x_start = x_end
                        x_end = temp
                        temp = y_start
                        y_start = y_end
                        y_end = temp
                    new_zb.append([x_start, y_start, x_end, y_end])
            elif param == 0:  # 垂直镜像翻转
                for i in jyz:
                    y_start = image_shape[0] - i[1] - 1
                    y_end = image_shape[0] - i[3] - 1
                    if y_start < 0:
                        y_start = 0
                    if y_end < 0:
                        y_end = 0
                    x_start = i[0]
                    x_end = i[2]
                    if x_start > x_end:
                        temp = x_start
                        x_start = x_end
                        x_end = temp
                        temp = y_start
                        y_start = y_end
                        y_end = temp
                    new_jyz.append([x_start, y_end, x_end, y_start])
                for i in zb:
                    y_start = image_shape[0] - i[1] - 1
                    y_end = image_shape[0] - i[3] - 1
                    if y_start < 0:
                        y_start = 0
                    if y_end < 0:
                        y_end = 0
                    x_start = i[0]
                    x_end = i[2]
                    if x_start > x_end:
                        temp = x_start
                        x_start = x_end
                        x_end = temp
                        temp = y_start
                        y_start = y_end
                        y_end = temp
                    new_zb.append([x_start, y_end, x_end, y_start])
            elif param == 1:  # 水平镜像翻转
                for i in jyz:
                    x_start = image_shape[1] - i[0] - 1
                    x_end = image_shape[1] - i[2] - 1
                    if x_start < 0:
                        x_start = 0
                    if x_end < 0:
                        x_end = 0
                    y_start = i[1]
                    y_end = i[3]
                    if x_start > x_end:
                        temp = x_start
                        x_start = x_end
                        x_end = temp
                        temp = y_start
                        y_start = y_end
                        y_end = temp
                    new_jyz.append([x_start, y_end, x_end, y_start])
                for i in zb:
                    x_start = image_shape[1] - i[0] - 1
                    x_end = image_shape[1] - i[2] - 1
                    if x_start < 0:
                        x_start = 0
                    if x_end < 0:
                        x_end = 0
                    y_start = i[1]
                    y_end = i[3]
                    if x_start > x_end:
                        temp = x_start
                        x_start = x_end
                        x_end = temp
                        temp = y_start
                        y_start = y_end
                        y_end = temp
                    new_zb.append([x_start, y_end, x_end, y_start])
        elif transform_type == 'rotate':
            new_jyz = []
            new_zb = []
            for i in jyz:
                x_start, y_start = angle2point(param, image_shape, i[0], i[1])
                x_end, y_end = angle2point(param, image_shape, i[2], i[3])
                if x_start > x_end:
                    temp = x_start
                    x_start = x_end
                    x_end = temp
                    temp = y_start
                    y_start = y_end
                    y_end = temp
                new_jyz.append([x_start, y_end, x_end, y_start])
            for i in zb:
                x_start, y_start = angle2point(param, image_shape, i[0], i[1])
                x_end, y_end = angle2point(param, image_shape, i[2], i[3])
                if x_start > x_end:
                    temp = x_start
                    x_start = x_end
                    x_end = temp
                    temp = y_start
                    y_start = y_end
                    y_end = temp
                new_zb.append([x_start, y_end, x_end, y_start])
        return new_jyz, new_zb

    def gen_hsv_tuning(self, out_file):
        """通过色域变换生成图像"""
        with open(out_file, 'w') as w_f:
            # 调节参数
            param_list = [[0.056880833629979394, 0.7042660496969584, 1.1621097752699185],
                          [0.04167807794212616, 1.296081632448806, 0.957791503108028],
                          [0.08304129276424002, 1.2840055684965987, 0.7742919291583074],
                          [0.04060578575492435, 0.7099565675278529, 0.8675518080241147]]
            for index, info in enumerate(self.img_infos):
                print(info["path"], "----------------")
                src_img = cv.imread(info["path"])
                src_name = info["path"].split('/')[-1]
                for i, param in enumerate(param_list):
                    dst_image = hsv_tuning(src_img, param[0], param[1], param[2])  # 色域变换
                    dst_name = "./input/helmet/xml_images/hsv_" + str("%s_" % i) + src_name
                    cv.imwrite(dst_name, dst_image)  # 保存图像
                    str_tmp = self.string2write(dst_name, info["jyz"], info["zb"])
                    w_f.write(str_tmp)  # 保存标签
            w_f.writelines(self.line_list)
            w_f.close()

    def gen_fancyPCA_tuning(self, outfile):
        """通过fancy_PCA方法生成图像"""
        with open(outfile, 'a') as w_f:
            for index, info in enumerate(self.img_infos):
                src_img = Image.open(info["path"])  # 用PIL中的Image.open打开图像
                src_name = info["path"].split('/')[-1]
                image_arr = np.array(src_img)  # 转化成numpy数组
                thres = [-100, -50, 50, 100]
                for i in range(len(thres)):
                    dst, alpha = fancy_pca(image_arr, thres[i])
                    im = Image.fromarray(dst)
                    dst_name = "./input/helmet/xml_images/fancyPCA_" + str("%s_" % i) + src_name
                    print(dst_name)
                    im.save(dst_name)
                    str_tmp = self.string2write(dst_name, info["jyz"], info["zb"])
                    w_f.write(str_tmp)  # 保存标签
                print('已完成第{}张图像生成。'.format(index + 1))
            # w_f.writelines(self.line_list)
            w_f.close()

    def gen_scale_transform(self, out_file):
        """通过尺度变换生成图像（1/2和1/4图像）"""
        with open(out_file, 'a') as w_f:
            for index, info in enumerate(self.img_infos):
                src_img = cv.imread(info["path"])
                src_name = info["path"].split('/')[-1]
                for i in range(2):
                    scale = 1 / ((i + 1) * 2)
                    dst_image = scale_transform(src_img, scale)  # 尺度变换
                    dst_name = "./input/helmet/xml_images/scale_" + str("%s_" % i) + src_name
                    cv.imwrite(dst_name, dst_image)  # 保存图像
                    # 修改坐标
                    jyz, zb = self.change_coordinate('scale', scale, info["jyz"], info["zb"])
                    # 写入文件
                    str_tmp = self.string2write(dst_name, jyz, zb)  # 需要修改坐标
                    w_f.write(str_tmp)  # 保存标签
                print('已完成第{}张图像生成。'.format(index + 1))
            #w_f.writelines(self.line_list)
            w_f.close()

    def gen_bright_transform(self, out_file):
        """通过图像对比度亮度变换生成图像"""
        with open(out_file, 'a') as w_f:
            for index, info in enumerate(self.img_infos):
                src_img = cv.imread(info["path"])
                src_name = info["path"].split('/')[-1]
                beta = [30, 60, -30, -60]
                for i in range(len(beta)):
                    dst_image = bright_transform(src_img, beta=beta[i])
                    dst_name = "./input/helmet/xml_images/bright_" + str("%s_" % beta[i]) + src_name
                    cv.imwrite(dst_name, dst_image)  # 保存图像
                    str_tmp = self.string2write(dst_name, info["jyz"], info["zb"])
                    w_f.write(str_tmp)  # 保存标签
                print('已完成第{}张图像生成。'.format(index + 1))
            #w_f.writelines(self.line_list)
            w_f.close()

    def gen_flip_transform(self, out_file):
        """通过图像镜像翻转变换生成图像"""
        with open(out_file, 'a') as w_f:
            for index, info in enumerate(self.img_infos):
                src_img = cv.imread(info["path"])
                src_name = info["path"].split('/')[-1]
                beta = [-1, 0, 1]
                for i in range(len(beta)):
                    dst_image = flip_transform(src_img, flip_param=beta[i])
                    dst_name = "./input/helmet/xml_images/flip_" + str("%s_" % beta[i]) + src_name
                    cv.imwrite(dst_name, dst_image)  # 保存图像
                    jyz, zb = self.change_coordinate('flip', beta[i], info["jyz"], info["zb"],
                                                     image_shape=src_img.shape[:2])
                    str_tmp = self.string2write(dst_name, jyz, zb)  # 需要修改坐标
                    w_f.write(str_tmp)  # 保存标签
                print('已完成第{}张图像生成。'.format(index + 1))
            #w_f.writelines(self.line_list)
            w_f.close()

    def gen_rotate_transform(self, out_file):
        """通过图像空间-顺时针旋转90度变换生成图像"""
        with open(out_file, 'a') as w_f:
            for index, info in enumerate(self.img_infos):
                src_img = cv.imread(info["path"])
                src_name = info["path"].split('/')[-1]
                dst_image = rotate_transform(src_img)
                dst_name = "./input/helmet/xml_images/rotate_90_" + src_name
                cv.imwrite(dst_name, dst_image)  # 保存图像
                param = 90
                jyz, zb = self.change_coordinate('rotate', param, info["jyz"], info["zb"],
                                                 image_shape=src_img.shape[:2])
                str_tmp = self.string2write(dst_name, jyz, zb)  # 需要修改坐标
                w_f.write(str_tmp)  # 保存标签
                print('已完成第{}张图像生成。'.format(index + 1))
            #w_f.writelines(self.line_list)
            w_f.close()
