# -*- coding:UTF-8 -*-
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

path = "../data_gen_and_train/"
path = path + input("输入权重文件目录(如ckpt):")
file_list = os.listdir(path)
# print(file_list)
total_loss_dict = []
val_loss_dict = []
for file_str in file_list:
    str_list = file_str.strip(".pth").split("-")
    epoch = int(str_list[0][5:])
    total_loss = float(str_list[1][10:])
    val_loss = float(str_list[2][8:])
    total_loss_dict.append([epoch, total_loss])
    val_loss_dict.append([epoch, val_loss])

# 按照epoch排序
total_loss_dict = np.array(sorted(total_loss_dict, key=lambda x:x[0]))
val_loss_dict = np.array(sorted(val_loss_dict, key=lambda x:x[0]))
# print(total_loss_dict)

epoch = np.array(total_loss_dict[15:, 0], dtype=int)
total_loss = total_loss_dict[15:, 1]
val_loss = val_loss_dict[15:, 1]

# 绘制
plt.plot(epoch, total_loss, linewidth=2)
plt.title("train loss", fontsize=16)
plt.xlabel("epoch", fontsize=10)
plt.ylabel("loss", fontsize=10)
plt.tick_params(axis='both', labelsize=10)
plt.savefig(path+"_train_loss.png")
plt.close('all')

plt.figure()
plt.plot(epoch, val_loss, linewidth=2)
plt.title("val loss", fontsize=16)
plt.xlabel("epoch", fontsize=10)
plt.ylabel("loss", fontsize=10)
plt.tick_params(axis='both', labelsize=10)
plt.savefig(path+"_val_loss.png")
plt.close('all')
