"""
step 1:
获取评价样本
"""
import  os
import numpy as np

val_split = 0.1
annotation_file = "../data_gen_and_train/param_files/expanded_train.txt"

with open(annotation_file, 'r') as file_r:
    with open("validation.txt", 'w') as file_w:
        lines = file_r.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines) * val_split)
        num_train = len(lines) - num_val
        for i in range(num_train, len(lines)):
            file_w.write(lines[i])
    file_w.close()
file_r.close()
print("Done...")




