"""
step 1:
获取评价的ground truth
"""
import sys
import os
import glob
import xml.etree.ElementTree as ET
from tqdm import tqdm

classes_file = "../data_gen_and_train/param_files/classes.name"
annotation_file = "../data_gen_and_train/param_files/test.txt"

def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

if not os.path.exists("ground-truth"):
    os.makedirs("ground-truth")

cls_names = get_class(classes_file)
with open(annotation_file, 'r') as file_r:
    lines = file_r.readlines()
    for line in tqdm(lines):
        line = line.strip('\n').split()
        print(line)
        img_path = line[0]
        objs = line[1:]
        img_id = img_path.split('/')[-1].strip('.jpg')
        with open("ground-truth/" + img_id + ".txt", "w") as new_f:
            for obj in objs:
                infos = obj.split(',')
                obj_name = cls_names[int(infos[-1])]
                new_f.write("%s %s %s %s %s\n" % (obj_name, infos[0], infos[1], infos[2], infos[3]))
        new_f.close()
    file_r.close()
print("Conversion completed!")
