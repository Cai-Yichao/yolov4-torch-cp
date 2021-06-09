import xml.etree.ElementTree as ET
from os import getcwd

sets = [('2007', 'train')]

wd = getcwd()
classes = ["name"]


def convert_annotation(xml_path, image_id, list_file):
    in_file = xml_path + image_id + ".xml"
    tree = ET.parse(in_file)
    root = tree.getroot()
    file_name = root.find('filename').text
    file_name = xml_path.replace("label", "image") + file_name
    list_file.write(file_name)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text),
             int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))

    list_file.write('\n')


def gen_label_txt(xml_path, input_path, anno_file):
    for year, image_set in sets:
        file_name = input_path + image_set + ".txt"
        image_ids = open(file_name).read().strip().split()
        list_file = open(anno_file, 'w')
        for image_id in image_ids:
            convert_annotation(xml_path, image_id, list_file)
        list_file.close()
