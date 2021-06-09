import os
import mlflow
mlflow.set_tracking_uri("http://192.168.64.22:5002")
mlflow.set_experiment("train-trail")

"""
(1) Data preparing
"""
from data_gen_and_train.utils.voc2yolo4 import *
from data_gen_and_train.utils.voc_annotation import *

xmlfilepath = r'./input/dataset/labels/'
saveBasePath = r"./input/dataset/"
anno_file = "./data_gen_and_train/param_files/init_train.txt"

voc2yolo(xmlfilepath, saveBasePath)
gen_label_txt(xmlfilepath, saveBasePath, anno_file)

print("Finished training set transformation.")

saveBasePath = r"./input/test/"
xmlfilepath = r'./input/test/labels/'
if os.path.exists(xmlfilepath):
    anno_file = "./data_gen_and_train/param_files/test.txt"
    voc2yolo(xmlfilepath, saveBasePath)
    gen_label_txt(xmlfilepath, saveBasePath, anno_file)

print("Finished evaluating set transformation.")


"""
(2) Data argumentation
"""
from data_gen_and_train.gen_data.generator import Generator
input_file = "./data_gen_and_train/param_files/init_train.txt"
output_file = "./data_gen_and_train/param_files/expanded_train.txt"

print("Generating data...")
gen = Generator(input_file)
gen.gen_hsv_tuning(output_file)             
# gen.gen_fancyPCA_tuning(output_file)     
# gen.gen_scale_transform(output_file)    
# gen.gen_bright_transform(output_file)   
# gen.gen_flip_transform(output_file)        
gen.gen_rotate_transform(output_file)    

print("Finished data argumentaion.")


"""
(3) Training
"""
from data_gen_and_train.train import Trainer

if __name__ == "__main__":

    with mlflow.start_run():
        
        # log parameters into mlflow
        mlflow.log_param("project_root", "192.168.64.22:/data1/yolov4-train/")
        mlflow.log_param("dataset_path", "192.168.64.22:/data5/dataset/")
        mlflow.log_param("ckpt_path", "./data_gen_and_train/ckpt/")
        mlflow.log_param("freeze_batch_size", "8")
        mlflow.log_param("freeze_epoch", "50")
        mlflow.log_param("freeze_learning_rate", "1e-3")
        mlflow.log_param("batch_size", "4")
        mlflow.log_param("total_epoch","250")
        mlflow.log_param("learning_rate", "2e-4")
        mlflow.log_param("tricks", "mosaic, no cosine_lr, no smooth_label")
        mlflow.log_param("input_size", "416")
        

        model = Trainer(
            annotation_path="./data_gen_and_train/param_files/expanded_train.txt",
            classes_path="./data_gen_and_train/param_files/classes.name",
            ckpt_path="./data_gen_and_train/ckpt/",
            freeze_bn_size=8,          
            freeze_epoch=50,            
            freeze_learning_rate=1e-3,         
            bn_size=4,  
            total_epoch=250,           
            learning_rate=2e-4,        
            cosine_lr=False,   
            mosaic=True,     
            smooth_label=0,       
            input_size=416          
        )

        model.train()
        mlflow.log_artifacts("./data_gen_and_train/ckpt/")

        print("Finished training model .")

