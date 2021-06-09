# -------------------------------------#
#       对数据集进行训练
# -------------------------------------#
import os
import time
import mlflow
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from data_gen_and_train.nets.yolo4 import YoloBody
from data_gen_and_train.nets.yolo_training import YOLOLoss, Generator
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_gen_and_train.utils.dataloader import yolo_dataset_collate, YoloDataset


# ---------------------------------------------------#
#   获得类和先验框
# ---------------------------------------------------#
def get_classes(classes_path):
    """loads the classes"""
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    """loads the anchors from a file"""
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape([-1, 3, 2])[::-1, :, :]


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


class Trainer:
    """
    Trainning Data with YOLOv4
    """

    def __init__(self, annotation_path, classes_path, ckpt_path, freeze_bn_size, freeze_epoch, freeze_learning_rate,
                 bn_size, total_epoch, learning_rate, cosine_lr=False, mosaic=True, smooth_label=0,  input_size=416):
        self.input_shape = (input_size, input_size)
        self.Cosine_lr = cosine_lr
        self.mosaic = mosaic
        self.smooth_label = smooth_label
        self.Use_Data_Loader = True

        self.annotation_path = annotation_path
        self.class_names = get_classes(classes_path)
        self.anchors = get_anchors('./data_gen_and_train/param_files/yolo_anchors.txt')
        self.num_classes = len(self.class_names)
        self.comare_list = [999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999, 999999]
        self.epoch_list = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]

        self.ckpt_path = ckpt_path
        if not os.path.exists(self.ckpt_path):
            os.mkdir(self.ckpt_path)

        self.freeze_bnsize = freeze_bn_size
        self.freeze_epoch = freeze_epoch
        self.freeze_lr = freeze_learning_rate
        self.bn_size = bn_size
        self.total_epoch = total_epoch
        self.lr = learning_rate

        # 用于设定是否使用cuda
        if torch.cuda.is_available():
            self.Cuda = True
            self.device = torch.device('cuda')
        else:
            self.Cuda = False
            self.device = torch.device('cpu')

        # 创建模型
        print('Loading weights into state dict...')
        self.model = YoloBody(len(self.anchors[0]), self.num_classes)
        model_path = "./data_gen_and_train/param_files/yolo4_weights.pth"
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=self.device)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        print('Finished loading pre-trained model!')

    def fit_one_epoch(self, net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, optimizer):
        total_loss = 0
        val_loss = 0
        start_time = time.time()
        with tqdm(total=epoch_size, desc=('Epoch %s/%s'%(epoch+1, Epoch)), postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(gen):
                if iteration >= epoch_size:
                    break
                images, targets = batch[0], batch[1]
                with torch.no_grad():
                    if self.Cuda:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                    else:
                        images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                        targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                optimizer.zero_grad()
                outputs = net(images)
                losses = []
                for i in range(3):
                    loss_item = yolo_losses[i](outputs[i], targets)
                    losses.append(loss_item[0])
                loss = sum(losses)
                loss.backward()
                optimizer.step()

                total_loss += loss
                waste_time = time.time() - start_time

                pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration + 1),
                                    'lr': get_lr(optimizer),
                                    'step/s': waste_time})
                pbar.update(1)

                start_time = time.time()

        print('Start Validation')
        with tqdm(total=epoch_size_val, desc=('Epoch %s/%s'%(epoch+1, Epoch)), postfix=dict, mininterval=0.3) as pbar:
            for iteration, batch in enumerate(genval):
                if iteration >= epoch_size_val:
                    break
                images_val, targets_val = batch[0], batch[1]

                with torch.no_grad():
                    if self.Cuda:
                        images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                        targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                    else:
                        images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                        targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                    optimizer.zero_grad()
                    outputs = net(images_val)
                    losses = []
                    for i in range(3):
                        loss_item = yolo_losses[i](outputs[i], targets_val)
                        losses.append(loss_item[0])
                    loss = sum(losses)
                    val_loss += loss
                pbar.set_postfix(**{'total_loss': val_loss.item() / (iteration + 1)})
                pbar.update(1)

        print('Finish Validation')
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

        epoch_train_loss = float(total_loss) / float(epoch_size + 1)
        epoch_value_loss = float(val_loss) / float(epoch_size_val + 1)
        print('Saving state, iter:', str(epoch + 1))

        is_save, rm_epoch = self.is_save(epoch_value_loss, epoch+1)
        if is_save:
            torch.save(self.model.state_dict(),
                    self.ckpt_path + str('Epoch%d.pth' % (epoch + 1)))
            rm_path = self.ckpt_path + str('Epoch%d.pth' % rm_epoch)
            if os.path.isfile(rm_path):
                os.remove(rm_path)
            
        mlflow.log_metric("train_loss", epoch_train_loss, int(epoch + 1))
        mlflow.log_metric("val_loss", epoch_value_loss, int(epoch + 1))

    def is_save(self, value_loss, epoch):
        if value_loss < max(self.comare_list):
            idmax = self.comare_list.index(max(self.comare_list))
            self.comare_list[idmax] = value_loss
            rm_epoch = self.epoch_list[idmax]
            self.epoch_list[idmax] = epoch
            return True, rm_epoch
        return False, -1

    def train(self):
        net = self.model.train()

        if self.Cuda:
            net = torch.nn.DataParallel(self.model)
            cudnn.benchmark = True
            net = net.cuda()

        # 建立loss函数
        yolo_losses = []
        for i in range(3):
            yolo_losses.append(YOLOLoss(np.reshape(self.anchors, [-1, 2]), self.num_classes, \
                                        (self.input_shape[1], self.input_shape[0]), self.smooth_label, self.Cuda))

        # 0.1用于验证，0.9用于训练
        val_split = 0.1
        with open(self.annotation_path) as f:
            lines = f.readlines()
        np.random.seed(10101)
        np.random.shuffle(lines)
        np.random.seed(None)
        num_val = int(len(lines) * val_split)
        num_train = len(lines) - num_val

        # 冻结特征提取网络训练
        if True:
            optimizer = optim.Adam(net.parameters(), self.freeze_lr, weight_decay=5e-4)
            if self.Cosine_lr:
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
            else:
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

            if self.Use_Data_Loader:
                train_dataset = YoloDataset(lines[:num_train], (self.input_shape[0], self.input_shape[1]),
                                            mosaic=self.mosaic)
                val_dataset = YoloDataset(lines[num_train:], (self.input_shape[0], self.input_shape[1]), mosaic=False)
                gen = DataLoader(train_dataset, batch_size=self.freeze_bnsize, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
                gen_val = DataLoader(val_dataset, batch_size=self.freeze_bnsize, num_workers=4, pin_memory=True,
                                     drop_last=True, collate_fn=yolo_dataset_collate)
            else:
                gen = Generator(self.freeze_bnsize, lines[:num_train],
                                (self.input_shape[0], self.input_shape[1])).generate(mosaic=self.mosaic)
                gen_val = Generator(self.freeze_bnsize, lines[num_train:],
                                    (self.input_shape[0], self.input_shape[1])).generate(mosaic=False)

            epoch_size = max(1, num_train // self.freeze_bnsize)
            epoch_size_val = num_val // self.freeze_bnsize
            # ------------------------------------#
            #   冻结一定部分训练
            # ------------------------------------#
            for param in self.model.backbone.parameters():
                param.requires_grad = False

            for epoch in range(self.freeze_epoch):
                self.fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, self.freeze_epoch,
                                   optimizer)
                lr_scheduler.step()

        if True:
            optimizer = optim.Adam(net.parameters(), self.lr, weight_decay=5e-4)
            if self.Cosine_lr:
                lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-5)
            else:
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)

            if self.Use_Data_Loader:
                train_dataset = YoloDataset(lines[:num_train], (self.input_shape[0], self.input_shape[1]),
                                            mosaic=self.mosaic)
                val_dataset = YoloDataset(lines[num_train:], (self.input_shape[0], self.input_shape[1]), mosaic=False)
                gen = DataLoader(train_dataset, batch_size=self.bn_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
                gen_val = DataLoader(val_dataset, batch_size=self.bn_size, num_workers=4, pin_memory=True,
                                     drop_last=True, collate_fn=yolo_dataset_collate)
            else:
                gen = Generator(self.bn_size, lines[:num_train],
                                (self.input_shape[0], self.input_shape[1])).generate(mosaic=self.mosaic)
                gen_val = Generator(self.bn_size, lines[num_train:],
                                    (self.input_shape[0], self.input_shape[1])).generate(mosaic=False)

            epoch_size = max(1, num_train // self.bn_size)
            epoch_size_val = num_val // self.bn_size
            # ------------------------------------#
            #   解冻后训练
            # ------------------------------------#
            for param in self.model.backbone.parameters():
                param.requires_grad = True

            for epoch in range(self.freeze_epoch, self.total_epoch):
                self.fit_one_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, self.total_epoch,
                                   optimizer)
                lr_scheduler.step()


def main():
    pass


if __name__ == '__main__':
    main()
