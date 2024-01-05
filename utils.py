import torch
import os
import numpy as np
from torch.utils import data
from torchvision import transforms as trans
from sklearn.metrics import average_precision_score, precision_recall_curve, accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc as cal_auc
from PIL import Image
import sys
import logging
import cv2

class FFDataset(data.Dataset):

    def __init__(self, dataset_root, frame_num=300, size=299, augment=True):#augment是否数据增强 frame_num我怕出错，但是应该是从train得到的
        self.data_root = dataset_root
        self.frame_num = frame_num
        self.train_list = self.collect_image(self.data_root)#看下面collect_image,返回图片列表
        if augment:
            self.transform = trans.Compose([trans.ToPILImage(),trans.RandomHorizontalFlip(p=0.5), trans.ToTensor()])#常用图像变换.水平翻转概率0.5.转换为tensor类
      
        else:
            self.transform = trans.Compose([trans.RandomHorizontalFlip(p=0.5), trans.ToTensor()])
            
        self.max_val = 1.#参数的最大值
        self.min_val = -1.
        self.size = size
          def collect_image(self, root):
        image_path_list = []
        img_names = os.listdir(root)
        img_names = img_names if len(img_names) < self.frame_num else img_names[:self.frame_num]
        for img_name in img_names:
            image_path_list.append(os.path.join(root,img_name))
        #print('img_names=',img_names.__len__())
        #print('image_path_list.len=',image_path_list.__len__())
        return image_path_list#返回图片列表
        

    def read_image(self, path):#path是哪个图片的路径
        #img = Image.open(path)#读取图片的RGBA，A表示图像的alpha通道
        img = cv2.imread(path)#BGR
        #img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        return img#返回图片四个参数RGBA

    def resize_image(self, image, size):#裁剪图像或者缩放
        image = np.array(image)
        img = cv2.resize(image,(size,size))
        #print('read2:',type(img))
        return img
    
    def __getitem__(self, index):
        image_path = self.train_list[index]#图片列表
        img = self.read_image(image_path)
        img = self.resize_image(img,size=self.size)

        #print("read_img:",img,type(img))#<PIL.Image.Image image mode=RGB size=299x299 at 0x7F47E60ECCA0> <class 'PIL.Image.Image'>
    
        img = self.transform(img)#数据处理后的图像tensor格式
        #print("read_img:",img,img.shape,type(img))#torch.Size([3, 299, 299]) <class 'torch.Tensor'>
        img = img * (self.max_val - self.min_val) + self.min_val
        #print("read_img:",img,img.shape,type(img))##torch.Size([3, 299, 299]) <class 'torch.Tensor'>

        return img

    def __len__(self):
        return len(self.train_list)#返回列表长度

def get_dataset(name = 'train', size=299, root='/data/yike/FF++_std_c40_300frames/', frame_num=300, augment=True):
    root = os.path.join(root, name)
    fake_root = os.path.join(root,'fake')

    total_len=1
    dset_lst = []
    for i in range(total_len):
        dset = FFDataset(fake_root, frame_num, size, augment)
        dset.size = size
        dset_lst.append(dset)
    return torch.utils.data.ConcatDataset(dset_lst), total_len

def evaluate(model, data_path, mode='valid'):
    
    root= data_path
    origin_root = root
    root = os.path.join(data_path, mode)
    real_root = os.path.join(root,'real')
    dataset_real = FFDataset(dataset_root=real_root, size=256, frame_num=8, augment=False)
    dataset_fake, _ = get_dataset(name=mode, root=origin_root, size=256, frame_num=8, augment=False)
    dataset_img = torch.utils.data.ConcatDataset([dataset_real, dataset_fake])#ConcatDataset数据处理.合并子集

    bz = 64

    with torch.no_grad():#在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。
        y_true, y_pred1, y_pred2, y_pred3 = [], [], [], []

        for i, d in enumerate(dataset_img.datasets):#遍历列表字符串
            dataloader = torch.utils.data.DataLoader(
                dataset = d,
                batch_size = bz,
                shuffle = True,
                num_workers = 8
            )
            for img in dataloader:
                if i == 0:
                    label = torch.zeros(img.size(0))
                else:
                    label = torch.ones(img.size(0))

                img = img.detach().cuda()#拦截梯度
                output3= model.forward(img)

                y_pred3.extend(output3.sigmoid().flatten().tolist())
                y_true.extend(label.flatten().tolist())

    y_true, y_pred3 = np.array(y_true), np.array(y_pred3)
    fpr, tpr, thresholds = roc_curve(y_true,y_pred3,pos_label=1)
    AUC = cal_auc(fpr, tpr)



    idx_real = np.where(y_true==0)[0]#行索引
    idx_fake = np.where(y_true==1)[0]

    r_acc = accuracy_score(y_true[idx_real], y_pred3[idx_real] > 0.5)#真实数据中判断对的
    f_acc = accuracy_score(y_true[idx_fake], y_pred3[idx_fake] > 0.5)#伪造数据中判断对的
    acc = accuracy_score(y_true,y_pred3>0.5)#应该如此   


     
    return AUC, r_acc, f_acc, acc


"""Utility functions for logging."""

__all__ = ['setup_logger']

DEFAULT_WORK_DIR = 'results'

def setup_logger(work_dir=None, logfile_name='log.txt', logger_name='logger'):
    """Sets up logger from target work directory.

    The function will sets up a logger with `DEBUG` log level. Two handlers will
    be added to the logger automatically. One is the `sys.stdout` stream, with
    `INFO` log level, which will print improtant messages on the screen. The other
    is used to save all messages to file `$WORK_DIR/$LOGFILE_NAME`. Messages will
    be added time stamp and log level before logged.

    NOTE: If `logfile_name` is empty, the file stream will be skipped. Also,
    `DEFAULT_WORK_DIR` will be used as default work directory.

    Args:
    work_dir: The work directory. All intermediate files will be saved here.
        (default: None)
    logfile_name: Name of the file to save log message. (default: `log.txt`)
    logger_name: Unique name for the logger. (default: `logger`)

    Returns:
    A `logging.Logger` object.

    Raises:
    SystemExit: If the work directory has already existed, of the logger with
        specified name `logger_name` has already existed.
    """

    logger = logging.getLogger(logger_name)#初始化日志
    if logger.hasHandlers():  # Already existed 检查记录器是否配置了任何处理程序
        raise SystemExit(f'Logger name `{logger_name}` has already been set up!\n'
                            f'Please use another name, or otherwise the messages '
                            f'may be mixed between these two loggers.')#好像是个报错

    logger.setLevel(logging.DEBUG)#用于调试，指定logger将会处理的那个安全等级日志信息
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s")#Python日志格式输出 字符串形式的当前时间 文本形式的日志级别 用户输出的消息
    # Print log message with `INFO` level or above onto the screen.
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    if not logfile_name:
        return logger

    work_dir = work_dir or DEFAULT_WORK_DIR
    logfile_name = os.path.join(work_dir, logfile_name)

    os.makedirs(work_dir, exist_ok=True)

    # Save log message with all levels in log file.
    fh = logging.FileHandler(logfile_name)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
