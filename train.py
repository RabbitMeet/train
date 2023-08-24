import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'#调用GPU
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
osenvs = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))

import torch
import torch.nn

from utils import evaluate, get_dataset, FFDataset, setup_logger
from trainer import Trainer
import random

# config
dataset_path ='./DFDC/'
pretrained_path = './pretrained/xception-b5690688.pth'
batch_size = 64 
gpu_ids = [*range(osenvs)]
max_epoch = 30 
loss_freq = 500 
mode = 'trans' # ['Original', 'FAD', 'LFS', 'Both', 'Mix','DWT']Both2，bag3
ckpt_dir = './ckpts'
ckpt_name = '8'

if __name__ == '__main__':
    dataset = FFDataset(dataset_root=os.path.join(dataset_path, 'train' ,'real'), size=256, frame_num=800000, augment=True)#os.path.join是路径拼接，也就是dataset_path\train\real路径
    dataloader_real = torch.utils.data.DataLoader(
        dataset=dataset,#数据读取接口,该输出是torch.utils.data.Dataset类的对象(或者继承自该类的自定义类的对象)
        batch_size=batch_size // 2,#批训练数据量大小
        shuffle=True,#是否打乱数据，一般在训练数据中会采用。
        num_workers=8)#这个参数必须大于等于0，为0时默认使用主线程读取数据，其他大于0的数表示通过多个进程来读取数据，可以加快数据读取速度，一般设置为2的N次方，且小于batch_size（默认：0）
    
    len_dataloader = dataloader_real.__len__()#长度

    dataset_img, total_len =  get_dataset(name='train', size=256, root=dataset_path, frame_num=800000, augment=True)
    dataloader_fake = torch.utils.data.DataLoader(#该接口主要用来将自定义的数据读取接口的输出或者PyTorch已有的数据读取接口的输入按照batch_size封装成Tensor
        dataset=dataset_img,
        batch_size=batch_size // 2,
        shuffle=True,
        num_workers=8
    )
    print('dataloader_real.len=',dataloader_real.__len__())
    print('dataloader_fake.len=',dataloader_fake.__len__())

    # init checkpoint and logger 初始化检查点和日志
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)#生成新路径
    logger = setup_logger(ckpt_path, 'result.log', 'logger')#日志管理
    best_val = 0.
    ckpt_model_name = './ckpts/8/best.pkl'
    
    # train
    model = Trainer(gpu_ids, mode, pretrained_path)#GPU编号 模型 预训练模型
    model.total_steps = 0
    epoch = 0
    
    while epoch < max_epoch:

        fake_iter = iter(dataloader_fake)#函数生成迭代器
        real_iter = iter(dataloader_real)#生成适合迭代的类对象
        #print("fake_iter:",fake_iter,type(fake_iter))#<torch.utils.data.dataloader._MultiProcessingDataLoaderIter object at 0x7ff78f76bfa0> <class 'torch.utils.data.dataloader._MultiProcessingDataLoaderIter'>
        
        logger.debug(f'No {epoch}')#又是一堆日志
        i = 0

        
        while i < len_dataloader:#realdata的长度值，也许是数量
            #print(f"len_dataloader : {len_dataloader}")
            #print(f"epoch : {epoch}, i : {i}")
            i += 1
            model.total_steps += 1#初始值为0

            try:
                data_real = real_iter.__next__()#光标移动到下一行数据，有值（数据）返回true并迭代遍历，没有值，说明表中的行数已经走完，所以返回false退出循环。
                data_fake = fake_iter.__next__()#依次放入数据
                #print('data_real:',data_real,data_real.shape,type(data_real))#torch.Size([64, 3, 299, 299]) <class 'torch.Tensor'>
            except StopIteration:
                break
            # -------------------------------------------------
            
            if data_real.shape[0] != data_fake.shape[0]:#如果是图像就是图像的高（垂直），如果是矩阵就是矩阵的行
                continue

            bz = data_real.shape[0]
            

            data = torch.cat([data_real,data_fake],dim=0)#将两个张量拼接dim=0按行拼接（往下） =1按列拼接（往右）
            label = torch.cat([torch.zeros(bz).unsqueeze(dim=0),torch.ones(bz).unsqueeze(dim=0)],dim=1).squeeze(dim=0)
            #torch.zeros(行，列)返回一个0填充张量..unsqueeze(dim=0)整体升一维
            # manually shuffle
            idx = list(range(data.shape[0]))#列表（创建一个整数列表）
            random.shuffle(idx)#打乱列表顺序
            data = data[idx]
            #print("data:",data[idx],data[idx].shape,type(data[idx]))# torch.Size([128, 3, 299, 299]) <class 'torch.Tensor'>
            label = label[idx]


            data = data.detach()#返回一个新的tensor，requires_grad为false，得到的这个tensor永远不需要计算其梯度，不具有grad。
            label = label.detach()

            model.set_input(data,label)#copy到GPU上运行
            loss = model.optimize_weight()#返回损失

            if model.total_steps % loss_freq == 0:
                logger.debug(f'loss: {loss} at step: {model.total_steps}')

            if i % int(len_dataloader / 10) == 0:
                model.model.eval()#评估模式。而非训练模式。在评估模式下，batchNorm层，dropout层等用于优化训练而添加的网络层会被关闭，从而使得评估时不会发生偏移。
                auc, r_acc, f_acc, acc = evaluate(model, dataset_path, mode='valid')#evaluate将文本形式的公式转换成值
                logger.debug(f'(Val @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}, acc:{acc}')
                auc, r_acc, f_acc, acc  = evaluate(model, dataset_path, mode='test')
                logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}, acc:{acc}')
                model.model.train()
        epoch = epoch + 1

    model.model.eval()
    auc, r_acc, f_acc , acc = evaluate(model, dataset_path, mode='test')
    logger.debug(f'(Test @ epoch {epoch}) auc: {auc}, r_acc: {r_acc}, f_acc:{f_acc}, acc:{acc}')
    #保存模型
    torch.save(model,ckpt_model_name)
