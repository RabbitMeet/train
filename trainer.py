import torch
import torch.nn as nn
from models import F3Net
import ml_collections
from transformer_pytorch_CV.models.modeling import  CONFIGS
config = CONFIGS["ViT-B_16"]

def initModel(mod, gpu_ids):
    model = mod.to(f'cuda:{gpu_ids[0]}')#将MOD放入GPU[0]
    mod = nn.DataParallel(model, gpu_ids)######多GPU训练（自定义的模型，gpu编号）,broadcast_buffers=False
    return mod

class Trainer(): 
    def __init__(self, gpu_ids, mode, pretrained_path):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')#调用指定GPU 
        self.model = F3Net(config,mode=mode, device=self.device)
        self.model = initModel(self.model, gpu_ids)#返回放入GPU多卡训练后的mod 问题：.double：Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same
        self.loss_fnc = nn.BCEWithLogitsLoss()#二元交叉熵损失函数
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),#从参数tensor集合-- parameters中找到可变的tensor形成一个新的集合
                                              lr=0.0002, betas=(0.9, 0.999))#adam优化器（待优化参数，学习率，一阶矩估计的衰减率，二阶矩估计的衰减率
    def set_input(self, input, label):
        self.input = input.to(self.device)#将所有最开始读取数据时的tensor变量copy一份到device所指定的GPU上去，之后的运算都在GPU上进行。
        self.label = label.to(self.device)

    def forward(self, x):
        out = self.model(x)
        return out
    
    def optimize_weight(self):
        stu_cla = self.model(self.input)
        self.loss_cla = self.loss_fn(stu_cla.squeeze(1), self.label) # classify loss #squeeze(1)单个元素升维
        self.loss = self.loss_cla

        self.optimizer.zero_grad()#将参数梯度置0 看看optimizer
        self.loss.backward()#损失反向传播计算当前梯度 看看loss
        self.optimizer.step() ####优化器.step()
        return self.loss

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
