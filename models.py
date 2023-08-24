from xception import Xception,Xception1
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt
import pywt.data
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from xception import SeparableConv2d
from transformer_pytorch_CV.models.modeling import Transformer
import ml_collections

# Filter Module
class Filter(nn.Module):
    def __init__(self, size, band_start, band_end, use_learnable=True, norm=False):
        super(Filter, self).__init__()
        self.use_learnable = use_learnable

        self.base = nn.Parameter(torch.tensor(generate_filter(band_start, band_end, size)), requires_grad=False)
        if self.use_learnable:
            self.learnable = nn.Parameter(torch.randn(size, size), requires_grad=True)#将一个不可训练的tensor转换成可以训练的类型parameter，并将这个parameter绑定到这个module里面。torch.randn()返回一个张量，包含了从标准正态分布（均值为0，方差为1，即高斯白噪声）中抽取的一组随机数。张量的形状由参数sizes定义
            self.learnable.data.normal_(0., 0.1)#随机初始化范围
            # Todo
            # self.learnable = nn.Parameter(torch.rand((size, size)) * 0.2 - 0.1, requires_grad=True)

        self.norm = norm
        if norm:
            self.ft_num = nn.Parameter(torch.sum(torch.tensor(generate_filter(band_start, band_end, size))), requires_grad=False)


    def forward(self, x):
        if self.use_learnable:
            filt = self.base + norm_sigma(self.learnable)
        else:
            filt = self.base

        if self.norm:
            y = x * filt / self.ft_num
        else:
            y = x * filt
        return y
class IDWT_Head(nn.Module):
    def __init__(self,size):#补充需求
        super(IDWT_Head, self).__init__()
        self.size = size
        self.count = 5
        self.a = 5
        self.b = 5
        self.width = 64
        self.height = 64
    def forward(self, img):

        x_gray = 0.299*img[:,2,:,:] + 0.587*img[:,1,:,:] + 0.114*img[:,0,:,:]
        x = x_gray.unsqueeze(1)
        
        '''分块置乱'''
        # x_gray = x_gray.cpu().numpy()
        # x_gray = x_gray.transpose(2,1,0)
        # x_gray = Arnold(x_gray,self.width,self.height,self.count,self.a,self.b)
        # x_gray.completion()
        # x_gray = x_gray.arnold1()
        # x_gray = x_gray.transpose(2,1,0)
        # x = x_gray[:,np.newaxis,:,:]
        
        # rescale to 0 - 255
        x = (x + 1.) * 127.5
        # print('x.shape:',x.shape)# x.shape: torch.Size([16, 1, 299, 299])
        x = x.cpu().numpy()
        coeffs = pywt.dwt2(x, 'haar')
        cA, (cH, cV, cD) = coeffs
        m = cA.shape
        M0 = np.zeros((m[0],1,128,128))#150
        # cH = pywt.idwt2((cA, (cH, M0, M0)),'haar')
        # cV = pywt.idwt2((cA, (M0, cV, M0)),'haar') 
        # cD = pywt.idwt2((cA, (M0, M0, cD)),'haar')
        cH = pywt.idwt2((cA, (cH, M0, M0)),'haar')
        cV = pywt.idwt2((cA, (M0, cV, M0)),'haar') 
        cD = pywt.idwt2((cA, (M0, M0, cD)),'haar')
        cH = torch.from_numpy(cH)
        cV = torch.from_numpy(cV)
        cD = torch.from_numpy(cD)
        image = torch.cat([cH, cV, cD],dim =1)
        return image

class DWT_Head(nn.Module):
    def __init__(self, maxlevel):
        self.maxlevel = 1
        super(DWT_Head, self).__init__()
    def forward(self,img):
        #print(img.shape,type(img)) #torch.Size([8, 3, 299, 299]) <class 'torch.Tensor'>
        #img = img.cpu().numpy()
        image_r = img[:,0, :, :]
        image_g = img[:,1, :, :]
        image_b = img[:,2, :, :]

        img2 = []
        for x in (image_r, image_g ,image_b):
            x = x.cpu().numpy()
            wp = pywt.WaveletPacket2D(data=x, wavelet='haar',mode='symmetric',maxlevel=1)
#计算每一个节点的系数，存在map中，key为'aa'等，value为列表
            map = {}
            map[1] = x
            # b,c,w,h = img.shape
            # img2 = torch.ones((b,3*c,w,h)).cuda()
            
            a=nn.Upsample(size = (256,256),mode='bilinear',align_corners=True )
            for row in range(1,self.maxlevel+1):
                #lev = []
                for i in [node.path for node in wp.get_level(row, 'natural')]:
                    map[i] = wp[i].data
                    map[i] = torch.from_numpy(map[i]).unsqueeze(1)
                    #print('map[i]:',map[i].shape)#torch.Size([8, 1, 150, 150])
                    map[i] = a(map[i])
                #print('i',i,map[i].shape)
            #img = map['a']
            #print('n:',n,'map[h]:',map['h'].shape, map['v'].shape, map['d'].shape)#n: 0 map[h]: torch.Size([8, 1, 299, 299]) torch.Size([8, 1, 299, 299]) torch.Size([8, 1, 299, 299])
            # img2.append(torch.cat([map['a'],map['h'], map['v'], map['d']],dim =1))
            img2.append(torch.cat([map['h'], map['v'], map['d']],dim =1))
            #n=n+1
            #img21 = torch.cat([map['hh'], map['vh'], map['dh']],dim =1)
            #img22 = torch.cat([map['hv'], map['vv'], map['dv']],dim =1)
            #img23 = torch.cat([map['hd'], map['vd'], map['dd']],dim =1)
        image = torch.cat([img2[0],img2[1],img2[2]],dim =1)
        # print('image',image.shape)torch.Size([16, 9, 299, 299])
        return image
      
class Bag_Head(nn.Module):
    def __init__(self, maxlevel):
        self.maxlevel = maxlevel
        super(Bag_Head, self).__init__()
    def forward(self,img):
        #print('img',img.shape)#img torch.Size([16, 3, 299, 299])
        x_gray = 0.299*img[:,2,:,:] + 0.587*img[:,1,:,:] + 0.114*img[:,0,:,:]#x1 torch.Size([16, 299, 299])
        x = x_gray.unsqueeze(1)#x2 torch.Size([16, 299, 299])
        # rescale to 0 - 255
        x = (x + 1.) * 122.5
        x = x.cpu().numpy()
        wp = pywt.WaveletPacket2D(data=x, wavelet='haar',mode='symmetric',maxlevel=self.maxlevel)
 
    #计算每一个节点的系数，存在map中，key为'aa'等，value为列表
        map = {}
        map[1] = x
        a=nn.Upsample(size = (256,256),mode='bilinear',align_corners=True )
        for row in range(1,self.maxlevel+1):
            lev = []
            for i in [node.path for node in wp.get_level(row, 'natural')]:
                map[i] = wp[i].data
                #print('map[i]:',map[i].shape)#torch.Size([8, 1, 75, 75])
                map[i] = torch.from_numpy(map[i])
                #print('map[i]:',map[i].shape)#torch.Size([8, 1, 75, 75])
                map[i] = a(map[i])
                #print('map[i]:',map[i].shape)
                #print('i',i,map[i].shape)
        img1 = torch.cat([map['h'], map['v'], map['d']],dim =1)
        # print('img1',img1.shape)torch.Size([16, 3, 299, 299])
        img21 = torch.cat([map['hh'], map['vh'], map['dh']],dim =1)
        img22 = torch.cat([map['hv'], map['vv'], map['dv']],dim =1)
        img23 = torch.cat([map['hd'], map['vd'], map['dd']],dim =1)
        image = torch.cat([img21,img22,img23],dim =1)
        # print('image',image.shape)torch.Size([16, 9, 299, 299])
        return img1,image


class Bag_Head3(nn.Module):
    def __init__(self, maxlevel):
        self.maxlevel = maxlevel
        super(Bag_Head3, self).__init__()
    def forward(self,img):
        image_r = img[:,0, :, :]
        image_g = img[:,1, :, :]
        image_b = img[:,2, :, :]

        img2 = []
        img1 = []
        img3 = []
        img4 = []
        

        for x in (image_r, image_g ,image_b):
            x = x.cpu().numpy()
            wp = pywt.WaveletPacket2D(data=x, wavelet='haar',mode='symmetric',maxlevel=2)
 
    #计算每一个节点的系数，存在map中，key为'aa'等，value为列表
            map = {}
            map[1] = x
            # b,c,w,h = img.shape
            # img2 = torch.ones((b,3*c,w,h)).cuda()
            
            a=nn.Upsample(size = (256,256),mode='bilinear',align_corners=True )
            for row in range(1,self.maxlevel+1):
                #lev = []
                for i in [node.path for node in wp.get_level(row, 'natural')]:
                    map[i] = wp[i].data
                    map[i] = torch.from_numpy(map[i]).unsqueeze(1)
                    #print('map[i]:',map[i].shape)#torch.Size([8, 1, 150, 150])
                    map[i] = a(map[i])
                
            img1.append(torch.cat([map['h'], map['v'], map['d']],dim =1))
            img2.append(torch.cat([map['hh'], map['hv'], map['hd']],dim =1))
            img3.append(torch.cat([map['vh'], map['vv'], map['vd']],dim =1))
            img4.append(torch.cat([map['dh'], map['dv'], map['dd']],dim =1))
            
        image1 = torch.cat([img1[0],img1[1],img1[2]],dim =1)
        image2 = torch.cat([img2[0],img2[1],img2[2],img3[0],img3[1],img3[2],img4[0],img4[1],img4[2]],dim =1)#,img5[0],img5[1],img5[2]
        return image1,image2

class F3Net(nn.Module):
    def __init__(self, config, num_classes=1, img_width=256, img_height=256, LFS_window_size=10, LFS_stride=2, LFS_M = 6, mode='FAD', device=True, zero_head=False, vis=False):
        super(F3Net, self).__init__()
        assert img_width == img_height
        img_size = img_width
        self.num_classes = num_classes
        self.mode = mode
        self.window_size = LFS_window_size
        self._LFS_M = LFS_M
        maxlevel = 2
        ####trans
        #self.transformer = nn.TransformerEncoderLayer(d_model=32, nhead=4, num_encoder_layers=12)
        
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis)
        self.head = nn.Linear(config.hidden_size, num_classes)#wm,768-->10

        # init branches


        if mode == 'IDWT'or mode == 'Both2':
            self.IDWT_head = IDWT_Head(img_size)
            self.init_xcep()

        if mode == 'Both' or mode == 'DWT'or mode == 'Both2':
            self.DWT_head = DWT_Head(img_size)
            self.init_xcep_FAD()
            self.init_xcep()
            

        if mode == 'Bag' or mode == 'Both2'or mode == 'Both3' or mode == 'Mix':
            self.Bag_head3 = Bag_Head3(maxlevel)#原本是没油3的
            self.init_xcep_FAD1()#_FAD是3通道以上，xcep是只有三通道
            self.init_xcep_FAD()#init_xcep()
        if mode == 'Bag3':
            self.Bag_head3 = Bag_Head3(maxlevel)
            self.init_xcep_FAD()#_FAD是3通道以上，xcep是只有三通道
            self.init_xcep_FAD1()

        if mode == 'Original' or mode == 'Both2':
            self.init_xcep()

        # classifier
        self.conv = nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(4096 if self.mode == 'Bag' else 2048, num_classes)#or self.mode == 'Both2'
        self.fc = nn.Linear(4096 if self.mode == 'Bag' or self.mode == 'Bag3' or self.mode == 'Mix' or self.mode == 'OUR'  else 2048, num_classes)#or self.mode == 'Both2'
        self.dp = nn.Dropout(p=0.2)

    def init_xcep_FAD(self):#放进网络的形式大概是
        self.FAD_xcep = Xception(self.num_classes)
        #self.FAD_xcep.conv1 = nn.Conv2d(9, 32, 3, 2, 0, bias=False)#12,32,3,2,0 改权重需要改这个地方的12为9
        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = get_xcep_state_dict()#预处理参数
        conv1_data = state_dict['conv1.weight'].data

        self.FAD_xcep.load_state_dict(state_dict, False)

        # copy on conv1
        # let new conv1 use old param to balance the network
        #权重被用在了小波包的权重里
        self.FAD_xcep.conv1 = nn.Conv2d(27, 32, 3, 2, 0, bias=False)#12,32,3,2,0 改权重需要改这个地方的12为9
        self.FAD_xcep.conv3 = SeparableConv2d(1024,1024,3,1,1)
        self.FAD_xcep.bn3 = nn.BatchNorm2d(1024)
        self.FAD_xcep.conv4 = SeparableConv2d(1024,1024,3,1,1)
        self.FAD_xcep.bn4 = nn.BatchNorm2d(1024)
        self.FAD_xcep.fc = nn.Linear(1024, 1)
        for i in range(9):#4
            self.FAD_xcep.conv1.weight.data[:, i*3:(i+1)*3, :, :] = conv1_data / 9.0#4.0

        # for i in range(int(self._LFS_M / 3)):#self._LFS_M与输入通道数相同
        #     self.LFS_xcep.conv1.weight.data[:, i * 3:(i + 1) * 3, :, :] = conv1_data / float(self._LFS_M / 3.0)
    def init_xcep_FAD1(self):#放进网络的形式
        self.Bag_xcep1 = Xception1(self.num_classes)
        #self.FAD_xcep.conv1 = nn.Conv2d(9, 32, 3, 2, 0, bias=False)#12,32,3,2,0 改权重需要改这个地方的12为9
        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = get_xcep_state_dict()#预处理参数
        conv1_data = state_dict['conv1.weight'].data

        self.Bag_xcep1.load_state_dict(state_dict, False)

        # copy on conv1
        # let new conv1 use old param to balance the network
        #权重被用在了小波包的权重里
        self.Bag_xcep1.conv1 = nn.Conv2d(9, 32, 3, 2, 0, bias=False)#12,32,3,2,0 改权重需要改这个地方的12为9
        self.Bag_xcep1.conv3 = SeparableConv2d(1024,1024,3,1,1)
        self.Bag_xcep1.bn3 = nn.BatchNorm2d(1024)
        self.Bag_xcep1.conv4 = SeparableConv2d(1024,1024,3,1,1)
        self.Bag_xcep1.bn4 = nn.BatchNorm2d(1024)
        self.Bag_xcep1.fc = nn.Linear(1024, 1)
        for i in range(3):#4
            self.Bag_xcep1.conv1.weight.data[:, i*3:(i+1)*3, :, :] = conv1_data / 3.0#4.0

        # for i in range(int(self._LFS_M / 3)):#self._LFS_M与输入通道数相同
        #     self.LFS_xcep.conv1.weight.data[:, i * 3:(i + 1) * 3, :, :] = conv1_data / float(self._LFS_M / 3.0)
    def init_xcep_LFS(self):
        self.LFS_xcep = Xception(self.num_classes)

        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        state_dict = get_xcep_state_dict()
        conv1_data = state_dict['conv1.weight'].data

        self.LFS_xcep.load_state_dict(state_dict, False)

        # copy on conv1
        # let new conv1 use old param to balance the network
        self.LFS_xcep.conv1 = nn.Conv2d(self._LFS_M, 32, 3, 1, 0, bias=False)
        for i in range(int(self._LFS_M / 3)):
            self.LFS_xcep.conv1.weight.data[:, i * 3:(i + 1) * 3, :, :] = conv1_data / float(self._LFS_M / 3.0)

    def init_xcep(self):
        self.xcep = Xception(self.num_classes)
        ###################################################################################################################################
        # To get a good performance, using ImageNet-pretrained Xception model is recommended
        #cat就不能调用预训练权重
        state_dict = get_xcep_state_dict()
        self.xcep.load_state_dict(state_dict, False)

        if self.mode == 'Original':
#########################################################################
            #print("x:",x,x.shape,type(x))#device='cuda:1') torch.Size([32, 3, 299, 299]) <class 'torch.Tensor'>
            fea = self.xcep.features(x)
            fea = self._norm_fea(fea)
            #print(fea.shape)
            y = fea
       
        if self.mode == 'DWT':
            fea = self.DWT_head(x).cuda()
            fea = self.FAD_xcep.features(fea)
            # fea = self.xcep.features(fea)
            fea = self._norm_fea(fea)
            y = fea

        if self.mode == 'IDWT':
            fea = self.IDWT_head(x).cuda()
            fea = fea.float()
            fea = self.xcep.features(fea)
            #fea = self.xcep.logits(fea)
            fea = self._norm_fea(fea)
            y = fea

            fea_Bag1,fea_Bag2 = self.Bag_head(x)
            fea_Bag1,fea_Bag2 = fea_Bag1.cuda(),fea_Bag2.cuda()           

            fea_Bag2 = self.FAD_xcep.features(fea_Bag2)
            
            fea_Bag2 = self._norm_fea(fea_Bag2)
              
            fea_Bag1 = self.xcep.features(fea_Bag1)

            fea_Bag1 = self._norm_fea(fea_Bag1)

            y = torch.cat((fea_Bag1, fea_Bag2), dim=1) 

        if self.mode == 'Bag3':

            fea_Bag1,fea_Bag2 = self.Bag_head3(x)
            fea_Bag1,fea_Bag2 = fea_Bag1.cuda(),fea_Bag2.cuda()           
            mix_block = MixBlock(728).cuda()
            fea_Bag1 = self.Bag_xcep1.fea_0_5(fea_Bag1)

            fea_Bag2 = self.FAD_xcep.fea_0_3(fea_Bag2)

            fea_Bag1,fea_Bag2 = mix_block(fea_Bag1,fea_Bag2)
            mix_block = MixBlock(728).cuda()
            fea_Bag1 = self.Bag_xcep1.fea_6_7(fea_Bag1)

            fea_Bag2 = self.FAD_xcep.fea_4_5(fea_Bag2)

            fea_Bag1,fea_Bag2 = mix_block(fea_Bag1,fea_Bag2)
            mix_block = MixBlock(728).cuda()
            fea_Bag1 = self.Bag_xcep1.fea_8_9(fea_Bag1)

            fea_Bag2 = self.FAD_xcep.fea_6_7(fea_Bag2)

            fea_Bag1,fea_Bag2 = mix_block(fea_Bag1,fea_Bag2)

            fea_Bag1 = self.Bag_xcep1.fea_10_12(fea_Bag1)

            fea_Bag2 = self.FAD_xcep.fea_10_12(fea_Bag2)

            fea_Bag1 = self._norm_fea(fea_Bag1)

            fea_Bag2 = self._norm_fea(fea_Bag2)
            y = torch.cat((fea_Bag1, fea_Bag2), dim=1)   
            #y = bilinear_pooling(fea,fea1)
            #y = self._norm_fea(y)

              if self.mode == 'Both2':#用的是bag3，万不可删

            # fea = self.DWT_head(x).cuda()#原本小波以及
            # fea1 = self.IDWT_head(x).cuda()
            # fea = self.FAD_xcep.features(fea)
            # fea1 = self.xcep.features(x)
            # y = bilinear_pooling(fea,fea1)
            # y = self._norm_fea(y)
            
            fea_Bag1,fea_Bag2 = self.Bag_head3(x)
            fea_Bag1,fea_Bag2 = fea_Bag1.cuda(),fea_Bag2.cuda()           
            mix_block = MixBlock(728).cuda()
            fea_Bag1 = self.Bag_xcep1.fea_0_5(fea_Bag1)

            fea_Bag2 = self.FAD_xcep.fea_0_3(fea_Bag2)

            fea_Bag1,fea_Bag2 = mix_block(fea_Bag1,fea_Bag2)
            mix_block = MixBlock(728).cuda()
            fea_Bag1 = self.Bag_xcep1.fea_6_7(fea_Bag1)

            fea_Bag2 = self.FAD_xcep.fea_4_5(fea_Bag2)

            fea_Bag1,fea_Bag2 = mix_block(fea_Bag1,fea_Bag2)
            mix_block = MixBlock(728).cuda()
            fea_Bag1 = self.Bag_xcep1.fea_8_9(fea_Bag1)

            fea_Bag2 = self.FAD_xcep.fea_6_7(fea_Bag2)

            fea_Bag1,fea_Bag2 = mix_block(fea_Bag1,fea_Bag2)

            fea_Bag1 = self.Bag_xcep1.fea_10_12(fea_Bag1)

            fea_Bag2 = self.FAD_xcep.fea_10_12(fea_Bag2)

            # print('shape',fea_Bag1.shape)
            fea_Bag = torch.cat((fea_Bag1, fea_Bag2), dim=1)
            fea_Bag = self.conv(fea_Bag)
            y = self._norm_fea(fea_Bag)

            if self.mode =='Mix':
            mix_block = MixBlock(728).cuda()
            fea_Bag1,fea_Bag2 = self.Bag_head(x)
            fea_Bag1,fea_Bag2 = fea_Bag1.cuda(),fea_Bag2.cuda()

            fea_Bag1 = self.xcep.fea_0_3(fea_Bag1)
            fea_Bag2 = self.FAD_xcep.fea_0_3(fea_Bag2)
            fea_Bag1,fea_Bag2 = mix_block(fea_Bag1,fea_Bag2)

            mix_block = MixBlock(728).cuda()
            fea_Bag1 = self.xcep.fea_4_6(fea_Bag1)
            fea_Bag2 = self.FAD_xcep.fea_4_6(fea_Bag2)
            fea_Bag1,fea_Bag2 = mix_block(fea_Bag1,fea_Bag2)

            mix_block = MixBlock(728).cuda()
            fea_Bag1 = self.xcep.fea_7_9(fea_Bag1)
            fea_Bag2 = self.FAD_xcep.fea_7_9(fea_Bag2)
            fea_Bag1,fea_Bag2 = mix_block(fea_Bag1,fea_Bag2)

            fea_Bag1 = self.xcep.fea_10_12(fea_Bag1)
            fea_Bag2 = self.FAD_xcep.fea_10_12(fea_Bag2)
            fea_Bag1 = self._norm_fea(fea_Bag1)
            fea_Bag2 = self._norm_fea(fea_Bag2)
            y = torch.cat((fea_Bag1, fea_Bag2), dim=1)#按列拼接（右）

        
        f = self.dp(y)
        f = self.fc(f)
        return f

def DCT_mat(size):#自己编写的DCT计算
    m = [[ (np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] for i in range(size)]
    return m

def generate_filter(start, end, size):
    return [[0. if i + j > end or i + j < start else 1. for j in range(size)] for i in range(size)]

def norm_sigma(x):
    return 2. * torch.sigmoid(x) - 1.

def get_xcep_state_dict(pretrained_path='./pretrained/xception-b5690688.pth'):
#def get_xcep_state_dict(pretrained_path='./ckpts/1/best.pkl'):
    # load Xception
    state_dict = torch.load(pretrained_path)#加载预训练模型
    for name, weights in state_dict.items():#返回键值对‘name’‘weight’
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)#扩充成n*1*1的tensor [[[]],[[]],[[]]]
    state_dict = {k:v for k, v in state_dict.items() if 'fc' not in k}#创建字典，与原本KV值互换
    return state_dict

def bilinear_pooling(x,y):
    x_size = x.size()
    y_size = y.size()
    assert(x_size[:-1] == y_size[:-1])

    out_size = list(x_size)
    #out_size[-1] = x_size[-1]*y_size[-1]

    x = x.view([-1,x_size[-1],1])   # [N*C, F, 1] 
    # print('x',x.shape)
    y = y.view([-1,1,y_size[-1]])   # [N*C, 1, F]
    # print('y',y.shape)

    out = torch.matmul(x, y)   # [N*C,F,F]
    # print('out',out.shape)
    out = out.view(out_size[0],out_size[1]*2,out_size[2]*2,out_size[3]*2)   #[N,C,F*8,F*8]

    return F.normalize(out, p=2, dim=1)   # L2归一化

class MixBlock(nn.Module):
    # An implementation of the cross attention module in F3-Net
    # Haven't added into the whole network yet
    # def __init__(self, c_in, width, height):
    def __init__(self, c_in):
        super(MixBlock, self).__init__()
        self.FAD_query = nn.Conv2d(c_in, c_in, (1,1))
        self.LFS_query = nn.Conv2d(c_in, c_in, (1,1))

        self.FAD_key = nn.Conv2d(c_in, c_in, (1,1))
        self.LFS_key = nn.Conv2d(c_in, c_in, (1,1))

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        #可训练参数
        self.FAD_gamma = nn.Parameter(torch.zeros(1))
        self.LFS_gamma = nn.Parameter(torch.zeros(1))

        self.FAD_conv = nn.Conv2d(c_in, c_in, (1,1), groups=c_in)
        self.FAD_bn = nn.BatchNorm2d(c_in)
        self.LFS_conv = nn.Conv2d(c_in, c_in, (1,1), groups=c_in)
        self.LFS_bn = nn.BatchNorm2d(c_in)
    def forward(self, x_FAD, x_LFS):#交叉和自注意力
        B, C, W, H = x_FAD.size()
        Bb, Cc, Ww, Hh = x_LFS.size()
        #print(B, C, W, H)
        #print(Bb, Cc, Ww, Hh)因为输入尺寸不同所以没法cat
        assert W == H

        q_FAD = self.FAD_query(x_FAD).view(-1, W, H)    # [BC, W, H]
        k_FAD = self.FAD_key(x_FAD).view(-1, W, H).transpose(1, 2)  # [BC, H, W]
        energy1 = torch.bmm(q_FAD, k_FAD)
        attention1 = self.softmax(energy1).view(B, C, W, W)
        att_FAD = x_FAD * attention1 * (torch.sigmoid(self.FAD_gamma) * 2.0 - 1.0)
        y_FAD = x_FAD + self.FAD_bn(self.FAD_conv(att_FAD))

        q_LFS = self.LFS_query(x_LFS).view(-1, W, H) 
        k_LFS = self.LFS_key(x_LFS).view(-1, W, H).transpose(1, 2)
        energy2 = torch.bmm(q_LFS, k_LFS)  #[BC, W, W]#####矩阵乘法
        attention2 = self.softmax(energy2).view(B, C, W, W)
        att_LFS = x_LFS * attention2 * (torch.sigmoid(self.LFS_gamma) * 2.0 - 1.0)
        y_LFS = x_LFS + self.LFS_bn(self.LFS_conv(att_LFS))


        q_FAD = self.FAD_query(y_FAD).view(-1, W, H)    # [BC, W, H]
        q_LFS = self.LFS_query(y_LFS).view(-1, W, H)
        M_query = torch.cat([q_FAD, q_LFS], dim=2)  # [BC, W, 2H]

        k_FAD = self.FAD_key(y_FAD).view(-1, W, H).transpose(1, 2)  # [BC, H, W]
        k_LFS = self.LFS_key(y_LFS).view(-1, W, H).transpose(1, 2)
        M_key = torch.cat([k_FAD, k_LFS], dim=1)    # [BC, 2H, W]

        energy = torch.bmm(M_query, M_key)  #[BC, W, W]#####矩阵乘法
        attention = self.softmax(energy).view(B, C, W, W)

        att_LFS = y_LFS * attention * (torch.sigmoid(self.LFS_gamma) * 2.0 - 1.0)
        y_FAD = y_FAD + self.FAD_bn(self.FAD_conv(att_LFS))
        
        att_FAD = y_FAD * attention * (torch.sigmoid(self.FAD_gamma) * 2.0 - 1.0)
        y_LFS = y_LFS + self.LFS_bn(self.LFS_conv(att_FAD))
        return y_FAD, y_LFS

  if __name__ == '__main__':
    F = F3Net(mode = 'Mix')
    print(F)
