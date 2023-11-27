import torch
from einops import rearrange, repeat
from torch import nn, einsum
import torch.nn.functional as F
import math
from functools import partial
import random
import numpy as np
import sys
sys.path.append("./models")
torch.set_printoptions(precision=3,edgeitems=32,linewidth=350)
from ST_Former import TimeSformer


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False, dilation=dilation) # Hin=Win,

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


##building block

class BasicBlock(nn.Module):
    expansion = 1

    """inplanes输入通道数，planes卷积输出通道数,group卷积层的分组数"""
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        self.inplanes=inplanes
        self.planes=planes
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride) # 3*3卷积
        self.bn1 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)#3*3卷积
        self.bn2 = norm_layer(planes)

        self.relu = nn.ReLU(inplace=True)
        self.conv3=conv3x3(planes,planes)
        self.bn3 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.ReLU(),
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes),

            nn.ReLU(),
            nn.Conv2d(planes, planes, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(planes),

            nn.Sigmoid()
        )
    def forward(self, x): # 2个3*3 一个5*5
        identity = self.attn(x)

        out = self.relu(x)
        out = self.conv1(out)
        out = self.bn1(out)

        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return out



class ResNet(nn.Module):
    # depth=2,dim=4095,num_patches=49,
    def __init__(self, block, layers, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None,
                 num_patches=7*7, dim=256*4*4, depth=2, heads=4, mlp_dim=512, dim_head=32, dropout=0,base_width=64):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.base_width=64
        self.inplanes = 64
        self.dilation = 1
        if  base_width != 64:
            raise ValueError(' only supports  base_width=64')
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None or"
                             " a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Sequential(nn.Conv2d(3, self.inplanes, kernel_size=3, stride=2, padding=1, bias=False,),
                                   nn.BatchNorm2d(self.inplanes),#沿着通道维度对前一个卷积层输出进行标准化
                                   nn.ReLU(),
                                   nn.Conv2d(self.inplanes, self.inplanes,kernel_size=3,stride=1,padding=1,bias=False),
                                   )


        self.bn1 = norm_layer(self.inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ## Conv1-3 layer
        self.layer1 = self._make_layer(block, 64, layers[0]) # 64*64 inplanes=64
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])#64*128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        ## the dynamic branch of the DSF module

        # SNP network
        self.d_branch=self._make_layer(block, 256, layers[3], stride=2, dilate=replace_stride_with_dilation[1])
        self.s_branch=self._make_layer(block, 256, layers[3], stride=2, dilate=replace_stride_with_dilation[1])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        #self.downsample=Downsample(3,256,torch.rand(256,3,480,480))
        self.patch_embedding=nn.Sequential(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,stride=2,padding=1,bias=False),
                                           nn.BatchNorm2d(256),
                                           nn.ReLU(),
                                           )


        self.cls_token = nn.Parameter(torch.randn(1, 1, 256*4*4))
        self.temporal_pos_embedding = nn.Parameter(torch.randn(1, 16, dim))
        self.pos_embedding_static = nn.Parameter(torch.randn(1, 16, 256))
        self.temporal_transformer = TimeSformer(num_frames=16,
						img_size=7,
						patch_size=1,
						attention_type='divided_space_time',
						use_learnable_pos_emb=True,
						return_cls_token=True)

        #self.s_branch = Transformer(dim=256, depth=2, heads=4, dim_head=32, mlp_dim=256, dropout=0.)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                nn.init.constant_(m.bn2.weight, 0)


    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))
        layers = []

        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))

        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation, norm_layer=norm_layer)) #不带下采样

        return nn.Sequential(*layers)


    def forward(self, x):
        """x张量对象，contiguous用于确保张量在内存中是连续存储，因为某些操作，可能张量不再连续村粗，例如inpalce,view对张量进行形状重塑为
        思维张量，-1根据其它维度大小自动推断该维度大小，3，112，112剩余维度大小，第一个维度可能是批次大小，后续表示图像通道核空间维度"""

        x = x.contiguous().view(-1, 3, 112, 112)
        x = self.conv1(x)
        x = self.relu(x) 
        x = self.bn1(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x) #[512,256,7,7]
        ##dynamic static fusion module
        b, c, h, w = x.shape  #
        act = ((x.view(b // 16, 16, c, h, w))[:, 1:16, :, :] - (x.view(b // 16, 16, c, h, w))[:, 0:15, :, :]).view(-1, c, h,w)  #[480,256,7,7]
        x_d = self.d_branch(act)#[480, 256, 4, 4]
        sta= (x.view(b // 16, 16, c, h, w))[:, 0:15, :, :].contiguous().view(-1, c, 7, 7)  #[480,256,7,7]
        x_s= self.s_branch(sta) #[480, 256, 4, 4]

        x=x_d+x_s #[480, 256, 4, 4]
        x = F.interpolate(x, size=(7, 7), mode='bilinear', align_corners=False) #[480, 256, 7, 7]

        b, c, h, w = x.shape
        x = x.reshape((b // 15, 15, c,h*w))  # [32,15,256,49]
        DCT = torch.mean(x, dim=1).unsqueeze(1)  # [32,1,256,49]
        x = torch.cat((DCT, x), dim=1)#[32,16,256,49]

        b,f,c,d = x.shape #[32,16,256,16]      [32,16,4096]

        x = x.reshape((b,f,c,d//7, 7)) #[32,16,256,7,7]

        x = self.temporal_transformer(x) #[32,512]

        return x



def backbone():
    return ResNet(BasicBlock, [1,1,1,3]) #


if __name__ == '__main__':# input[32，3，112，112】=1204224
    img = torch.randn((1, 16, 3, 112, 112)) # 输入通道16
    model = backbone()
    model(img)
