import torch
from torch import nn
from models.modules import backbone


class GenerateModel(nn.Module):
    def __init__(self):
        super(GenerateModel,self).__init__()
        # 输入通道改变
        self.s_former = backbone( ) # ResNet18
        #self.fc = nn.Linear(256*4*4, 7)#4096
        self.fc = nn.Linear(512,7)

    def forward(self, x):
        x= self.s_former(x) #[32,512]
     
        ##feature before fc layer 
        x_FER=x # [32,512]

        #x_FER = x[:, 0] #torch.Size([32])
        #x_FER = x_FER.unsqueeze(0)#torch.Size([1,32])

        ##output
        x=self.fc(x) #[32,7]
        return x,x_FER


if __name__ == '__main__':
    img = torch.randn((1, 16, 3, 112, 112))
    model = GenerateModel()
    model(img)
