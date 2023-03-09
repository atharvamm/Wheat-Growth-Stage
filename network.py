import torch.nn as nn
import torch

class BasicBlock(nn.Module):
    def __init__(self,inplanes,downsample) -> None:
        super(BasicBlock,self).__init__()
        self.in_planes = inplanes
        self.dilation = 1
        self.stride = 1
        self.kernel_size = 3
        self.padding = 1

        self.basicblock = nn.Sequential(
        nn.Conv2d(in_channels = self.in_planes,out_channels = self.in_planes,kernel_size = self.kernel_size,dilation = self.dilation,stride =  self.stride, padding= self.padding),
        nn.BatchNorm2d(self.in_planes),
        nn.ReLU(),
        nn.Conv2d(in_channels = self.in_planes,out_channels = self.in_planes,kernel_size = self.kernel_size, dilation = self.dilation, stride = self.stride,padding= self.padding))


    def forward(self,x):
        residual = x      
        x = self.basicblock(x)
        x = x + residual
        return x

class BottleNeck(nn.Module):
    def __init__(self) -> None:
        super(BottleNeck,self).__init__()

        self.in_planes = 3
        self.stride = 1
        self.dilation = 1
        self.padding = 1

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels = self.in_planes,out_channels = 64,kernel_size=1,dilation = self.dilation,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64,out_channels = 64,kernel_size= 3,dilation = self.dilation,stride=1,padding=self.padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = 64, out_channels = self.in_planes, kernel_size = 1, dilation = self.dilation,stride=1))

    def forward(self,x):
        residual = x
        x = self.bottleneck(x)
        x = x+residual
        return x


# Tip: In resnet instead of padding the input or the output pad the convolutions operations.

class ResNet(nn.Module):
    def __init__(self,num_classes = 7) -> None:
        super(ResNet,self).__init__()
        self.num_classes = num_classes

        self.main_network = nn.Sequential(
            BottleNeck(),
        )
        self.cls_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(start_dim=1),
            nn.Linear(3,num_classes),
        )


    def forward(self,x):
        residual = x
        x = self.main_network(x)
        x = self.cls_layer(x)
        # x = x+residual
        return x
