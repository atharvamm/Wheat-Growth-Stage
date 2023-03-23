import torch.nn as nn
import torch

class BasicBlock(nn.Module):
    def __init__(self,inplanes) -> None:
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
    def __init__(self,inplanes,out_planes) -> None:
        super(BottleNeck,self).__init__()

        self.in_planes = inplanes
        self.out_planes = out_planes
        self.stride = 1
        self.dilation = 1
        self.padding = 1

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels = self.in_planes,out_channels = self.in_planes,kernel_size=1,dilation = self.dilation,stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = self.in_planes,out_channels = self.in_planes,kernel_size= 3,dilation = self.dilation,stride=1,padding=self.padding),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels = self.in_planes, out_channels = self.out_planes, kernel_size = 1, dilation = self.dilation,stride=1))

    def forward(self,x):
        residual = x
        x = self.bottleneck(x)
        x = x+residual
        return x


# Tip: In resnet instead of padding the input or the output pad the convolutions operations.

class ResNet34(nn.Module):
    def __init__(self,num_classes = 7,num_blocks = [3,4,6,3],num_channels = [64,128,256,512],) -> None:
        super(ResNet34,self).__init__()
        self.num_classes = num_classes
        self.channels = num_channels
        self.blocks = num_blocks

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=self.channels[0],kernel_size=7,padding=0,stride=2),
            nn.MaxPool2d(kernel_size=3,stride=2),
        )

        self.conv2x = self.__make_layer(block_type=BasicBlock,num_planes=self.channels[0],num_blocks=self.blocks[0])
        self.downsample1 = self.__downsample_layer(in_planes=self.channels[0],out_planes=self.channels[1])
        
        self.conv3x = self.__make_layer(block_type=BasicBlock,num_planes=self.channels[1],num_blocks=self.blocks[1])
        self.downsample2 = self.__downsample_layer(in_planes=self.channels[1],out_planes=self.channels[2])
        
        self.conv4x = self.__make_layer(block_type=BasicBlock,num_planes=self.channels[2],num_blocks=self.blocks[2])
        self.downsample3 = self.__downsample_layer(in_planes=self.channels[2],out_planes=self.channels[3])
        
        self.conv5x = self.__make_layer(block_type=BasicBlock,num_planes=self.channels[3],num_blocks=self.blocks[3])

        self.network = nn.Sequential(*[self.stem,self.conv2x,self.downsample1,self.conv3x,self.downsample2,self.conv4x,self.downsample3,self.conv5x])
        
        self.cls_layer = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(start_dim=1),
            nn.Linear(self.channels[-1],num_classes),
        )

    def __make_layer(self,block_type = BasicBlock,num_planes = 3,num_blocks = 3):
        blocks = []
        for i in range(num_blocks):
            blocks.append(block_type(inplanes = num_planes))
        return nn.Sequential(*blocks)

    def __downsample_layer(self,in_planes,out_planes,):
        return nn.Conv2d(in_channels=in_planes,out_channels=out_planes,kernel_size=4,padding=1,stride=2)

    def forward(self,x):
        # print(x.shape)
        x = self.network(x)
        x = self.cls_layer(x)
        return x

# Implement Resnet 34
# 1. 7×7, 64, stride 2
# 2. 3×3 max pool, stride 2
# 3. 3*3 conv 64 3 blocks
# 4. 3*3 conv 256 4 blocks
# 5. 3*3 conv 256 6 blocks
# 6. 3*3 conv 512 3 blocks
# 7. avg_pool
# 8. fc to numclasses
# 9. Check difference between dotted lines and full lines.



# class ResNet50(nn.Module):
#     def __init__(self,num_classes = 7,num_blocks = [3,4,6,3],in_channels = [64,128,256,512],out_channels = [256,512,1024,2048]) -> None:
#         super(ResNet50,self).__init__()
#         self.num_classes = num_classes
#         self.channels = num_channels
#         self.blocks = num_blocks

#         self.stem = nn.Sequential(
#             nn.Conv2d(in_channels=3,out_channels=self.channels[0],kernel_size=7,padding=0,stride=2),
#             nn.MaxPool2d(kernel_size=3,stride=2),
#         )

#         self.conv2x = self.__make_layer(block_type=BasicBlock,num_planes=self.channels[0],num_blocks=self.blocks[0])
#         self.downsample1 = self.__downsample_layer(in_planes=self.channels[0],out_planes=self.channels[1])
        
#         self.conv3x = self.__make_layer(block_type=BasicBlock,num_planes=self.channels[1],num_blocks=self.blocks[1])
#         self.downsample2 = self.__downsample_layer(in_planes=self.channels[1],out_planes=self.channels[2])
        
#         self.conv4x = self.__make_layer(block_type=BasicBlock,num_planes=self.channels[2],num_blocks=self.blocks[2])
#         self.downsample3 = self.__downsample_layer(in_planes=self.channels[2],out_planes=self.channels[3])
        
#         self.conv5x = self.__make_layer(block_type=BasicBlock,num_planes=self.channels[3],num_blocks=self.blocks[3])

#         self.network = nn.Sequential(*[self.stem,self.conv2x,self.downsample1,self.conv3x,self.downsample2,self.conv4x,self.downsample3,self.conv5x])
        
#         self.cls_layer = nn.Sequential(
#             nn.AdaptiveAvgPool2d(output_size=(1,1)),
#             nn.Flatten(start_dim=1),
#             nn.Linear(self.channels[-1],num_classes),
#         )

#     def __make_layer(self,block_type = BasicBlock,num_planes = 3,num_blocks = 3):
#         blocks = []
#         for i in range(num_blocks):
#             blocks.append(block_type(inplanes = num_planes))
#         return nn.Sequential(*blocks)

#     def __downsample_layer(self,in_planes,out_planes,):
#         return nn.Conv2d(in_channels=in_planes,out_channels=out_planes,kernel_size=4,padding=1,stride=2)

#     def forward(self,x):
#         # print(x.shape)
#         x = self.network(x)
#         x = self.cls_layer(x)
#         return x