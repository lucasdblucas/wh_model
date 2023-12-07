'''It is a modified version of the unofficial implementaion of 
'Wide Residual Networks'
Paper: https://arxiv.org/abs/1605.07146
Code: https://github.com/xternalz/WideResNet-pytorch

MIT License
Copyright (c) 2020 Ivan Sosnovik, Michał Szmaja
Copyright (c) 2019 xternalz
'''
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

from .impl.ses_conv import SESConv_H_H, SESConv_Z2_H, SESConv_H_H_1x1, SESMaxProjection

# def get_resnet50(num_out):
    
#     model = wide_resnet50_2(pretrained=False)
#     num_ftrs = model.fc.in_features
    
#     fc_weight = torch.nn.Linear(num_ftrs, num_out/2)
#     fc_height = torch.nn.Linear(num_ftrs, num_out/2)
    
#     fc_list = torch.nn.ModuleList([fc_weight, fc_height])
    
#     return model

# class WideResNetWithTwoOutputs(torch.nn.Module):
#     def __init__(self, model, fc_list):
#         super().__init__()
#         self.model = model
#         self.fc_list = fc_list

#     def forward(self, x):
#         x = self.model(x)
#         outputs = []
#         for fc in self.fc_list:
#             outputs.append(fc(x))
#         return torch.cat(outputs, dim=1)

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, scales=[1.0], pool=False, interscale=False, basis_type='A'):
        super(BasicBlock, self).__init__()
        
        ###
        # in_planes - the number of input planes. The corresponding to the input channels.
        # out_planes - the number of output planes. The corresponding to the output channels.
        ###
        self.bn1 = nn.BatchNorm3d(in_planes)
        ###
        # inplace=True - nn.ReLU(inplace=True) saves memory during both training and testing.
        # https://stackoverflow.com/questions/69913781/is-it-true-that-inplace-true-activations-in-pytorch-make-sense-only-for-infere
        ###
        self.relu1 = nn.ReLU(inplace=True)
        ###
        # pool=True - Define how the first layer from the block will be. In this case pool is True, the first layer is comprised with one projection layer
        # followed by a conv2d_z2_H layer.
        # The projection layer will squeeze the scale dimension. So a Conv_H_H will not be suitable.
        ###
        # SESMaxProjection() first?
        ##
        if pool:
            self.conv1 = nn.Sequential(
                SESMaxProjection(),
                SESConv_Z2_H(in_planes, out_planes, kernel_size=7, effective_size=3, stride=stride, padding=3, bias=False, scales=scales, basis_type=basis_type) # default to 'A'
            )
        ##
        # interscale=True - Witch indicate that the input has the scale dimension.
        # interscale=False - ? the difference is in scale_size. Definition: scale_size: Size of scale filter.
        ##
        else:
            if interscale:
                self.conv1 = SESConv_H_H(in_planes, out_planes, 2, kernel_size=7, effective_size=3, stride=stride, padding=3, bias=False, scales=scales, basis_type=basis_type)
            else:
                self.conv1 = SESConv_H_H(in_planes, out_planes, 1, kernel_size=7, effective_size=3, stride=stride, padding=3, bias=False, scales=scales, basis_type=basis_type)
        
        self.bn2 = nn.BatchNorm3d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = SESConv_H_H(out_planes, out_planes, 1, kernel_size=7, effective_size=3, stride=1, padding=3, bias=False, scales=scales, basis_type=basis_type)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and SESConv_H_H_1x1(in_planes, out_planes, stride=stride, bias=False, num_scales=len(scales)) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
            
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        
        out = self.conv2(out)
        
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, scales=[0.0], pool=False, interscale=False, basis_type='A'):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, scales, pool, interscale, basis_type)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, scales, pool, interscale, basis_type):
        layers = []
        for i in range(nb_layers):
            pool_layer = pool and (i == 0)
            interscale_layer = interscale and (i == 0)
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, scales, pool=pool_layer, interscale=interscale_layer, basis_type=basis_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNetSplit(nn.Module):
    def __init__(self, img_channel, depth, widen_factor=1, dropRate=0.0, scales=[1.0], pools=[False, False, False], interscale=[False, False, False], basis_type="A"):
        super(WideResNetSplit, self).__init__()
        ##
        # nChannels - number of channels per block.
        ##
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        ##
        # This restriction is imposed by the WideResnet authors -> 'depth should be 6n+4'.
        # I think, that it is has something to do with optime rate between d (number of ResNetBlocks) and k (widening factor)
        # Then the function receives depth, witch is the total number of convolutional layers and it has to be calculated n (depth_factor), witch is the number of convolutional layer per block.
        # My question is, if I want to change the depth or block quantities, this property continues? We'll see.
        ##
        assert((depth - 4) % 6 == 0)
        ##
        # n - (deep_factor) number of conv layers, in this case n = 2 layers per block.
        ##
        n = (depth - 4) // 6
        ##
        # block - is the function BasicBlock, witch will be called inside NetworkBlock with a NetworkBlock configuration.
        ##
        block = BasicBlock
        ##
        # 1st conv before any network block
        # in this case:
        ##
        """img_channel=1, nChannels[0]=16, kernel_size=7x7, effective_size=3, stride=1, padding=3, bias=False, scales=scales, basis_type='A' # compared to the WideResnet, the filters didn't exceed 3x3 size. But in scale-equivariant this is different. 
        The difference between 'kernel_size' and 'effective_size'
        """
        self.conv1 = SESConv_Z2_H(img_channel, nChannels[0], kernel_size=7, effective_size=3, stride=1, padding=3, bias=False, scales=scales, basis_type=basis_type) #nChannels[0] = 16 
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, scales=scales, pool=pools[0], interscale=interscale[0], basis_type=basis_type)  ##nChannels[1] = 16 * 8 = 128
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, scales=scales, pool=pools[1], interscale=interscale[1], basis_type=basis_type) #nChannels[2] = 32 * 8 = 256
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, scales=scales, pool=pools[2], interscale=interscale[2], basis_type=basis_type) #nChannels[3] = 64 * 8 = 512
        
        # global average pooling and weight and height predictor
        self.proj = SESMaxProjection()
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc_weight = nn.Linear(nChannels[3], 1)
        self.fc_height = nn.Linear(nChannels[3], 1)
        
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, (SESConv_H_H, SESConv_Z2_H, SESConv_H_H_1x1)):
                nelement = m.weight.nelement()
                n = nelement / m.in_channels
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.proj(out)
        out = self.relu(self.bn1(out))

        out = F.adaptive_avg_pool2d(out, 1)
        ##
        # view - Returns a new tensor with the same data as the self tensor but of a different shape.
        # In this case, for what I notice, It is used for Flatten the tensor.
        ##
        out = out.view(-1, self.nChannels)
        
        out_weight = self.fc_weight(out)
        out_height = self.fc_height(out)
        
        return out_weight, out_height
    
class WideResNet(nn.Module):
    def __init__(self, img_channel, depth, num_classes, widen_factor=1, dropRate=0.0, scales=[1.0], pools=[False, False, False], interscale=[False, False, False]):
        super(WideResNet, self).__init__()
        ##
        # nChannels - 
        ##
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        ##
        # depth - Why this restriction for depth?
        # The architecture has 4 blocks, but why the difference has to be divisible for 6?
        ##
        assert((depth - 4) % 6 == 0)
        ##
        # n - number of conv layers
        ##
        n = (depth - 4) // 6
        ##
        # block - is the function BasicBlock, witch will be called inside NetworkBlock with a NetworkBlock configuration.
        ##
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = SESConv_Z2_H(img_channel, nChannels[0], kernel_size=7, effective_size=3, stride=1, padding=3, bias=False, scales=scales, basis_type='A')
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate, scales=scales, pool=pools[0], interscale=interscale[0])
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, scales=scales, pool=pools[1], interscale=interscale[1])
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, scales=scales, pool=pools[2], interscale=interscale[2])
        
        # global average pooling and weight and height predictor
        self.proj = SESMaxProjection()
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, (SESConv_H_H, SESConv_Z2_H, SESConv_H_H_1x1)):
                nelement = m.weight.nelement()
                n = nelement / m.in_channels
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.proj(out)
        out = self.relu(self.bn1(out))

        out = F.adaptive_avg_pool2d(out, 1)
        ##
        # view - Returns a new tensor with the same data as the self tensor but of a different shape.
        # In this case, for what I notice, It is used for Flatten the tensor.
        ##
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out


#RGB num_chanel = 3
def wh_wrn_16_8_ses(**kwargs): 
    transform_config = kwargs["current_project_info"]["data_config"]["transform_config"]
    deep_config = kwargs["current_project_info"]["train_infe_config"]["deep_config"]
    
    scales = [0.9 * 1.41**i for i in range(3)]
    grayscale = transform_config["grayscale"]["use"]
    img_channel = 1 if grayscale else 3
    dropout = deep_config["dropout"]["use"][0] #* the first from the list
    task_config = kwargs["current_project_info"]["task_config"]
    regression_config = task_config["regression_config"]
    
    num_classes = len(regression_config["targets"]["use"])
    
    return WideResNet(img_channel=img_channel, depth=16, num_classes=num_classes, widen_factor=8, dropRate=dropout, scales=scales, pools=[False, True, True])

# Gray Scale image_chanel = 1
def wh_split_wrn_16_8_ses(**kwargs):
    transform_config = kwargs["current_project_info"]["data_config"]["transform_config"]
    deep_config = kwargs["current_project_info"]["train_infe_config"]["deep_config"]
    
    scales = [0.9 * 1.41**i for i in range(3)]
    grayscale = transform_config["grayscale"]["use"]
    img_channel = 1 if grayscale else 3
    dropout = deep_config["dropout"]["use"][0] #* the first from the list
    
    return WideResNetSplit(img_channel=img_channel, depth=16, widen_factor=8, dropRate=dropout, scales=scales, pools=[False, True, True], basis_type="A")