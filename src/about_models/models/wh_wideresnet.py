import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None
        
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
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)

class WideResNetTwoOutputs(nn.Module):
    def __init__(self, img_channel, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNetTwoOutputs, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(in_channels=img_channel, out_channels=nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        
        out_features = int(num_classes/2)
        self.fc_weight = nn.Linear(in_features=nChannels[3], out_features=out_features)
        self.fc_height = nn.Linear(nChannels[3], out_features=out_features)
        
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        
        out = out.view(-1, self.nChannels)
        
        out_weight = self.fc_weight(out)
        out_height = self.fc_height(out)
        
        return out_weight, out_height, None

class WideResNetOneOutput(nn.Module):
    def __init__(self, img_channel, depth, num_classes, widen_factor=1, dropRate=0.0):
        super(WideResNetOneOutput, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(in_channels=img_channel, out_channels=nChannels[0], kernel_size=3, stride=1, padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        
        self.fc = nn.Linear(nChannels[3], out_features=num_classes)
        
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
                
    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1)
        
        out = out.view(-1, self.nChannels)
        
        out = self.fc(out)
                
        return out
    
def wh_split_wrn_16_8(**kwargs):
    transform_config = kwargs["current_project_info"]["data_config"]["transform_config"]
    deep_config = kwargs["current_project_info"]["train_infe_config"]["deep_config"]
    task_config = kwargs["current_project_info"]["task_config"]
    regression_config = task_config["regression_config"]
    
    num_classes = len(regression_config["targets"]["use"])
    grayscale = transform_config["grayscale"]["use"]
    img_channel = 1 if grayscale else 3
    dropout = deep_config["dropout"]["use"][0] #* the first from the list
    
    return WideResNetTwoOutputs(
        img_channel=img_channel, 
        depth=16,
        num_classes=num_classes,
        widen_factor=8, 
        dropRate=dropout
    )

def wh_wrn_16_8(**kwargs):
    transform_config = kwargs["current_project_info"]["data_config"]["transform_config"]
    deep_config = kwargs["current_project_info"]["train_infe_config"]["deep_config"]
    task_config = kwargs["current_project_info"]["task_config"]
    regression_config = task_config["regression_config"]
    
    num_classes = len(regression_config["targets"]["use"])
    grayscale = transform_config["grayscale"]["use"]
    img_channel = 1 if grayscale else 3
    dropout = deep_config["dropout"]["use"][0] #* the first from the list
    
    return WideResNetOneOutput(
        img_channel=img_channel, 
        depth=16,
        num_classes=num_classes,
        widen_factor=8, 
        dropRate=dropout
    )