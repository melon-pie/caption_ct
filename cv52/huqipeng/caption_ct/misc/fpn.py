import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.ffc = nn.Linear(512, 14)
        self.v = nn.Parameter(torch.FloatTensor(3,1,1))
        self.dp = nn.Dropout(0.5)
        self.relu = nn.ReLU()

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((2,2))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

        self.toplayer = nn.Conv2d(512 * block.expansion, 256, kernel_size=1, stride=1, padding=0)

        self.smooth = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        # self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        # self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.latlayer3 = nn.Conv2d( 256 * block.expansion, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d( 128 * block.expansion, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer1 = nn.Conv2d( 64 * block.expansion , 256, kernel_size=1, stride=1, padding=0)
        self.latlayer4 = nn.Conv2d( 64 * block.expansion , 256, kernel_size=1, stride=1, padding=0)
        self.latlayer0 = nn.Conv2d( 3 , 256, kernel_size=1, stride=1, padding=0)
        


    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        return F.interpolate(x, size=(H,W), mode='bilinear',align_corners=True) + y

    def _downsample_add(self, x, y):
        _,_,H,W = y.size()
        return nn.AdaptiveAvgPool2d((H,W))(x) + y

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x, boundary=None, simage=None):

        simage = simage*torch.softmax(self.v,2)
        simage = simage.sum(1).unsqueeze(1)
        simage = torch.cat((simage,simage,simage),1)
        x = x * simage 
        
            
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        l1 = self.layer1(x)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        # p4 = self.toplayer(l4)        
        # p3 = self._upsample_add(p4, self.latlayer3(l3))
        # p2 = self._upsample_add(p3, self.latlayer2(l2))
        # p1 = self._upsample_add(p2, self.latlayer1(l1))

        p1 = self._downsample_add(self.latlayer0(boundary), self.latlayer1(l1))
        p2 = self._downsample_add(p1, self.latlayer2(l2))
        p3 = self._downsample_add(p2, self.latlayer3(l3))
        p4 = self._downsample_add(p3, self.toplayer(l4))

        #bottom-up
        # p2 = self._downsample_add(self.latlayer1(l1), self.latlayer2(l2))
        # p3 = self._downsample_add(p2, self.latlayer3(l3))
        # p4 = self._downsample_add(p3, self.toplayer(l4))

        # p4 = self.smooth1(p4)
        x = self.smooth(p4)
        # p2 = self.smooth3(p2)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # x = self.dp(x)
        # x = self.ffc(x)
        

        return x


def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

def resnet50(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)
