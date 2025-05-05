import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cbam import CBAM
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.cbam = CBAM(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.cbam(out)
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class CBAMPoseResNet(nn.Module):
    def __init__(self, block, layers, heads, head_conv):
        super(CBAMPoseResNet, self).__init__()
        self.inplanes = 64
        self.heads = heads
        self.deconv_with_bias = False

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.conv_up_level1 = nn.Conv2d(768, 256, kernel_size=1, stride=1, padding=0)
        self.conv_up_level2 = nn.Conv2d(384, 128, kernel_size=1, stride=1, padding=0)
        self.conv_up_level3 = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0)

        fpn_channels = [256, 128, 64]
        for fpn_idx, fpn_c in enumerate(fpn_channels):
            for head in sorted(self.heads):
                num_output = self.heads[head]
                if head_conv > 0:
                    fc = nn.Sequential(
                        nn.Conv2d(fpn_c, head_conv, kernel_size=3, padding=1),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(head_conv, num_output, kernel_size=1))
                else:
                    fc = nn.Conv2d(fpn_c, num_output, kernel_size=1)
                self.__setattr__('fpn{}_{}'.format(fpn_idx, head), fc)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        _, _, h, w = x.size()
        hm_h, hm_w = h // 4, w // 4
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        up1 = F.interpolate(out4, scale_factor=2, mode='bilinear', align_corners=True)
        concat1 = torch.cat((up1, out3), dim=1)
        up2 = F.interpolate(self.conv_up_level1(concat1), scale_factor=2, mode='bilinear', align_corners=True)
        concat2 = torch.cat((up2, out2), dim=1)
        up3 = F.interpolate(self.conv_up_level2(concat2), scale_factor=2, mode='bilinear', align_corners=True)
        up4 = self.conv_up_level3(torch.cat((up3, out1), dim=1))

        ret = {}
        for head in self.heads:
            temp_outs = []
            for idx, input_feat in enumerate([up2, up3, up4]):
                out = self.__getattr__('fpn{}_{}'.format(idx, head))(input_feat)
                if out.size(2) != hm_h or out.size(3) != hm_w:
                    out = F.interpolate(out, size=(hm_h, hm_w))
                temp_outs.append(out)
            out_cat = torch.cat([o.unsqueeze(-1) for o in temp_outs], dim=-1)
            softmax_out = F.softmax(out_cat, dim=-1)
            ret[head] = (out_cat * softmax_out).sum(dim=-1)

        return ret

    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            for fpn_idx in [0, 1, 2]:
                for head in self.heads:
                    final_layer = self.__getattr__('fpn{}_{}'.format(fpn_idx, head))
                    for m in final_layer.modules():
                        if isinstance(m, nn.Conv2d):
                            if m.weight.shape[0] == self.heads[head]:
                                if 'hm' in head:
                                    nn.init.constant_(m.bias, -2.19)
                                else:
                                    nn.init.normal_(m.weight, std=0.001)
                                    nn.init.constant_(m.bias, 0)
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)

resnet_spec = {
    18: (BasicBlock, [2, 2, 2, 2]),
    34: (BasicBlock, [3, 4, 6, 3]),
    50: (Bottleneck, [3, 4, 6, 3]),
    101: (Bottleneck, [3, 4, 23, 3]),
    152: (Bottleneck, [3, 8, 36, 3])
}

def get_pose_net(num_layers, heads, head_conv, imagenet_pretrained):
    block_class, layers = resnet_spec[num_layers]
    model = CBAMPoseResNet(block_class, layers, heads, head_conv=head_conv)
    model.init_weights(num_layers, pretrained=imagenet_pretrained)
    return model
