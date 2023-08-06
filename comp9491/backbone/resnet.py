import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200'
]
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
    'resnext':'https://download.pytorch.org/models/resnext50_32x4d-1a0047aa.pth',
    'mobilenet_v2':'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    
}


class Get_Correlation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        reduction_channel = channels//16
        self.down_conv = nn.Conv3d(channels, reduction_channel, kernel_size=1, bias=False)

        self.down_conv2 = nn.Conv3d(channels, channels, kernel_size=1, bias=False)
        self.spatial_aggregation1 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,1,1), groups=reduction_channel)
        self.spatial_aggregation2 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,2,2), dilation=(1,2,2), groups=reduction_channel)
        self.spatial_aggregation3 = nn.Conv3d(reduction_channel, reduction_channel, kernel_size=(9,3,3), padding=(4,3,3), dilation=(1,3,3), groups=reduction_channel)
        self.weights = nn.Parameter(torch.ones(3) / 3, requires_grad=True)
        self.weights2 = nn.Parameter(torch.ones(2) / 2, requires_grad=True)
        self.conv_back = nn.Conv3d(reduction_channel, channels, kernel_size=1, bias=False)

    def forward(self, x):

        x2 = self.down_conv2(x)
        affinities = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat([x2[:,:,1:], x2[:,:,-1:]], 2))  # repeat the last frame
        affinities2 = torch.einsum('bcthw,bctsd->bthwsd', x, torch.concat([x2[:,:,:1], x2[:,:,:-1]], 2))  # repeat the first frame 
        features = torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:,:,1:], x2[:,:,-1:]], 2), F.sigmoid(affinities)-0.5 )* self.weights2[0] + \
            torch.einsum('bctsd,bthwsd->bcthw', torch.concat([x2[:,:,:1], x2[:,:,:-1]], 2), F.sigmoid(affinities2)-0.5 ) * self.weights2[1] 

        x = self.down_conv(x)
        aggregated_x = self.spatial_aggregation1(x)*self.weights[0] + self.spatial_aggregation2(x)*self.weights[1] \
                    + self.spatial_aggregation3(x)*self.weights[2]
        aggregated_x = self.conv_back(aggregated_x)

        return features * (F.sigmoid(aggregated_x)-0.5)
        
#resnet.............

def conv3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(1,3,3),
        stride=(1,stride,stride),
        padding=(0,1,1),
        bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
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
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1,7,7), stride=(1,2,2), padding=(0,3,3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.corr1 = Get_Correlation(self.inplanes)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.corr2 = Get_Correlation(self.inplanes)
        self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.corr3 = Get_Correlation(self.inplanes)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
            
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=(1,stride,stride), bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('!!!!!!!!!!!!:  ',x.shape)
        N, C, T, H, W = x.size()
        x = self.conv1(x)
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x) 
        x = x + self.corr1(x) * self.alpha[0]
        x = self.layer3(x)
        x = x + self.corr2(x) * self.alpha[1]
        x = self.layer4(x)
        x = x + self.corr3(x) * self.alpha[2]
        
        # print('!!!!!!!!!!!!:  ',x.shape)
        x = x.transpose(1,2).contiguous()
        
        x = x.view((-1,)+x.size()[2:]) #b,t,c,h,w

        x = self.avgpool(x)
        
        x = x.view(x.size(0), -1) #bt,c
        x = self.fc(x) #bt,c
        # print('!!!!!!!!!!!!:  ',x.shape)
        return x

def resnet18(**kwargs):
    """Constructs a ResNet-18 based model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet18'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)  
    model.load_state_dict(checkpoint, strict=False)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet34'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)  
    model.load_state_dict(checkpoint, strict=False)
    return model

def resnet101(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 23, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet101'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)  
    model.load_state_dict(checkpoint, strict=False)
    return model


#mobilenet......................



def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv3d(inp, oup, kernel_size=(1,3,3), stride=stride, padding=(0,1,1), bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm3d(oup),
        nn.ReLU6(inplace=True)
    )

class Block_mobile(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(Block_mobile, self).__init__()
        self.stride = stride

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == (1,1,1) and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, (1,3,3), stride, (0,1,1), groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv3d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv3d(hidden_dim, hidden_dim, (1,3,3), stride, (0,1,1), groups=hidden_dim, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv3d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm3d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
 
 
    
class MobileNetV2(nn.Module):
    def __init__(self,block, num_classes=1000):
        super(MobileNetV2, self).__init__()
        self.inplanes = 32
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, (1, 1, 1)],
            [6, 24, 2, (1, 2, 2)],
            [6, 32, 3, (1, 2, 2)],
            [6, 64, 4, (1, 2, 2)],
            [6, 96, 3, (1, 1, 1)],
            [6, 160, 3, (1, 2, 2)],
            [6, 320, 1, (1, 1, 1)],
        ]
 
        self.conv1 = conv_bn(3, self.inplanes, stride=(1, 2, 2))
        
        self.layer1 = self._make_layers_mobile(block, self.cfgs[0])
        self.layer2 = self._make_layers_mobile(block, self.cfgs[1])
        self.corr1 = Get_Correlation(self.inplanes)
        self.layer3 = self._make_layers_mobile(block,self.cfgs[2])
        self.corr2 = Get_Correlation(self.inplanes)
        self.layer4 = self._make_layers_mobile(block,self.cfgs[3])
        self.corr3 = Get_Correlation(self.inplanes)
        self.layer5 = self._make_layers_mobile(block,self.cfgs[4])
        self.corr4 = Get_Correlation(self.inplanes)
        self.layer6 = self._make_layers_mobile(block,self.cfgs[5])
        self.corr5 = Get_Correlation(self.inplanes)
        self.layer7= self._make_layers_mobile(block,self.cfgs[6])
        self.corr6 = Get_Correlation(self.inplanes)
        self.alpha = nn.Parameter(torch.zeros(6), requires_grad=True)
        
        self.avgpool = nn.AvgPool2d(7, stride=1)
        
        # self.layers = self._make_layers_mobile(in_planes=64)
        
        self.conv2 = conv_1x1x1_bn(self.inplanes, 1280)
        
        self.linear = nn.Linear(1280, num_classes)
 


    def _make_layers_mobile(self,block,config):
        layers = []
        t, c, n, s=config[0],config[1],config[2],config[3]
        for i in range(n):
            stride = s if i == 0 else (1, 1, 1)
            layers.append(block(self.inplanes, c, stride,t ))
            self.inplanes = c
        return nn.Sequential(*layers)
 


    def forward(self, x):
        # print('!!!!!!!!!!!!:  ',x.shape)
        out = self.conv1(x)
        # out = self.layers(out)
        
        
        out = self.layer1(out)
        
        out = self.layer2(out) 
        # print('!!!!!!!!!!!!:  ',out.shape)
        out = out + self.corr1(out) * self.alpha[0]
        
        out = self.layer3(out)
        out = out + self.corr2(out) * self.alpha[1]
        out = self.layer4(out)
        out = out + self.corr3(out) * self.alpha[2]
        out = self.layer5(out) 
        out = out + self.corr4(out) * self.alpha[3]
        out = self.layer6(out)
        out = out + self.corr5(out) * self.alpha[4]
        out = self.layer7(out)
        out = out + self.corr6(out) * self.alpha[5]
        
        

        # print('!!!!!!!!!!!!:  ',out.shape)
        out = self.conv2(out)
        # print('!!!!!!!!!!!!:  ',out.shape)
        
        out = out.transpose(1,2).contiguous()
        out = out.view((-1,)+out.size()[2:]) #bt,c,h,w
        out = self.avgpool(out)
        out = out.view(out.size(0), -1) #bt,c
        # print(out.shape)
        # out = F.avg_pool3d(out, out.data.size()[-3:])
        # print('!!!!!!!!!!!!:  ',out.shape)
        # out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return out








def mobilenet_v2(**kwargs):

    model = MobileNetV2(Block_mobile, **kwargs)
    return model





#resnext.....................


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class ResNeXtBottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, cardinality, stride=1,
                 downsample=None):
        super(ResNeXtBottleneck, self).__init__()
        mid_planes = cardinality * int(planes / 32)
        self.conv1 = nn.Conv3d(inplanes, mid_planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(mid_planes)
        self.conv2 = nn.Conv3d(
            mid_planes,
            mid_planes,
            kernel_size=(1,3,3),
            stride=(1,stride,stride),
            padding=(0,1,1),
            groups=cardinality,
            bias=False)
        self.bn2 = nn.BatchNorm3d(mid_planes)
        self.conv3 = nn.Conv3d(
            mid_planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
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


class ResNeXt(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 shortcut_type='B',
                 cardinality=32,
                 n_classes=512,
                 ):
        self.inplanes = 64
        super(ResNeXt, self).__init__()
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=(1,7,7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False)
        #self.conv1 = nn.Conv3d(
        #    3,
        #    64,
        #    kernel_size=(3,7,7),
        #    stride=(1, 2, 2),
        #    padding=(1, 3, 3),
        #    bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1,2,2), padding=(0,1,1))
        self.layer1 = self._make_layer(block, 128, layers[0], shortcut_type,
                                       cardinality)
        
        self.layer2 = self._make_layer(
            block, 256, layers[1], shortcut_type, cardinality, stride=2)
        self.corr1 = Get_Correlation(self.inplanes)
        self.layer3 = self._make_layer(
            block, 512, layers[2], shortcut_type, cardinality, stride=2)
        self.corr2 = Get_Correlation(self.inplanes)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], shortcut_type, cardinality, stride=2)
        self.corr3 = Get_Correlation(self.inplanes)
        
        
        self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc1 = nn.Linear(cardinality * 32 * block.expansion,n_classes )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    shortcut_type,
                    cardinality,
                    stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=(1,stride,stride))
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=(1,stride,stride),
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(self.inplanes, planes, cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print('!!!!!!!!!!!!:  ',x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        
        x = self.layer2(x)
        x = x + self.corr1(x) * self.alpha[0]
        x = self.layer3(x)
        x = x + self.corr2(x) * self.alpha[1]
        x = self.layer4(x)
        x = x + self.corr3(x) * self.alpha[2]
        
        # print('!!!!!!!!!!!!:  ',x.shape)
        x = x.transpose(1,2).contiguous()
        
        x = x.view((-1,)+x.size()[2:]) #b,t,c,h,w

        x = self.avgpool(x)
        x = x.view(x.size(0), -1) #bt,c
        
        
        x = self.fc1(x)
        # print('!!!!!!!!!!!!:  ',x.shape)
        return x





def resnext(**kwargs):
    model = ResNeXt(ResNeXtBottleneck,[3, 4, 6, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnext'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if 'conv' in ln or 'downsample.0.weight' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)  
    model.load_state_dict(checkpoint, strict=False)
    
    return model








'''

Shuffle Net V2
'''


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels,depth, height, width = x.data.size()
    channels_per_group = num_channels // groups
    

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group,depth, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1,depth, height, width)

    return x

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()
        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            print(f"stride is {self.stride}")
            self.branch1 = nn.Sequential(
                nn.Conv3d(inp, inp, kernel_size=(1,3,3), stride=(1,self.stride,self.stride), padding=(0,1,1), groups=inp),  # Depthwise convolution,
                nn.BatchNorm3d(inp),
                nn.Conv3d(inp, branch_features, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False),
                nn.BatchNorm3d(branch_features),
                nn.ReLU(inplace=True),
            )

        else:
            print(f"stride is {self.stride}")
            self.branch1 = nn.Sequential()
            
        self.branch2 = nn.Sequential(
            nn.Conv3d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False),
            nn.BatchNorm3d(branch_features),
            nn.ReLU(inplace=True),

            nn.Conv3d(branch_features, branch_features, kernel_size=(1,3,3), stride=(1,self.stride,self.stride), padding=(0,1,1), groups=branch_features),
    
            nn.BatchNorm3d(branch_features),
            nn.Conv3d(branch_features, branch_features, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False),
            nn.BatchNorm3d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:

            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
          #  print("    Stride not is 1")
            branch1_output = self.branch1(x)
            branch2_output = self.branch2(x)
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = channel_shuffle(out, 2)

        return out


class ShuffleNetV2(nn.Module):
    def __init__(self, stages_repeats, stages_out_channels, num_classes = 512,inverted_residual=InvertedResidual):
        super(ShuffleNetV2, self).__init__()


        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 5:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels
        self.alpha = nn.Parameter(torch.zeros(3), requires_grad=True)
        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size = (1,3,3), padding = (0, 1, 1), stride = (1, 2, 2), bias=False),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.lastmaxpool = nn.MaxPool2d(7,stride = 1)
        self.relu = nn.ReLU(inplace=True)
        input_channels = self._stage_out_channels[0]
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels

        output_channels = self._stage_out_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=False),
            nn.BatchNorm3d(output_channels),
            nn.ReLU(inplace=True),
        )


        self.fc = nn.Linear(output_channels, num_classes)

        self.corr1 = Get_Correlation(stages_out_channels[1])
        self.corr2 = Get_Correlation(stages_out_channels[2])
        self.corr3 = Get_Correlation(stages_out_channels[3])
    
        
    def forward(self,x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = x + self.corr1(x) * self.alpha[0]
        x = self.stage3(x)
        x = x + self.corr2(x) * self.alpha[1]
        x = self.stage4(x)
        x = x + self.corr3(x) * self.alpha[2]
        x = self.conv5(x)
        x = x.transpose(1,2).contiguous()
        x = x.view((-1,)+x.size()[2:]) #bt,c,h,w
        x = self.lastmaxpool(x)
        x = x.view(x.size(0), -1) #bt,c
        x = self.fc(x)
        return x
        
def _shufflenetv2(arch, *args, **kwargs):
    model = ShuffleNetV2(*args, **kwargs)
    return model


def shufflenet_v2_x0_5(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _shufflenetv2('shufflenetv2_x0.5', 
                         [4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['shufflenetv2_x0_5'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if len(checkpoint[ln].size()) == 1:
            continue
        if 'conv' in ln or 'stage' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)  
    model.load_state_dict(checkpoint, strict=False)
    return model
    

def shufflenet_v2_x1_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = _shufflenetv2('shufflenetv2_x1.0', 
                         [4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['shufflenetv2_x1_0'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name :
        if len(checkpoint[ln].size()) == 1:
            continue
        if 'conv' in ln or 'stage' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)  
    model.load_state_dict(checkpoint, strict=False)
    return model


def shufflenet_v2_x1_5(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.5',
                         [4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)


def shufflenet_v2_x2_0(pretrained=False, progress=True, **kwargs):
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x2.0', pretrained, progress,
                         [4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)












def test():
    net = mobilenet_v2()
    y = net(torch.randn(8, 3, 16, 112, 112))
    print(y.size())
# test()