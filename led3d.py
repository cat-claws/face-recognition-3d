import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

_weights_dict = dict()

def load_weights(weight_file):
    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file, allow_pickle=True).item()
    except:
        weights_dict = np.load(weight_file, allow_pickle=True, encoding='bytes').item()

    return weights_dict

class Led3D(nn.Module):

    
    def __init__(self, weight_file = None):
        super(Led3D, self).__init__()
        # global _weights_dict
        # _weights_dict = load_weights(weight_file)

        self.s1_conv = self.__conv(2, name='s1_conv', in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.s1_bn = self.__batch_normalization(2, 's1_bn', num_features=32, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.s2_conv = self.__conv(2, name='s2_conv', in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.s2_bn = self.__batch_normalization(2, 's2_bn', num_features=64, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.s3_conv = self.__conv(2, name='s3_conv', in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.s3_bn = self.__batch_normalization(2, 's3_bn', num_features=128, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.s4_conv = self.__conv(2, name='s4_conv', in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.s4_bn = self.__batch_normalization(2, 's4_bn', num_features=256, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.s5_conv = self.__conv(2, name='s5_conv', in_channels=480, out_channels=960, kernel_size=(3, 3), stride=(1, 1), groups=1, bias=False)
        self.s5_bn = self.__batch_normalization(2, 's5_bn', num_features=960, eps=1.9999999494757503e-05, momentum=0.8999999761581421)
        self.s5_global_conv = self.__conv(2, name='s5_global_conv', in_channels=960, out_channels=960, kernel_size=(8, 8), stride=(8, 8), groups=960, bias=True)
        self.fc = self.__dense(name = 'fc', in_features = 960, out_features = 340, bias = True)

    def forward(self, x):
        s1_conv_pad     = F.pad(x, (1, 1, 1, 1))
        s1_conv         = self.s1_conv(s1_conv_pad)
        s1_bn           = self.s1_bn(s1_conv)
        s1_relu         = F.relu(s1_bn)
        s1_global_pool_pad = F.pad(s1_relu, (16, 16, 16, 16), value=float('-inf'))
        s1_global_pool, s1_global_pool_idx = F.max_pool2d(s1_global_pool_pad, kernel_size=(33, 33), stride=(16, 16), padding=0, ceil_mode=False, return_indices=True)
        pooling0_pad    = F.pad(s1_relu, (1, 1, 1, 1), value=float('-inf'))
        pooling0, pooling0_idx = F.max_pool2d(pooling0_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        s2_conv_pad     = F.pad(pooling0, (1, 1, 1, 1))
        s2_conv         = self.s2_conv(s2_conv_pad)
        s2_bn           = self.s2_bn(s2_conv)
        s2_relu         = F.relu(s2_bn)
        s2_global_pool_pad = F.pad(s2_relu, (8, 8, 8, 8), value=float('-inf'))
        s2_global_pool, s2_global_pool_idx = F.max_pool2d(s2_global_pool_pad, kernel_size=(17, 17), stride=(8, 8), padding=0, ceil_mode=False, return_indices=True)
        s2_pool_pad     = F.pad(s2_relu, (1, 1, 1, 1), value=float('-inf'))
        s2_pool, s2_pool_idx = F.max_pool2d(s2_pool_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        s3_conv_pad     = F.pad(s2_pool, (1, 1, 1, 1))
        s3_conv         = self.s3_conv(s3_conv_pad)
        s3_bn           = self.s3_bn(s3_conv)
        s3_relu         = F.relu(s3_bn)
        s3_global_pool_pad = F.pad(s3_relu, (4, 4, 4, 4), value=float('-inf'))
        s3_global_pool, s3_global_pool_idx = F.max_pool2d(s3_global_pool_pad, kernel_size=(9, 9), stride=(4, 4), padding=0, ceil_mode=False, return_indices=True)
        s3_pool_pad     = F.pad(s3_relu, (1, 1, 1, 1), value=float('-inf'))
        s3_pool, s3_pool_idx = F.max_pool2d(s3_pool_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        s4_conv_pad     = F.pad(s3_pool, (1, 1, 1, 1))
        s4_conv         = self.s4_conv(s4_conv_pad)
        s4_bn           = self.s4_bn(s4_conv)
        s4_relu         = F.relu(s4_bn)
        s4_pool_pad     = F.pad(s4_relu, (1, 1, 1, 1), value=float('-inf'))
        s4_pool, s4_pool_idx = F.max_pool2d(s4_pool_pad, kernel_size=(3, 3), stride=(2, 2), padding=0, ceil_mode=False, return_indices=True)
        feature         = torch.cat((s1_global_pool, s2_global_pool, s3_global_pool, s4_pool,), 1)
        s5_conv_pad     = F.pad(feature, (1, 1, 1, 1))
        s5_conv         = self.s5_conv(s5_conv_pad)
        s5_bn           = self.s5_bn(s5_conv)
        s5_relu         = F.relu(s5_bn)
        s5_global_conv  = self.s5_global_conv(s5_relu)
        drop            = F.dropout(input = s5_global_conv, p = 0.20000000298023224, training = self.training, inplace = True)
        fc              = self.fc(drop.view(drop.size(0), -1))
        softmax         = F.softmax(fc)
        return softmax


    @staticmethod
    def __conv(dim, name, **kwargs):
        if   dim == 1:  layer = nn.Conv1d(**kwargs)
        elif dim == 2:  layer = nn.Conv2d(**kwargs)
        elif dim == 3:  layer = nn.Conv3d(**kwargs)
        else:           raise NotImplementedError()

        # layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        # if 'bias' in _weights_dict[name]:
        #     layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __dense(name, **kwargs):
        layer = nn.Linear(**kwargs)
        # layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['weights']))
        # if 'bias' in _weights_dict[name]:
        #     layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        return layer

    @staticmethod
    def __batch_normalization(dim, name, **kwargs):
        if   dim == 0 or dim == 1:  layer = nn.BatchNorm1d(**kwargs)
        elif dim == 2:  layer = nn.BatchNorm2d(**kwargs)
        elif dim == 3:  layer = nn.BatchNorm3d(**kwargs)
        else:           raise NotImplementedError()

        # if 'scale' in _weights_dict[name]:
        #     layer.state_dict()['weight'].copy_(torch.from_numpy(_weights_dict[name]['scale']))
        # else:
        #     layer.weight.data.fill_(1)

        # if 'bias' in _weights_dict[name]:
        #     layer.state_dict()['bias'].copy_(torch.from_numpy(_weights_dict[name]['bias']))
        # else:
        #     layer.bias.data.fill_(0)

        # layer.state_dict()['running_mean'].copy_(torch.from_numpy(_weights_dict[name]['mean']))
        # layer.state_dict()['running_var'].copy_(torch.from_numpy(_weights_dict[name]['var']))
        return layer

