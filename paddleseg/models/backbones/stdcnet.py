# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from .stdcnet import *
import math

import paddle
import paddle.nn as nn

from paddleseg.utils import utils
from paddleseg.cvlibs import manager, param_init
from paddleseg.models.layers.layer_libs import SyncBatchNorm
import paddle.nn.functional as F

__all__ = ["STDC1", "STDC2"]




# #----------------------------------------------------------CA注意力机制
# import torch
# import torch.nn as nn
# import math
# import torch.nn.functional as F
#
#
# class h_sigmoid(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_sigmoid, self).__init__()
#         self.relu = nn.ReLU6(inplace=inplace)
#
#     def forward(self, x):
#         return self.relu(x + 3) / 6
#
#
# class h_swish(nn.Module):
#     def __init__(self, inplace=True):
#         super(h_swish, self).__init__()
#         self.sigmoid = h_sigmoid(inplace=inplace)
#
#     def forward(self, x):
#         return x * self.sigmoid(x)
#
# class CoordAtt(nn.Module):
#     def __init__(self, inp, oup, reduction=32):
#         super(CoordAtt, self).__init__()
#         self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
#         self.pool_w = nn.AdaptiveAvgPool2d((1, None))
#
#         mip = max(8, inp // reduction)
#
#         self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(mip)
#         self.act = h_swish()
#
#         self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         identity = x
#
#         n, c, h, w = x.size()
#         x_h = self.pool_h(x)
#         x_w = self.pool_w(x).permute(0, 1, 3, 2)
#
#         y = torch.cat([x_h, x_w], dim=2)
#         y = self.conv1(y)
#         y = self.bn1(y)
#         y = self.act(y)
#
#         x_h, x_w = torch.split(y, [h, w], dim=2)
#         x_w = x_w.permute(0, 1, 3, 2)
#
#         a_h = self.conv_h(x_h).sigmoid()
#         a_w = self.conv_w(x_w).sigmoid()
#
#         out = identity * a_w * a_h
#
#         return out
#
#
# #----------------------------------------------------------
#
# ---------------------------------------------------------
#
#
# class GhostModule(nn.Module):
#     def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
#         super(GhostModule, self).__init__()
#         #ratio一般会指定成2，保证输出特征层的通道数等于exp
#         self.oup = oup
#         init_channels = math.ceil(oup / ratio)
#         new_channels = init_channels*(ratio-1)
#
#         #利用1x1卷积对输入进来的特征图进行通道的浓缩，获得特征通缩
#         #跨通道的特征提取
#         self.primary_conv = nn.Sequential(
#             nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),  #1x1卷积的输入通道数为GhostModule的输出通道数oup/2
#             nn.BatchNorm2d(init_channels),                       #1x1卷积后进行标准化
#             nn.ReLU(inplace=True) if relu else nn.Sequential(),  #ReLU激活函数
#         )
#
#         #在获得特征浓缩后，使用逐层卷积，获得额外的特征图
#         #跨特征点的特征提取    一般会设定大于1的卷积核大小
#         self.cheap_operation = nn.Sequential(
#             nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),  #groups参数的功能就是将普通卷积转换成逐层卷据
#             nn.BatchNorm2d(new_channels),
#             nn.ReLU(inplace=True) if relu else nn.Sequential(),
#         )
#
#     def forward(self, x):
#         x1 = self.primary_conv(x)
#         x2 = self.cheap_operation(x1)
#         #将1x1卷积后的结果和逐层卷积后的结果进行堆叠
#         out = torch.cat([x1,x2], dim=1)
#         return out[:,:self.oup,:,:]
# #-----------------------------------------------------------
# def position(H, W, is_cuda=True):
#     if is_cuda:
#         loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
#         loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
#     else:
#         loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
#         loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
#     loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
#     return loc
#
#
# def stride(x, stride):
#     b, c, h, w = x.shape
#     return x[:, :, ::stride, ::stride]
#
#
# def init_rate_half(tensor):
#     if tensor is not None:
#         tensor.data.fill_(0.5)
#
#
# def init_rate_0(tensor):
#     if tensor is not None:
#         tensor.data.fill_(0.)
#
# #-----------------------------------------------------------
#
# class ConvBNRelu(nn.Module):
#     def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
#         super(ConvBNRelu, self).__init__()
#         self.conv = GhostModule(in_chan, out_chan, kernel_size=ks, stride=stride, dw_size=ks, relu=True)
#         self.bn = nn.BatchNorm2d(out_chan)
#         self.relu = nn.ReLU()
#         self.init_weight()
#
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.bn(x)
#         x = self.relu(x)
#         return x
#
#     def init_weight(self):
#         for ly in self.children():
#             if isinstance(ly, nn.Conv2d):
#                 nn.init.kaiming_normal_(ly.weight, a=1)
#                 if not ly.bias is None: nn.init.constant_(ly.bias, 0)


#--------------------PSA-------------------------


# class SEWeightModule(nn.Layer):
#
#     def __init__(self, channels, reduction=16):
#         super(SEWeightModule, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2D(1)
#         self.fc1 = nn.Conv2D(channels, channels//reduction, kernel_size=1, padding=0)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Conv2D(channels//reduction, channels, kernel_size=1, padding=0)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         out = self.avg_pool(x)
#         out = self.fc1(out)
#         out = self.relu(out)
#         out = self.fc2(out)
#         weight = self.sigmoid(out)
#
#         return weight
#
#
# def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
#     """standard convolution with padding"""
#     return nn.Conv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
#                      padding=padding, dilation=dilation, groups=groups, bias_attr=False)
#
#
# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)
#
#
# class PSAModule(nn.Layer):
#
#     def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
#         super(PSAModule, self).__init__()
#         self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
#                             stride=stride, groups=conv_groups[0])
#         self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
#                             stride=stride, groups=conv_groups[1])
#         self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
#                             stride=stride, groups=conv_groups[2])
#         self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
#                             stride=stride, groups=conv_groups[3])
#         self.se = SEWeightModule(planes // 4)
#         self.split_channel = planes // 4
#         self.softmax = nn.Softmax(axis=1)
#
#     def forward(self, x):
#         batch_size = x.shape[0]
#         x1 = self.conv_1(x)
#         x2 = self.conv_2(x)
#         x3 = self.conv_3(x)
#         x4 = self.conv_4(x)
#
#         x1 = paddle.to_tensor(x1)
#         x2 = paddle.to_tensor(x2)
#         x3 = paddle.to_tensor(x3)
#         x4 = paddle.to_tensor(x4)
#
#         # 将四个输入张量按照第1个维度进行拼接
#         feats = paddle.concat((x1, x2, x3, x4), axis=1)
#
#         # 修改feats的形状为(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])
#         feats = feats.reshape([batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3]])
#
#         x1_se = self.se(x1)
#         x2_se = self.se(x2)
#         x3_se = self.se(x3)
#         x4_se = self.se(x4)
#
#         x1_se = paddle.to_tensor(x1_se)
#         x2_se = paddle.to_tensor(x2_se)
#         x3_se = paddle.to_tensor(x3_se)
#         x4_se = paddle.to_tensor(x4_se)
#
#         # 将四个输入张量按照第1个维度进行拼接
#         x_se = paddle.concat((x1_se, x2_se, x3_se, x4_se), axis=1)
#
#         # 修改x_se的形状为(batch_size, 4, self.split_channel, 1, 1)
#         attention_vectors = x_se.reshape([batch_size, 4, self.split_channel, 1, 1])
#         attention_vectors = self.softmax(attention_vectors)
#         feats_weight = feats * attention_vectors
#         for i in range(4):
#             x_se_weight_fp = feats_weight[:, i, :, :]
#             if i == 0:
#                 out = x_se_weight_fp
#             else:
#                 out = paddle.concat((x_se_weight_fp, out), 1)
#
#         return out
#
# from paddleseg.models.backbones.a import Ghostblockv2
#
#
# class Bottleneck(nn.Layer):
#     # Darknet bottleneck
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
#         super(Bottleneck, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = ConvBNRelu(c1, c_, 1, 1)
#         self.cv2 = ConvBNRelu(c_, c2, 3, 1)
#         self.add = shortcut and c1 == c2
#
#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
#
#
# class C3(nn.Layer):
#     # CSP Bottleneck with 3 convolutions
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = ConvBNRelu(c1, c_, 1, 1)
#         self.cv2 = ConvBNRelu(c1, c_, 1, 1)
#         self.cv3 = ConvBNRelu(2 * c_, c2, 1)  # optional act=FReLU(c2)
#         self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
#
#     def forward(self, x):
#         return self.cv3(paddle.concat((self.m(self.cv1(x)), self.cv2(x)), 1))
#
#
# class C3GhostV2(C3):
#     # C3GV2 module with GhostV2(for iscyy)
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         self.c1_ = 16
#         self.c2_ = 16 * e
#         c_ = int(c2 * e)
#         self.m = nn.Sequential(*(Ghostblockv2(c_, self.c1_, c_) for _ in range(n)))


class STDCNet(nn.Layer):
    """
    The STDCNet implementation based on PaddlePaddle.

    The original article refers to Meituan
    Fan, Mingyuan, et al. "Rethinking BiSeNet For Real-time Semantic Segmentation."
    (https://arxiv.org/abs/2104.13188)

    Args:
        base(int, optional): base channels. Default: 64.
        layers(list, optional): layers numbers list. It determines STDC block numbers of STDCNet's stage3\4\5. Defualt: [4, 5, 3].
        block_num(int,optional): block_num of features block. Default: 4.
        type(str,optional): feature fusion method "cat"/"add". Default: "cat".
        relative_lr(float,optional): parameters here receive a different learning rate when updating. The effective 
            learning rate is the prodcut of relative_lr and the global learning rate. Default: 1.0. 
        in_channels (int, optional): The channels of input image. Default: 3.
        pretrained(str, optional): the path of pretrained model.
    """

    def __init__(self,
                 base=64,
                 layers=[4, 5, 3],
                 block_num=4,
                 type="cat",
                 relative_lr=1.0,
                 in_channels=3,
                 pretrained=None):
        super(STDCNet, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.layers = layers
        self.feat_channels = [base // 2, base, base * 4, base * 8, base * 16]
        self.features = self._make_layers(in_channels, base, layers, block_num,
                                          block, relative_lr)

        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        """
        forward function for feature extract.
        """
        out_feats = []

        x = self.features[0](x)
        out_feats.append(x)
        x = self.features[1](x)
        out_feats.append(x)

        idx = [[2, 2 + self.layers[0]],
               [2 + self.layers[0], 2 + sum(self.layers[0:2])],
               [2 + sum(self.layers[0:2]), 2 + sum(self.layers)]]
        for start_idx, end_idx in idx:
            for i in range(start_idx, end_idx):
                x = self.features[i](x)
            out_feats.append(x)

        return out_feats

    def _make_layers(self, in_channels, base, layers, block_num, block,
                     relative_lr):
        features = []
        features += [ConvBNRelu(in_channels, base // 2, 3, 2, relative_lr)]
        features += [ConvBNRelu(base // 2, base, 3, 2, relative_lr)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(
                        block(base, base * 4, block_num, 2, relative_lr))
                elif j == 0:
                    features.append(
                        block(base * int(math.pow(2, i + 1)), base * int(
                            math.pow(2, i + 2)), block_num, 2, relative_lr))
                else:
                    features.append(
                        block(base * int(math.pow(2, i + 2)), base * int(
                            math.pow(2, i + 2)), block_num, 1, relative_lr))

        return nn.Sequential(*features)

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, nn.Conv2D):
                param_init.normal_init(layer.weight, std=0.001)
            elif isinstance(layer, (nn.BatchNorm, nn.SyncBatchNorm)):
                param_init.constant_init(layer.weight, value=1.0)
                param_init.constant_init(layer.bias, value=0.0)
        if self.pretrained is not None:
            utils.load_pretrained_model(self, self.pretrained)


class ConvBNRelu(nn.Layer):
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel=3,
                 stride=1,
                 relative_lr=1.0):
        super(ConvBNRelu, self).__init__()
        param_attr = paddle.ParamAttr(learning_rate=relative_lr)
        self.conv = nn.Conv2D(
            in_planes,
            out_planes,
            kernel_size=kernel,
            stride=stride,
            padding=kernel // 2,
            weight_attr=param_attr,
            bias_attr=False)
        self.bn = nn.BatchNorm2D(
            out_planes, weight_attr=param_attr, bias_attr=param_attr)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out


class AddBottleneck(nn.Layer):
    def __init__(self,
                 in_planes,
                 out_planes,
                 block_num=3,
                 stride=1,
                 relative_lr=1.0):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride
        param_attr = paddle.ParamAttr(learning_rate=relative_lr)
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    weight_attr=param_attr,
                    bias_attr=False),
                nn.BatchNorm2D(
                    out_planes // 2,
                    weight_attr=param_attr,
                    bias_attr=param_attr), )
            self.skip = nn.Sequential(
                nn.Conv2D(
                    in_planes,
                    in_planes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=in_planes,
                    weight_attr=param_attr,
                    bias_attr=False),
                nn.BatchNorm2D(
                    in_planes, weight_attr=param_attr, bias_attr=param_attr),
                nn.Conv2D(
                    in_planes,
                    out_planes,
                    kernel_size=1,
                    bias_attr=False,
                    weight_attr=param_attr),
                nn.BatchNorm2D(
                    out_planes, weight_attr=param_attr, bias_attr=param_attr), )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(
                        in_planes,
                        out_planes // 2,
                        kernel=1,
                        relative_lr=relative_lr))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2,
                        out_planes // 2,
                        stride=stride,
                        relative_lr=relative_lr))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2,
                        out_planes // 4,
                        stride=stride,
                        relative_lr=relative_lr))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx + 1)),
                        relative_lr=relative_lr))
            else:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx)),
                        relative_lr=relative_lr))

    def forward(self, x):
        out_list = []
        out = x
        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)
        if self.stride == 2:
            x = self.skip(x)
        return paddle.concat(out_list, axis=1) + x


class CatBottleneck(nn.Layer):
    def __init__(self,
                 in_planes,
                 out_planes,
                 block_num=3,
                 stride=1,
                 relative_lr=1.0):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, "block number should be larger than 1."
        self.conv_list = nn.LayerList()
        self.stride = stride

        param_attr = paddle.ParamAttr(learning_rate=relative_lr)
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2D(
                    out_planes // 2,
                    out_planes // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=out_planes // 2,
                    weight_attr=param_attr,
                    bias_attr=False),
                nn.BatchNorm2D(
                    out_planes // 2,
                    weight_attr=param_attr,
                    bias_attr=param_attr), )
            self.skip = nn.AvgPool2D(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(
                    ConvBNRelu(
                        in_planes,
                        out_planes // 2,
                        kernel=1,
                        relative_lr=relative_lr))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2,
                        out_planes // 2,
                        stride=stride,
                        relative_lr=relative_lr))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // 2,
                        out_planes // 4,
                        stride=stride,
                        relative_lr=relative_lr))
            elif idx < block_num - 1:
                self.conv_list.append(
                    ConvBNRelu(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx + 1)),
                        relative_lr=relative_lr))
            else:
                self.conv_list.append(
                    # C3GhostV2(out_planes // int(math.pow(2, idx)), out_planes // int(math.pow(2, idx))))
                    ConvBNRelu(
                        out_planes // int(math.pow(2, idx)),
                        out_planes // int(math.pow(2, idx)),
                        relative_lr=relative_lr))
        # self.reg = RepGhostBottleneck(out_planes, out_planes // 2, out_planes)

    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)
        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)
        out = paddle.concat(out_list, axis=1)
        # out = self.reg(out)
        return out


# from paddleseg.models.backbones.repghost import RepGhostBottleneck
#
#
# class C2f(nn.Layer):
#     def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
#         super(C2f, self).__init__()
#         self.c = int(c2 * e)
#         self.cv1 = nn.Conv2D(c1, 2 * self.c, 1, 1)
#         self.cv2 = nn.Conv2D((2 + n) * self.c, c2, 1)
#         self.m = nn.LayerList([RepGhostBottleneck(self.c, self.c // 2, self.c, shortcut) for _ in range(n)])
#
#     def forward(self, x):
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend([m(y[-1]) for m in self.m])
#         return self.cv2(paddle.concat(y, 1))
#
#     def forward_split(self, x):
#         y = list(self.cv1(x).split((self.c, self.c), 1))
#         y.extend([m(y[-1]) for m in self.m])
#         return self.cv2(paddle.concat(y, 1))

# class PConv3(nn.Layer):
#     def __init__(self, dim=int, n_div=2, forward="split_cat", kernel_size=3):
#         super().__init__()
#         self.dim_conv = dim // n_div
#         self.dim_untouched = dim - self.dim_conv
#         self.conv = nn.Conv2D(
#             self.dim_conv,
#             self.dim_conv,
#             kernel_size,
#             stride=1,
#             padding=(kernel_size -1) // 2,
#             bias_attr=False
#         )
#
#         if forward == "slicing":
#             self.forward = self.forward_slicing
#         elif forward == "split_cat":
#             self.forward = self.forward_split_cat
#         else:
#             raise NotImplementedError
#
#     def forward_slicing(self, x):
#         x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])
#         return x
#
#     def forward_split_cat(self, x):
#         x1, x2 = paddle.split(x, [self.dim_conv, self.dim_untouched], axis=1)
#         x1 = self.conv(x1)
#         x = paddle.concat((x1, x2), 1)
#         return x
#
#
# class FasterNetBlock(nn.Layer):
#     def __init__(self, c1, c2, shortcut=False, e=0.5):
#         super().__init__()
#         c_ = int(c2 * e)
#         self.cv1 = PConv3(c1, 2, "split_cat", 3)
#         self.cv2 = ConvBNRelu(c1, c_, 1, 1)
#         self.cv3 = ConvBNRelu(c_, c2, 1, 1)
#         self.add = shortcut
#
#     def forward(self, x):
#         return  x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))
#
#
# class PConv(nn.Layer):
#     def __init__(self, dim=int, n_div = 2, forward = "split_cat", kernel_size = 3):
#         super().__init__()
#         self.dim_conv = dim // n_div
#         self.dim_untouched = dim - self.dim_conv
#         self.conv = nn.Conv2D(
#             self.dim_conv,
#             self.dim_conv,
#             kernel_size,
#             stride=1,
#             padding=(kernel_size -1) // 2,
#             bias_attr=False
#         )
#
#         if forward == "slicing":
#             self.forward = self.forward_slicing
#         elif forward == "split_cat":
#             self.forward = self.forward_split_cat
#         else:
#             raise NotImplementedError
#
#     def forward_slicing(self, x):
#         x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])
#         return x
#
#     def forward_split_cat(self, x):
#         x1, x2 = paddle.split(x, [self.dim_conv, self.dim_untouched], axis=1)
#         x1 = self.conv(x1)
#         x = paddle.concat((x1, x2), 1)
#         return x


# class GhostModule(nn.Layer):
#     def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
#         super(GhostModule, self).__init__()
#         self.oup = oup
#         init_channels = math.ceil(oup / ratio)
#         new_channels = init_channels * (ratio - 1)
#
#         self.primary_conv = nn.Sequential(
#             nn.Conv2D(inp, init_channels, kernel_size, stride, kernel_size // 2, bias_attr=False),
#             nn.BatchNorm2D(init_channels),
#             nn.ReLU() if relu else nn.Sequential()
#         )
#
#         self.cheap_operation = nn.Sequential(
#             nn.Conv2D(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias_attr=False),
#             nn.BatchNorm2D(new_channels),
#             nn.ReLU() if relu else nn.Sequential()
#         )
#
#     def forward(self, x):
#         x1 = self.primary_conv(x)
#         x2 = self.cheap_operation(x1)
#         out = paddle.concat([x1, x2], axis=1)
#         return out[:, :self.oup, :, :]


# class Multi_Concat_Block(nn.Layer):
#     def __init__(self, c1, c2, c3, n=4, e=1, ids=[0]):
#         super(Multi_Concat_Block, self).__init__()
#         c_ = int(c2 * e)
#         self.ids = ids
#         self.cv1 = ConvBNRelu(c1, c_, 1, 1)
#         self.cv2 = ConvBNRelu(c1, c_, 1, 1)
#         self.cv3 = nn.LayerList(
#             [ConvBNRelu(c_ if i ==0 else c2, c2, 3, 1) for i in range(n)]
#         )
#         self.cv4 = ConvBNRelu(c_ * 2 + c2 * (len(ids) - 2), c3, 1, 1)
#     def forward(self, x):
#         x_1 = self.cv1(x)
#         x_2 = self.cv2(x)
#         x_all = [x_1, x_2]
#         for i in range(len(self.cv3)):
#             x_2 = self.cv3[i](x_2)
#             x_all.append(x_2)
#         out = self.cv4(paddle.concat([x_all[id] for id in self.ids], 1))
#         return out
#
# class Yolov7_E_ELAN_NECK(nn.Layer):
#     def __init__(self, inc, ouc, hidc):
#         super(Yolov7_E_ELAN_NECK, self).__init__()
#
#         self.conv1 = ConvBNRelu(inc, ouc, kernel=1)
#         self.conv2 = ConvBNRelu(inc, ouc, kernel=1)
#         self.conv3 = ConvBNRelu(ouc, hidc, kernel=3)
#         self.conv4 = ConvBNRelu(hidc, hidc, kernel=3)
#         self.conv5 = ConvBNRelu(hidc, hidc, kernel=3)
#         self.conv6 = ConvBNRelu(hidc, hidc, kernel=3)
#         self.conv7 = ConvBNRelu(hidc * 4 + ouc * 2, ouc, kernel=1)
#
#     def forward(self, x):
#         x1, x2 = self.conv1(x), self.conv2(x)
#         x3 = self.conv3(x2)
#         x4 = self.conv4(x3)
#         x5 = self.conv5(x4)
#         x6 = self.conv6(x5)
#         x_concat = paddle.concat([x1, x2, x3, x4, x5, x6], dim=1)
#         x_final = self.conv7(x_concat)
#         return x_final
#
#
# class PConv(nn.Layer):
#     def __init__(self, dim=int, n_div = 2, forward = "split_cat", kernel_size = 3):
#         super().__init__()
#         self.dim_conv = dim // n_div
#         self.dim_untouched = dim - self.dim_conv
#         self.conv = nn.Conv2D(
#             self.dim_conv,
#             self.dim_conv,
#             kernel_size,
#             stride=1,
#             padding=(kernel_size -1) // 2,
#             bias_attr=False
#         )
#
#         if forward == "slicing":
#             self.forward = self.forward_slicing
#         elif forward == "split_cat":
#             self.forward = self.forward_split_cat
#         else:
#             raise NotImplementedError
#
#     def forward_slicing(self, x):
#         x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])
#         return x
#
#     def forward_split_cat(self, x):
#         x1, x2 = paddle.split(x, [self.dim_conv, self.dim_untouched], axis=1)
#         x1 = self.conv(x1)
#         x = paddle.concat((x1, x2), 1)
#         return x


@manager.BACKBONES.add_component
def STDC2(**kwargs):
    model = STDCNet(base=64, layers=[4, 5, 3], **kwargs)
    return model


@manager.BACKBONES.add_component
def STDC1(**kwargs):
    model = STDCNet(base=64, layers=[2, 2, 2], **kwargs)
    return model

