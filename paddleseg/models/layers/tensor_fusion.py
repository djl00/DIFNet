# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant
from paddleseg.models import layers


def avg_max_reduce_channel(x):
    # Reduce hw by avg and max
    # Return cat([avg_ch_0, max_ch_0, avg_ch_1, max_ch_1, ...])
    if not isinstance(x, (list, tuple)):
        return avg_max_reduce_channel_helper(x)
    elif len(x) == 1:
        return avg_max_reduce_channel_helper(x[0])
    else:
        res = []
        for xi in x:
            res.extend(avg_max_reduce_channel_helper(xi, False))
        return paddle.concat(res, axis=1)


# shan
class LCGS(nn.Layer):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(LCGS, self).__init__()
        inter_channels = int(channels // r)
        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2D(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(inter_channels),
            nn.ReLU(),
            nn.Conv2D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(channels),
        )
        self._scale = self.create_parameter(
            shape=[1],
            attr=ParamAttr(initializer=Constant(value=1.)),
            dtype="float32")
        self._scale.stop_gradient = True
        self.sp = nn.Sequential(
            layers.ConvBNReLU(
                4, 2, kernel_size=3, padding=1, bias_attr=False),
            layers.ConvBN(
                2, 1, kernel_size=3, padding=1, bias_attr=False))
        self.pconv = PConv3(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        wei = self.sigmoid(xl)
        xi = x * wei + residual * (1 - wei)

        atten = avg_max_reduce_channel([x, residual])
        atten = F.sigmoid(self.sp(atten))
        out = x * atten + residual * (self._scale - atten)

        xo = xi + out
        xo = self.pconv(xo)
        return xo

# shan
class PConv3(nn.Layer):
    def __init__(self, dim=int, n_div=2, forward="split_cat", kernel_size=3):
        super().__init__()
        self.dim_conv = dim // n_div
        self.dim_untouched = dim - self.dim_conv
        self.conv = layers.ConvBNReLU(
            self.dim_conv,
            self.dim_conv,
            kernel_size,
            stride=1,
            padding=(kernel_size -1) // 2,
            bias_attr=False
        )

        if forward == "slicing":
            self.forward = self.forward_slicing
        elif forward == "split_cat":
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        x[:, :self.dim_conv, :, :] = self.conv(x[:, :self.dim_conv, :, :])
        return x

    def forward_split_cat(self, x):
        x1, x2 = paddle.split(x, [self.dim_conv, self.dim_untouched], axis=1)
        x1 = self.conv(x1)
        x = paddle.concat((x1, x2), 1)
        return x


# shan
class CAM(nn.Layer):
    """
    The base of Unified Attention Fusion Module.
    Args:
        x_ch (int): The channel of x tensor, which is the low level feature.
        y_ch (int): The channel of y tensor, which is the high level feature.
        out_ch (int): The channel of output tensor.
        ksize (int, optional): The kernel size of the conv for x tensor. Default: 3.
        resize_mode (str, optional): The resize model in unsampling y tensor. Default: bilinear.
    """

    def __init__(self, x_ch, y_ch, out_ch, resize_mode='bilinear'):
        super().__init__()
        self.conv_x = layers.ConvBNReLU(
            x_ch, y_ch, kernel_size=1, padding=0, bias_attr=False)
        self.conv_out = layers.ConvBNReLU(
            y_ch, out_ch, kernel_size=1, padding=0, bias_attr=False)
        self.resize_mode = resize_mode
        self.lcgs = LCGS(y_ch)

    def check(self, x, y):
        assert x.ndim == 4 and y.ndim == 4
        x_h, x_w = x.shape[2:]
        y_h, y_w = y.shape[2:]
        assert x_h >= y_h and x_w >= y_w

    def prepare(self, x, y):
        x = self.prepare_x(x, y)
        y = self.prepare_y(x, y)
        return x, y

    def prepare_x(self, x, y):
        x = self.conv_x(x)
        return x

    def prepare_y(self, x, y):
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        return y_up

    def fuse(self, x, y):
        out = self.lcgs(x, y)
        out = self.conv_out(out)
        return out

    def forward(self, x, y):
        """
        Args:
            x (Tensor): The low level feature.
            y (Tensor): The high level feature.
        """
        self.check(x, y)
        x, y = self.prepare(x, y)
        out = self.fuse(x, y)
        return out
