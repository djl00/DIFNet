# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
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

from paddleseg import utils
from paddleseg.models import layers
from paddleseg.cvlibs import manager
from paddleseg.utils import utils


@manager.MODELS.add_component
class PPLiteSeg(nn.Layer):
    """
    The PP_LiteSeg implementation based on PaddlePaddle.

    The original article refers to "Juncai Peng, Yi Liu, Shiyu Tang, Yuying Hao, Lutao Chu,
    Guowei Chen, Zewu Wu, Zeyu Chen, Zhiliang Yu, Yuning Du, Qingqing Dang,Baohua Lai,
    Qiwen Liu, Xiaoguang Hu, Dianhai Yu, Yanjun Ma. PP-LiteSeg: A Superior Real-Time Semantic
    Segmentation Model. https://arxiv.org/abs/2204.02681".

    Args:
        num_classes (int): The number of target classes.
        backbone(nn.Layer): Backbone network, such as stdc1net and resnet18. The backbone must
            has feat_channels, of which the length is 5.
        backbone_indices (List(int), optional): The values indicate the indices of output of backbone.
            Default: [2, 3, 4].
        arm_type (str, optional): The type of attention refinement module. Default: ARM_Add_SpAttenAdd3.
        cm_bin_sizes (List(int), optional): The bin size of context module. Default: [1,2,4].
        cm_out_ch (int, optional): The output channel of the last context module. Default: 128.
        arm_out_chs (List(int), optional): The out channels of each arm module. Default: [64, 96, 128].
        seg_head_inter_chs (List(int), optional): The intermediate channels of segmentation head.
            Default: [64, 64, 64].
        resize_mode (str, optional): The resize mode for the upsampling operation in decoder.
            Default: bilinear.
        pretrained (str, optional): The path or url of pretrained model. Default: None.

    """

    def __init__(self,
                 num_classes,
                 backbone,
                 backbone_indices=[1, 2, 3, 4],
                 arm_type='CAM',
                 cm_bin_sizes=[1, 2, 4],
                 cm_out_ch=128,
                 arm_out_chs=[64, 96, 128],
                 seg_head_inter_chs=[64, 64, 64],
                 resize_mode='bilinear',
                 pretrained=None
                 ):
        super().__init__()

        # backbone
        assert hasattr(backbone, 'feat_channels'), \
            "The backbone should has feat_channels."
        assert len(backbone.feat_channels) >= len(backbone_indices), \
            f"The length of input backbone_indices ({len(backbone_indices)}) should not be" \
            f"greater than the length of feat_channels ({len(backbone.feat_channels)})."
        assert len(backbone.feat_channels) > max(backbone_indices), \
            f"The max value ({max(backbone_indices)}) of backbone_indices should be " \
            f"less than the length of feat_channels ({len(backbone.feat_channels)})."
        self.backbone = backbone

        assert len(backbone_indices) > 1, "The lenght of backbone_indices " \
            "should be greater than 1"
        self.backbone_indices = backbone_indices  # [..., x16_id, x32_id]
        backbone_out_chs = [backbone.feat_channels[i] for i in backbone_indices]

        # head
        if len(arm_out_chs) == 1:
            arm_out_chs = arm_out_chs * len(backbone_indices)
       #  assert len(arm_out_chs) == len(backbone_indices), "The length of " \
            "arm_out_chs and backbone_indices should be equal"

        self.difseg_head = DIFSegHead(backbone_out_chs, arm_out_chs,
                                        cm_bin_sizes, cm_out_ch, arm_type,
                                        resize_mode)

        if len(seg_head_inter_chs) == 1:
            seg_head_inter_chs = seg_head_inter_chs * len(backbone_indices)
        # assert len(seg_head_inter_chs) == len(backbone_indices), "The length of " \
            "seg_head_inter_chs and backbone_indices should be equal"
        self.seg_heads = nn.LayerList()  # [..., head_16, head32]
        for in_ch, mid_ch in zip(arm_out_chs, seg_head_inter_chs):
            self.seg_heads.append(SegHead(in_ch, mid_ch, num_classes))
        # pretrained
        self.pretrained = pretrained
        self.init_weight()

    def forward(self, x):
        x_hw = paddle.shape(x)[2:]

        feats_backbone = self.backbone(x)  # [x2, x4, x8, x16, x32]
        assert len(feats_backbone) >= len(self.backbone_indices), \
            f"The nums of backbone feats ({len(feats_backbone)}) should be greater or " \
            f"equal than the nums of backbone_indices ({len(self.backbone_indices)})"

        feats_selected = [feats_backbone[i] for i in self.backbone_indices]

        feats_head = self.difseg_head(feats_selected)  # [..., x8, x16, x32]
        if self.training:
            logit_list = []

            for x, seg_head in zip(feats_head, self.seg_heads):
                x = seg_head(x)
                logit_list.append(x)

            logit_list = [
                F.interpolate(
                    x, x_hw, mode='bilinear', align_corners=False)
                for x in logit_list
            ]
        else:
            x = self.seg_heads[0](feats_head[0])
            x = F.interpolate(x, x_hw, mode='bilinear', align_corners=False)
            logit_list = [x]

        return logit_list

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

class DIFSegHead(nn.Layer):
    """
    The head of PPLiteSeg.

    Args:
        backbone_out_chs (List(Tensor)): The channels of output tensors in the backbone.
        arm_out_chs (List(int)): The out channels of each arm module.
        cm_bin_sizes (List(int)): The bin size of context module.
        cm_out_ch (int): The output channel of the last context module.
        arm_type (str): The type of attention refinement module.
        resize_mode (str): The resize mode for the upsampling operation in decoder.
    """

    def __init__(self, backbone_out_chs, arm_out_chs, cm_bin_sizes, cm_out_ch,
                 arm_type, resize_mode):
        super().__init__()
        self.cm = CPM(backbone_out_chs[-1], cm_out_ch, cm_out_ch,
                                  cm_bin_sizes)
        assert hasattr(layers,arm_type), \
            "Not support arm_type ({})".format(arm_type)
        arm_class = eval("layers." + arm_type)
        self.fusion_list = nn.LayerList()  # 加的
        self.arm_list = nn.LayerList()  # [..., arm8, arm16, arm32]
        for i in range(len(backbone_out_chs)-1):
            low_chs = backbone_out_chs[i]
            high_ch = cm_out_ch if i == len(
                backbone_out_chs) - 2 else arm_out_chs[i + 1]
            out_ch = arm_out_chs[i]
            fusion = Fusion(low_chs, backbone_out_chs[i+1], low_chs)
            self.fusion_list.append(fusion)
            arm = arm_class(
                low_chs, high_ch, out_ch, ksize=3, resize_mode=resize_mode)
            self.arm_list.append(arm)


    def forward(self, in_feat_list):
        """
        Args:
            in_feat_list (List(Tensor)): Such as [x2, x4, x8, x16, x32].
                x2, x4 and x8 are optional.
        Returns:
            out_feat_list (List(Tensor)): Such as [x2, x4, x8, x16, x32].
                x2, x4 and x8 are optional.
                The length of in_feat_list and out_feat_list are the same.
        """
        high_feat = self.cm(in_feat_list[-1])
        out_feat_list = []
        for idx, i in enumerate(reversed(range(len(in_feat_list)))):
            if idx >= len(in_feat_list) - 3:
                fusion = self.fusion_list[i]
                low_feat = fusion(in_feat_list[i], in_feat_list[i+1])
                arm = self.arm_list[i]
                high_feat = arm(low_feat, high_feat)
                out_feat_list.insert(0, high_feat)

        return out_feat_list


# shan
class LCA(nn.Layer):
    '''
    多特征融合 iAFF
    '''

    def __init__(self, channels=64, r=4):
        super(LCA, self).__init__()
        inter_channels = int(channels // r)
        # 本地注意力
        self.local_att = nn.Sequential(
            nn.Conv2D(channels, inter_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(inter_channels),
            nn.ReLU(),
            nn.Conv2D(inter_channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2D(channels),
         #   GCT(channels)
        )
        self.pconv = PConv3(channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, residual):
        xa = x + residual
        xl = self.local_att(xa)
        wei = self.sigmoid(xl)
        xi = x * wei + residual * (1 - wei)
        xo = self.pconv(xi)
        return xo
# shan
class Fusion(nn.Layer):
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
        self.conv_y = layers.ConvBNReLU(
            y_ch, x_ch, kernel_size=1, padding=0, bias_attr=False)
        self.conv_out = layers.ConvBNReLU(
            x_ch, out_ch, kernel_size=1, padding=0, bias_attr=False)
        self.resize_mode = resize_mode
        self.lca = LCA(out_ch)

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
        return x

    def prepare_y(self, x, y):
        y = self.conv_y(y)
        y_up = F.interpolate(y, paddle.shape(x)[2:], mode=self.resize_mode)
        return y_up

    def fuse(self, x, y):
        out = self.lca(x, y)
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

class CPM(nn.Layer):
    """
    Simple Context module.

    Args:
        in_channels (int): The number of input channels to pyramid pooling module.
        inter_channels (int): The number of inter channels to pyramid pooling module.
        out_channels (int): The number of output channels after pyramid pooling module.
        bin_izes (tuple, optional): The out size of pooled feature maps. Default: (1, 3).
        align_corners (bool): An argument of F.interpolate. It should be set to False
            when the output size of feature is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
    """

    def __init__(self,
                 in_channels,
                 inter_channels,
                 out_channels,
                 bin_sizes,
                 align_corners=False):
        super().__init__()

        self.stages = nn.LayerList([
            self._make_stage(in_channels, inter_channels, size)
            for size in bin_sizes
        ])
        self.conv0 = layers.ConvBNReLU(in_channels=in_channels, out_channels=inter_channels, kernel_size=1)
        self.conv_out = PConv3(out_channels)

        self.align_corners = align_corners

    def _make_stage(self, in_channels, out_channels, size):
        conv0 = layers.ConvBNReLU(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        conv = layers.ConvBNReLU(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=size + 1, dilation=size + 1)

        return nn.Sequential(conv0, conv)

    def forward(self, input):
        out = None

        for stage in self.stages:
            x = stage(input)
            if out is None:
                out = x
            else:
                out += x
        out = out + self.conv0(input)
        out = self.conv_out(out)
        return out


class SegHead(nn.Layer):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = layers.ConvBNReLU(
            in_chan,
            mid_chan,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.conv_out = nn.Conv2D(
            mid_chan, n_classes, kernel_size=1, bias_attr=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

#
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
