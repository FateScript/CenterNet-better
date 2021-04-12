#!/usr/bin/python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class SingleHead(nn.Module):

    def __init__(self, in_channel, out_channel, bias_fill=False, bias_value=0):
        super(SingleHead, self).__init__()
        self.feat_conv = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.out_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        if bias_fill:
            self.out_conv.bias.data.fill_(bias_value)

    def forward(self, x):
        x = self.feat_conv(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x


class CenternetHead(nn.Module):
    """
    The head used in CenterNet for object classification and box regression.
    It has three subnet, with a common structure but separate parameters.
    """
    def __init__(self, cfg):
        super(CenternetHead, self).__init__()
        self.cls_head = SingleHead(
            64,
            cfg.MODEL.CENTERNET.NUM_CLASSES,
            bias_fill=True,
            bias_value=cfg.MODEL.CENTERNET.BIAS_VALUE,
        )
        self.wh_head = SingleHead(64, 2)
        self.reg_head = SingleHead(64, 2)
        self.segmentation_head_x = SegHead(num_polygon_points=cfg.MODEL.CENTERNET.NUM_POLYGON_POINTS)
        self.segmentation_head_y = SegHead(num_polygon_points=cfg.MODEL.CENTERNET.NUM_POLYGON_POINTS)

    def forward(self, x):
        cls = self.cls_head(x)
        cls = torch.sigmoid(cls)
        wh = self.wh_head(x)
        reg = self.reg_head(x)
        segmentation_x = self.segmentation_head_x(x)
        segmentation_y = self.segmentation_head_y(x)
        pred = {
            'cls': cls,
            'wh': wh,
            'reg': reg,
            'segmentation_x': segmentation_x,
            'segmentation_y': segmentation_y
        }
        return pred

class SegHead(nn.Module):
    def __init__(self, num_convs=2, in_channels=64, conv_out_channels=64, conv_kernel_size=3, num_polygon_points=4):
        super(SegHead, self).__init__()
        self.num_convs = num_convs
        self.in_channels = in_channels
        self.conv_out_channels = conv_out_channels
        self.conv_kernel_size = conv_kernel_size
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = nn.Conv2d(conv_out_channels, num_polygon_points, 1)

        self.convs = nn.ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                nn.Conv2d(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        x = self.relu(x)
        x = self.out_conv(x)
        return x
