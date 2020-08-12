#!/usr/bin/python3
# -*- coding:utf-8 -*-
# author: wangfeng19950315@163.com

import numpy as np
import torch


class CenterNetGT(object):

    @staticmethod
    def generate(config, batched_input):
        box_scale = 1 / config.MODEL.CENTERNET.DOWN_SCALE
        num_classes = config.MODEL.CENTERNET.NUM_CLASSES
        output_size = config.INPUT.OUTPUT_SIZE
        min_overlap = config.MODEL.CENTERNET.MIN_OVERLAP
        tensor_dim = config.MODEL.CENTERNET.TENSOR_DIM

        scoremap_list, wh_list, reg_list, reg_mask_list, index_list = [[] for i in range(5)]
        for data in batched_input:
            # img_size = (data['height'], data['width'])

            bbox_dict = data['instances'].get_fields()

            # init gt tensors
            gt_scoremap = torch.zeros(num_classes, *output_size)
            gt_wh = torch.zeros(tensor_dim, 2)
            gt_reg = torch.zeros_like(gt_wh)
            reg_mask = torch.zeros(tensor_dim)
            gt_index = torch.zeros(tensor_dim)
            # pass

            boxes, classes = bbox_dict['gt_boxes'], bbox_dict['gt_classes']
            num_boxes = boxes.tensor.shape[0]
            boxes.scale(box_scale, box_scale)

            centers = boxes.get_centers()
            centers_int = centers.to(torch.int32)
            gt_index[:num_boxes] = centers_int[..., 1] * output_size[1] + centers_int[..., 0]
            gt_reg[:num_boxes] = centers - centers_int
            reg_mask[:num_boxes] = 1

            wh = torch.zeros_like(centers)
            box_tensor = boxes.tensor
            wh[..., 0] = box_tensor[..., 2] - box_tensor[..., 0]
            wh[..., 1] = box_tensor[..., 3] - box_tensor[..., 1]
            CenterNetGT.generate_score_map(
                gt_scoremap, classes, wh,
                centers_int, min_overlap,
            )
            gt_wh[:num_boxes] = wh

            scoremap_list.append(gt_scoremap)
            wh_list.append(gt_wh)
            reg_list.append(gt_reg)
            reg_mask_list.append(reg_mask)
            index_list.append(gt_index)

        gt_dict = {
            "score_map": torch.stack(scoremap_list, dim=0),
            "wh": torch.stack(wh_list, dim=0),
            "reg": torch.stack(reg_list, dim=0),
            "reg_mask": torch.stack(reg_mask_list, dim=0),
            "index": torch.stack(index_list, dim=0),
        }
        return gt_dict

    @staticmethod
    def generate_score_map(fmap, gt_class, gt_wh, centers_int, min_overlap):
        radius = CenterNetGT.get_gaussian_radius(gt_wh, min_overlap)
        radius = torch.clamp_min(radius, 0)
        radius = radius.type(torch.int).cpu().numpy()
        for i in range(gt_class.shape[0]):
            channel_index = gt_class[i]
            CenterNetGT.draw_gaussian(fmap[channel_index], centers_int[i], radius[i])

    @staticmethod
    def get_gaussian_radius(box_size, min_overlap):
        """
        copyed from CornerNet
        box_size (w, h), it could be a torch.Tensor, numpy.ndarray, list or tuple
        notice: we are using a bug-version, please refer to fix bug version in CornerNet
        """
        box_tensor = torch.Tensor(box_size)
        width, height = box_tensor[..., 0], box_tensor[..., 1]

        a1  = 1
        b1  = (height + width)
        c1  = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = torch.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1  = (b1 + sq1) / 2

        a2  = 4
        b2  = 2 * (height + width)
        c2  = (1 - min_overlap) * width * height
        sq2 = torch.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2  = (b2 + sq2) / 2

        a3  = 4 * min_overlap
        b3  = -2 * min_overlap * (height + width)
        c3  = (min_overlap - 1) * width * height
        sq3 = torch.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3  = (b3 + sq3) / 2

        return torch.min(r1, torch.min(r2, r3))

    @staticmethod
    def gaussian2D(radius, sigma=1):
        # m, n = [(s - 1.) / 2. for s in shape]
        m, n = radius
        y, x = np.ogrid[-m:m + 1, -n:n + 1]

        gauss = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        gauss[gauss < np.finfo(gauss.dtype).eps * gauss.max()] = 0
        return gauss

    @staticmethod
    def draw_gaussian(fmap, center, radius, k=1):
        diameter = 2 * radius + 1
        gaussian = CenterNetGT.gaussian2D((radius, radius), sigma=diameter / 6)
        gaussian = torch.Tensor(gaussian)
        x, y = int(center[0]), int(center[1])
        height, width = fmap.shape[:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_fmap  = fmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
        if min(masked_gaussian.shape) > 0 and min(masked_fmap.shape) > 0:
            masked_fmap = torch.max(masked_fmap, masked_gaussian * k)
            fmap[y - top:y + bottom, x - left:x + right] = masked_fmap
        # return fmap
