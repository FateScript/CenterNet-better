# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import itertools
from typing import Any, Iterator, List, Union

import numpy as np
import pycocotools.mask as mask_utils
import torch

from dl_lib.layers import ROIAlign, cat

from .boxes import Boxes


def polygon_area(x, y):
    # Using the shoelace formula
    # https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def polygons_to_bitmask(polygons: List[np.ndarray], height: int, width: int) -> np.ndarray:
    """
    Args:
        polygons (list[ndarray]): each array has shape (Nx2,)
        height, width (int)

    Returns:
        ndarray: a bool mask of shape (height, width)
    """
    assert len(polygons) > 0, "COCOAPI does not support empty polygons"
    rles = mask_utils.frPyObjects(polygons, height, width)
    rle = mask_utils.merge(rles)
    return mask_utils.decode(rle).astype(np.bool)


def rasterize_polygons_within_box(
    polygons: List[np.ndarray], box: np.ndarray, mask_size: int
) -> torch.Tensor:
    """
    Rasterize the polygons into a mask image and
    crop the mask content in the given box.
    The cropped mask is resized to (mask_size, mask_size).

    This function is used when generating training targets for mask head in Mask R-CNN.
    Given original ground-truth masks for an image, new ground-truth mask
    training targets in the size of `mask_size x mask_size`
    must be provided for each predicted box. This function will be called to
    produce such targets.

    Args:
        polygons (list[ndarray[float]]): a list of polygons, which represents an instance.
        box: 4-element numpy array
        mask_size (int):

    Returns:
        Tensor: BoolTensor of shape (mask_size, mask_size)
    """
    # 1. Shift the polygons w.r.t the boxes
    w, h = box[2] - box[0], box[3] - box[1]

    polygons = copy.deepcopy(polygons)
    for p in polygons:
        p[0::2] = p[0::2] - box[0]
        p[1::2] = p[1::2] - box[1]

    # 2. Rescale the polygons to the new box size
    ratio_h = mask_size / max(h, 0.1)
    ratio_w = mask_size / max(w, 0.1)

    if ratio_h == ratio_w:
        for p in polygons:
            p *= ratio_h
    else:
        for p in polygons:
            p[0::2] *= ratio_w
            p[1::2] *= ratio_h

    # 3. Rasterize the polygons with coco api
    mask = polygons_to_bitmask(polygons, mask_size, mask_size)
    mask = torch.from_numpy(mask)
    return mask


class BitMasks:
    """
    This class stores the segmentation masks for all objects in one image, in
    the form of bitmaps.

    Attributes:
        tensor: bool Tensor of N,H,W, representing N instances in the image.
    """

    def __init__(self, tensor: Union[torch.Tensor, np.ndarray]):
        """
        Args:
            tensor: bool Tensor of N,H,W, representing N instances in the image.
        """
        device = tensor.device if isinstance(tensor, torch.Tensor) else torch.device("cpu")
        tensor = torch.as_tensor(tensor, dtype=torch.bool, device=device)
        assert tensor.dim() == 3, tensor.size()
        self.image_size = tensor.shape[1:]
        self.tensor = tensor

    def to(self, device: str) -> "BitMasks":
        return BitMasks(self.tensor.to(device))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "BitMasks":
        """
        Returns:
            BitMasks: Create a new :class:`BitMasks` by indexing.

        The following usage are allowed:

        1. `new_masks = masks[3]`: return a `BitMasks` which contains only one mask.
        2. `new_masks = masks[2:10]`: return a slice of masks.
        3. `new_masks = masks[vector]`, where vector is a torch.BoolTensor
           with `length = len(masks)`. Nonzero elements in the vector will be selected.

        Note that the returned object might share storage with this object,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return BitMasks(self.tensor[item].view(1, -1))
        m = self.tensor[item]
        assert m.dim() == 3, "Indexing on BitMasks with {} returns a tensor with shape {}!".format(
            item, m.shape
        )
        return BitMasks(m)

    def __iter__(self) -> torch.Tensor:
        yield from self.tensor

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def nonempty(self) -> torch.Tensor:
        """
        Find masks that are non-empty.

        Returns:
            Tensor: a BoolTensor which represents
                whether each mask is empty (False) or non-empty (True).
        """
        return self.tensor.flatten(1).any(dim=1)

    @staticmethod
    def from_polygon_masks(
        polygon_masks: Union["PolygonMasks", List[List[np.ndarray]]], height: int, width: int
    ) -> "BitMasks":
        """
        Args:
            polygon_masks (list[list[ndarray]] or PolygonMasks)
            height, width (int)
        """
        if isinstance(polygon_masks, PolygonMasks):
            polygon_masks = polygon_masks.polygons
        masks = [polygons_to_bitmask(p, height, width) for p in polygon_masks]
        return BitMasks(torch.stack([torch.from_numpy(x) for x in masks]))

    def crop_and_resize(self, boxes: torch.Tensor, mask_size: int) -> torch.Tensor:
        """
        Crop each bitmask by the given box, and resize results to (mask_size, mask_size).
        This can be used to prepare training targets for Mask R-CNN.
        It has less reconstruction error compared to rasterization with polygons.
        However we observe no difference in accuracy,
        but BitMasks requires more memory to store all the masks.

        Args:
            boxes (Tensor): Nx4 tensor storing the boxes for each mask
            mask_size (int): the size of the rasterized mask.

        Returns:
            Tensor:
                A bool tensor of shape (N, mask_size, mask_size), where
                N is the number of predicted boxes for this image.
        """
        assert len(boxes) == len(self), "{} != {}".format(len(boxes), len(self))
        device = self.tensor.device

        batch_inds = torch.arange(len(boxes), device=device).to(dtype=boxes.dtype)[:, None]
        rois = torch.cat([batch_inds, boxes], dim=1)  # Nx5

        bit_masks = self.tensor.to(dtype=torch.float32)
        rois = rois.to(device=device)
        output = (
            ROIAlign((mask_size, mask_size), 1.0, 0, aligned=True)
            .forward(bit_masks[:, None, :, :], rois)
            .squeeze(1)
        )
        output = output >= 0.5
        return output

    def get_bounding_boxes(self) -> None:
        # not needed now
        raise NotImplementedError

    @staticmethod
    def cat(bitmasks_list: List["BitMasks"]) -> "BitMasks":
        """
        Concatenates a list of BitMasks into a single BitMasks
        Arguments:
            bitmasks_list (list[BitMasks])
        Returns:
            BitMasks: the concatenated BitMasks
        """
        assert isinstance(bitmasks_list, (list, tuple))
        assert len(bitmasks_list) > 0
        assert all(isinstance(bitmask, BitMasks) for bitmask in bitmasks_list)

        cat_bitmasks = type(bitmasks_list[0])(cat([bm.tensor for bm in bitmasks_list], dim=0))
        return cat_bitmasks


class PolygonMasks:
    """
    This class stores the segmentation masks for all objects in one image, in the form of polygons.

    Attributes:
        polygons: list[list[ndarray]]. Each ndarray is a float64 vector representing a polygon.
    """

    def __init__(self, polygons: List[List[Union[torch.Tensor, np.ndarray]]]):
        """
        Arguments:
            polygons (list[list[Tensor[float]]]): The first
                level of the list correspond to individual instances,
                the second level to all the polygons that compose the
                instance, and the third level to the polygon coordinates.
                The third level Tensor should have the format of
                torch.Tensor([x0, y0, x1, y1, ..., xn, yn]) (n >= 3).
        """
        assert isinstance(polygons, list)

        def _make_array(t: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
            # Use float64 for higher precision, because why not?
            # Always put polygons on CPU (self.to is a no-op) since they
            # are supposed to be small tensors.
            # May need to change this assumption if GPU placement becomes useful
            if isinstance(t, torch.Tensor):
                t = t.cpu().numpy()
            return np.asarray(t).astype("float64")

        def process_polygons(
            polygons_per_instance: List[Union[torch.Tensor, np.ndarray]]
        ) -> List[torch.Tensor]:
            assert isinstance(polygons_per_instance, list), type(polygons_per_instance)
            # transform the polygon to a tensor
            polygons_per_instance = [_make_array(p) for p in polygons_per_instance]
            for polygon in polygons_per_instance:
                assert len(polygon) % 2 == 0 and len(polygon) >= 6
            return polygons_per_instance

        self.polygons: List[List[torch.Tensor]] = [
            process_polygons(polygons_per_instance) for polygons_per_instance in polygons
        ]

    def to(self, *args: Any, **kwargs: Any) -> "PolygonMasks":
        return self

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    def get_bounding_boxes(self) -> Boxes:
        """
        Returns:
            Boxes: tight bounding boxes around polygon masks.
        """
        boxes = torch.zeros(len(self.polygons), 4, dtype=torch.float32)
        for idx, polygons_per_instance in enumerate(self.polygons):
            minxy = torch.as_tensor([float("inf"), float("inf")], dtype=torch.float32)
            maxxy = torch.zeros(2, dtype=torch.float32)
            for polygon in polygons_per_instance:
                coords = torch.from_numpy(polygon).view(-1, 2).to(dtype=torch.float32)
                minxy = torch.min(minxy, torch.min(coords, dim=0).values)
                maxxy = torch.max(maxxy, torch.max(coords, dim=0).values)
            boxes[idx, :2] = minxy
            boxes[idx, 2:] = maxxy
        return Boxes(boxes)

    def nonempty(self) -> torch.Tensor:
        """
        Find masks that are non-empty.

        Returns:
            Tensor:
                a BoolTensor which represents whether each mask is empty (False) or not (True).
        """
        keep = [1 if len(polygon) > 0 else 0 for polygon in self.polygons]
        return torch.as_tensor(keep, dtype=torch.bool)

    def __getitem__(self, item: Union[int, slice, List[int], torch.BoolTensor]) -> "PolygonMasks":
        """
        Support indexing over the instances and return a `PolygonMasks` object.
        `item` can be:

        1. An integer. It will return an object with only one instance.
        2. A slice. It will return an object with the selected instances.
        3. A list[int]. It will return an object with the selected instances,
           correpsonding to the indices in the list.
        4. A vector mask of type BoolTensor, whose length is num_instances.
           It will return an object with the instances whose mask is nonzero.
        """
        if isinstance(item, int):
            selected_polygons = [self.polygons[item]]
        elif isinstance(item, slice):
            selected_polygons = self.polygons[item]
        elif isinstance(item, list):
            selected_polygons = [self.polygons[i] for i in item]
        elif isinstance(item, torch.Tensor):
            # Polygons is a list, so we have to move the indices back to CPU.
            if item.dtype == torch.bool:
                assert item.dim() == 1, item.shape
                item = item.nonzero().squeeze(1).cpu().numpy().tolist()
            elif item.dtype in [torch.int32, torch.int64]:
                item = item.cpu().numpy().tolist()
            else:
                raise ValueError("Unsupported tensor dtype={} for indexing!".format(item.dtype))
            selected_polygons = [self.polygons[i] for i in item]
        return PolygonMasks(selected_polygons)

    def __iter__(self) -> Iterator[List[torch.Tensor]]:
        """
        Yields:
            list[ndarray]: the polygons for one instance.
            Each Tensor is a float64 vector representing a polygon.
        """
        return iter(self.polygons)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.polygons))
        return s

    def __len__(self) -> int:
        return len(self.polygons)

    def crop_and_resize(self, boxes: torch.Tensor, mask_size: int) -> torch.Tensor:
        """
        Crop each mask by the given box, and resize results to (mask_size, mask_size).
        This can be used to prepare training targets for Mask R-CNN.

        Args:
            boxes (Tensor): Nx4 tensor storing the boxes for each mask
            mask_size (int): the size of the rasterized mask.

        Returns:
            Tensor: A bool tensor of shape (N, mask_size, mask_size), where
            N is the number of predicted boxes for this image.
        """
        assert len(boxes) == len(self), "{} != {}".format(len(boxes), len(self))

        device = boxes.device
        # Put boxes on the CPU, as the polygon representation is not efficient GPU-wise
        # (several small tensors for representing a single instance mask)
        boxes = boxes.to(torch.device("cpu"))

        results = [
            rasterize_polygons_within_box(poly, box.numpy(), mask_size)
            for poly, box in zip(self.polygons, boxes)
        ]
        """
        poly: list[list[float]], the polygons for one instance
        box: a tensor of shape (4,)
        """
        if len(results) == 0:
            return torch.empty(0, mask_size, mask_size, dtype=torch.bool, device=device)
        return torch.stack(results, dim=0).to(device=device)

        def area(self):
            """
            Computes area of the mask.
            Only works with Polygons, using the shoelace formula:
            https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
            Returns:
                Tensor: a vector, area for each instance
            """

            area = []
            for polygons_per_instance in self.polygons:
                area_per_instance = 0
                for p in polygons_per_instance:
                    area_per_instance += polygon_area(p[0::2], p[1::2])
                area.append(area_per_instance)

            return torch.tensor(area)

    def normalized_by_length(self, bboxes, num_of_target, bbox_scale):
        if bboxes.shape[0] == 0:
            return torch.ones(0, num_of_target) * -128, torch.ones(0, num_of_target) * -128

        polygons_x = []
        polygons_y = []
        for idx, polygons_per_instance in enumerate(self.polygons):
            # sum_of_points = sum([int(line.shape[0]/2) for line in polygons_per_instance])
            # lines_per_polygons_x = []
            # lines_per_polygons_y = []
            # bbox = bboxes[idx]
            # for line in polygons_per_instance:
            #     num_of_points = int(line.shape[0]/2)
            #     line *= bbox_scale
            #     line = np.roll(line, -np.argwhere(line==line[np.argwhere(line[1::2]==line[1::2].min())*2].min()).min())
            #     normalized_line_x = (line[0::2] - np.tile(bbox[0], num_of_points))
            #     normalized_line_y = (line[1::2] - np.tile(bbox[1], num_of_points))
            #     lines_per_polygons_x.append(normalized_line_x)
            #     lines_per_polygons_y.append(normalized_line_y)
            # if sum_of_points > num_of_target:
            #     sample_idx = np.linspace(0, sum_of_points, num=num_of_target, endpoint=False, dtype=np.int32)
            #     sum_of_points = int(num_of_target)
            # else:
            #     sample_idx = np.arange(0, sum_of_points)
            # target_numpy_x = np.ones(num_of_target) * -128
            # target_numpy_x[:sum_of_points] = np.concatenate(lines_per_polygons_x)[sample_idx]
            # target_numpy_y = np.ones(num_of_target) * -128
            # target_numpy_y[:sum_of_points] = np.concatenate(lines_per_polygons_y)[sample_idx]
            line = np.concatenate(polygons_per_instance) * bbox_scale

            min_in_x = np.squeeze(np.argwhere(line[0::2] == line[0::2].min()))
            left_in_x = line[0::2].min()
            left_in_y = line[line == line[min_in_x * 2 + 1].max()].min()

            min_in_y = np.squeeze(np.argwhere(line[1::2] == line[1::2].min()))
            up_in_x = line[line == line[min_in_y * 2].min()].min()
            up_in_y = line[1::2].min()

            max_in_x = np.squeeze(np.argwhere(line[0::2] == line[0::2].max()))
            right_in_x = line[0::2].max()
            right_in_y = line[line == line[max_in_x * 2 + 1].min()].min()

            max_in_y = np.squeeze(np.argwhere(line[1::2] == line[1::2].max()))
            down_in_x = line[line == line[max_in_y * 2].max()].min()
            down_in_y = line[1::2].max()

            target_numpy_x = np.array([left_in_x, up_in_x, right_in_x, down_in_x])
            target_numpy_y = np.array([left_in_y, up_in_y, right_in_y, down_in_y])

            polygons_x.append(target_numpy_x)
            polygons_y.append(target_numpy_y)

        return torch.tensor(polygons_x), torch.tensor(polygons_y)


    @staticmethod
    def cat(polymasks_list: List["PolygonMasks"]) -> "PolygonMasks":
        """
        Concatenates a list of PolygonMasks into a single PolygonMasks
        Arguments:
            polymasks_list (list[PolygonMasks])
        Returns:
            PolygonMasks: the concatenated PolygonMasks
        """
        assert isinstance(polymasks_list, (list, tuple))
        assert len(polymasks_list) > 0
        assert all(isinstance(polymask, PolygonMasks) for polymask in polymasks_list)

        cat_polymasks = type(polymasks_list[0])(
            list(itertools.chain.from_iterable(pm.polygons for pm in polymasks_list))
        )
        return cat_polymasks
