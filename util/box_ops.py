# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area
import math

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def ciou_loss(preds, gts, eps=1e-7, reduction='mean'):
    '''
    https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/loss/multibox_loss.py
    :param preds:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param gts:[[x1,y1,x2,y2], [x1,y1,x2,y2],,,]
    :param eps: eps to avoid divide 0
    :param reduction: mean or sum
    :return: diou-loss
    '''
    assert (preds[:, 2:] >= preds[:, :2]).all()
    assert (gts[:, 2:] >= gts[:, :2]).all()

    ix1 = torch.max(preds[:, None, 0], gts[:, 0])
    iy1 = torch.max(preds[:, None, 1], gts[:, 1])
    ix2 = torch.min(preds[:, None, 2], gts[:, 2])
    iy2 = torch.min(preds[:, None, 3], gts[:, 3])

    iw = (ix2 - ix1 + 1.0).clamp(min=0.)
    ih = (iy2 - iy1 + 1.0).clamp(min=0.)

    # overlaps
    inters = iw * ih

    # union
    uni = (preds[:, None, 2] - preds[:, None, 0] + 1.0) * (preds[:, None, 3] - preds[:, None, 1] + 1.0) + (gts[:, 2] - gts[:, 0] + 1.0) * (gts[:, 3] - gts[:, 1] + 1.0) - inters

    # iou
    iou = inters / (uni + eps)

    #iou, uni = box_iou(preds, bbox)

    # inter_diag
    cxpreds = (preds[:, None, 2] + preds[:, None, 0]) / 2
    cypreds = (preds[:, None, 3] + preds[:, None, 1]) / 2

    cxbbox = (gts[:, 2] + gts[:, 0]) / 2
    cybbox = (gts[:, 3] + gts[:, 1]) / 2

    inter_diag = (cxbbox - cxpreds) ** 2 + (cybbox - cypreds) ** 2

    # gts
    ox1 = torch.min(preds[:, None, 0], gts[:, 0])
    oy1 = torch.min(preds[:, None, 1], gts[:, 1])
    ox2 = torch.max(preds[:, None, 2], gts[:, 2])
    oy2 = torch.max(preds[:, None, 3], gts[:, 3])

    outer_diag = (ox1 - ox2) ** 2 + (oy1 - oy2) ** 2

    diou = iou - inter_diag / outer_diag

    # calculate v,alpha
    wbbox = gts[:, 2] - gts[:, 0] + 1.0
    hbbox = gts[:, 3] - gts[:, 1] + 1.0
    wpreds = preds[:, None, 2] - preds[:, None, 0] + 1.0
    hpreds = preds[:, None, 3] - preds[:, None, 1] + 1.0
    v = torch.pow((torch.atan(wbbox / hbbox) - torch.atan(wpreds / hpreds)), 2) * (4 / (math.pi ** 2))
    alpha = v / (1 - iou + v)
    ciou = diou - alpha * v
    ciou = torch.clamp(ciou, min=-1.0, max=1.0)

    # diou_loss = 1 - diou
    # ciou_loss = 1 - ciou
    # loss = torch.mean(diou_loss)
    # if reduction == 'mean':
    #     loss1 = torch.mean(ciou_loss)
    # elif reduction == 'sum':
    #     loss1 = torch.sum(ciou_loss)
    # else:
    #     raise NotImplementedError
    
    return ciou

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
