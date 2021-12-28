import tensorflow as tf
import numpy as np

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

from box_ops import xywh_to_xyxy, get_giou

from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

class HungarianMatcher():
    def __init__(self, cost_class=1.0, cost_bbox=1.0, cost_giou=1.0):
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    def match(self, true, pred):
        batch_size = len(true)

        batch_indices = []
        for batch in range(batch_size):
            tgt_ids = true[batch]['labels'].reshape(-1).astype(int)
            tgt_bbox = true[batch]['boxes']

            num_batch_boxes = len(tgt_bbox)

            out_prob = pred['pred_logits'][batch]
            out_bbox = pred['pred_boxes'][batch]

            cost_class = 1 - out_prob[:, tgt_ids]

            cost_bbox = cdist(out_bbox, tgt_bbox)

            cost_giou = 1 - get_giou(xywh_to_xyxy(out_bbox, return_tensor=True), xywh_to_xyxy(tgt_bbox, return_tensor=True))

            C = self.cost_class * cost_class + (self.cost_bbox * cost_bbox + self.cost_giou * cost_giou) / num_batch_boxes

            batch_indices.append(linear_sum_assignment(C))

        return batch_indices
