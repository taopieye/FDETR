import tensorflow as tf
import numpy as np

def xywh_to_xyxy(x, return_tensor=False):
    if tf.is_tensor(x):
       pass
    else:
       x = tf.convert_to_tensor(x, dtype=tf.float32)
    cx, cy, w, h = tf.unstack(x, axis=-1)

    box = [(cx - 0.5 * w), (cy - 0.5 * h),
           (cx + 0.5 * w), (cy + 0.5 * h)]
    out = tf.stack(box, axis=-1)

    if return_tensor:
       return out
    return np.array(out)
    
def get_iou(boxes1, boxes2, return_tensor=False):
    if tf.is_tensor(boxes1) and tf.is_tensor(boxes2):
        pass
    else:
        boxes1 = tf.convert_to_tensor(boxes1, dtype=tf.float32)
        boxes2 = tf.convert_to_tensor(boxes2, dtype=tf.float32)

    lu = tf.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rd = tf.minimum(boxes1[:, None, 2:], boxes2[:, :2])

    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    
    boxes1_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    boxes2_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )

    iou = tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

    if return_tensor:
        return iou, union_area

    return np.array(iou), np.array(union_area)

def get_giou(boxes1, boxes2, return_tensor=False):
    # check for degenerate boxes (to avoid inf/NaN)
    assert tf.reduce_all(boxes1[:, 2:] >= boxes1[:, :2])
    assert tf.reduce_all((boxes2[:, 2:] >= boxes2[:, :2]))

    iou, union = get_iou(boxes1, boxes2, return_tensor=True)

    lu = tf.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rd = tf.minimum(boxes1[:, None, 2:], boxes2[:, :2])

    max_val = tf.math.reduce_max(rd - lu)

    enclose = tf.clip_by_value(rd - lu, 0.0, max_val)
    area = enclose[:, :, 0] * enclose[:, :, 1]

    giou = iou - (area - union) / area

    if return_tensor:
        return tf.clip_by_value(giou, -1.0, 1.0)

    return tf.clip_by_value(giou, -1.0, 1.0).numpy()