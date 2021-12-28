import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow.keras.layers import Dense, Dropout

from box_ops import xywh_to_xyxy

class FDETR(tf.keras.Model):
    '''DETR with FNet Self-Attn'''
    def __init__(self, backbone, transformer, num_classes, N_obj, d_model=256):
        super(FDETR, self).__init__()
        self.N_obj = N_obj
        self.hidden_dim = d_model
        self.num_classes = num_classes

        self.backbone = backbone
        self.transformer = transformer

        self.train_clf_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.train_box_metric = tf.keras.metrics.MeanAbsoluteError()


    def build(self, input_shape):

        self.query_embed = self.add_weight(
            shape=(self.N_obj, self.hidden_dim),
            initializer = 'random_normal',
            trainable=True,
            name='query_embedding'
        )

    def call(self, inputs, training=None):
        features, pos_enc = self.backbone(inputs)

        hs = self.transformer(features, self.query_embed, pos_enc, training=training)

        hs = Dense(self.hidden_dim, activation='tanh')(hs)
        hs = Dropout(0.1)(hs, training=training)
        hs = Dense(self.hidden_dim, activation='tanh')(hs)

        
        pred_boxes = Dense(4, activation='sigmoid')(hs)

        pred_logits = Dense(self.num_classes + 1, activation='softmax')(hs)

        final_output = {'pred_logits': pred_logits, 'pred_boxes': pred_boxes}

        return final_output


class FDETR_Loss(tf.keras.losses.Loss):
    def __init__(self, num_classes, matcher, lambda_h=5.0, lambda_i=2.0):
        super(FDETR_Loss,self).__init__()
        self.mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)
        self.scce_loss = tf.keras.losses.SparseCategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        self.GIoU = tfa.losses.GIoULoss(reduction=tf.keras.losses.Reduction.NONE)
        self.lambda_h = lambda_h
        self.lambda_i = lambda_i
        self.matcher = matcher
        self.num_classes = num_classes #do not include no-object category

    def call(self, y_true, y_pred):
        batch_size = len(y_true)
        #compute hungarian assignment here
        indices = self.matcher.match(y_true, y_pred)

        #match/filter boxes and labels
        src_boxes, tgt_boxes, src_logits, tgt_labels, num_total_boxes = get_src_tgt_pairs(y_true, y_pred, batch_size, indices)

        loss_bbox = self.lambda_h * self.mae(tgt_boxes, src_boxes)

        loss_giou = self.lambda_i * self.GIoU(xywh_to_xyxy(tgt_boxes, return_tensor=True), xywh_to_xyxy(src_boxes, return_tensor=True))
        
        loss_cce = self.scce_loss(tgt_labels, src_logits)
        
        loss = loss_cce + (loss_bbox + loss_giou) / num_total_boxes
 
        return tf.reduce_sum(tf.reduce_mean(loss))

def get_src_tgt_pairs(y_true, y_pred, batch_size, indices):
    pred_logits = y_pred['pred_logits']
    pred_boxes = y_pred['pred_boxes']

    src_boxes, tgt_boxes = [], []
    src_logits, tgt_labels = [], []

    num_total_boxes = 0
    for sample in range(batch_size):
        t_labels = y_true[sample]['labels'] #np array, shape (num_tgt_boxes, 1)
        t_boxes = y_true[sample]['boxes'] #np array, shape (num_tgt_boxes, 4)

        num_total_boxes += len(t_boxes)
        
        p_labels = pred_logits[sample] #tf tensor, shape (N_obj, num_classes)
        p_boxes = pred_boxes[sample] #tf tensor, shape (N_obj, 4)

        sample_idx, sample_jdx = indices[sample]

        for idx, row in enumerate(sample_idx):
            src_boxes.append(p_boxes[row])
            tgt_boxes.append(t_boxes[sample_jdx[idx]])

            src_logits.append(p_labels[row])
            tgt_labels.append(t_labels[sample_jdx[idx]])
    
    return np.array(src_boxes), np.array(tgt_boxes), src_logits, tgt_labels, num_total_boxes

class FDETR_LRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, start_lr=1e-5, 
                 min_lr=1e-5, max_lr=1e-3, 
                 rampup_epochs=40, sustain_epochs=10,
                 exp_decay=0.8):
        super(FDETR_LRSchedule, self).__init__()
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.rampup_epochs = rampup_epochs
        self.sustain_epochs = sustain_epochs
        self.exp_decay = exp_decay

    def __call__(self, epoch):
        if epoch < self.rampup_epochs:
            return (self.max_lr - self.start_lr) / self.rampup_epochs * epoch + self.start_lr
        elif epoch < self.rampup_epochs + self.sustain_epochs:
            return self.max_lr
        else:
            return (self.max_lr - self.min_lr) * self.exp_decay**(epoch-self.rampup_epochs-self.sustain_epochs) + self.min_lr