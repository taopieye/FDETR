import tensorflow as tf
import pandas as pd
import numpy as np
import math

def class_name2int(class_name):
    mapper = {'missing_hole': 1,
              'mouse_bite': 2,
              'open_circuit': 3,
              'short': 4,
              'spur': 5,
              'spurious_copper': 6 
    }
    return mapper[class_name]

class FDETRDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df, N_obj=10, batch_size=16, input_shape=(600, 600, 3)):

        self.df = df.copy()
        self.batch_size = batch_size
        self.N_obj = N_obj
        self.num_classes = df['class'].nunique() + 1
        self.files = df['filename'].unique()
        np.random.shuffle(self.files)
        self.input_shape = input_shape
        self.num_row, self.num_col, self.channels = input_shape

    def call(self):
        return self

    def __getitem__(self,index):
        files = self.files[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__get_data(files)

    def __len__(self):
        return math.ceil(len(self.files) / self.batch_size)

    def __get_data(self, files):
        X = np.zeros((self.batch_size, self.num_row, self.num_col, self.channels))
        y = []
        for idx, fid in enumerate(files):
            X[idx] = self.__get_input(fid)
            y.append(self.__get_output(fid))
        return X, y

    def __get_input(self, path):
        image = tf.keras.preprocessing.image.load_img(path)
        image_array = tf.keras.preprocessing.image.img_to_array(image)

        return image_array / 255.
    
    def __get_output(self, fid):
        
        fid_df = self.df[self.df['filename'] == fid]
        num_boxes = len(fid_df)
        tgt_labels = np.zeros((num_boxes, 1))
        tgt_boxes = np.zeros((num_boxes, 4))

        for idx in range(num_boxes):
            sample = fid_df.iloc[idx]
            
            #get bounding boxes in cx, cy, w, h normalized wrt image shape
            cx = 0.5 * (sample['xmin'] + sample['xmax']) / self.num_col
            cy = 0.5 * (sample['ymin'] + sample['ymax']) / self.num_row
            w = (sample['xmax'] - sample['xmin']) / self.num_col
            h = (sample['ymax'] - sample['ymin']) / self.num_row

            box = np.array([cx, cy, w, h], dtype=np.float32)

            label = class_name2int(sample['class'])

            tgt_labels[idx] = label
            tgt_boxes[idx] = box
        
        return {'labels': tgt_labels, 'boxes': tgt_boxes}
        
    

    

    