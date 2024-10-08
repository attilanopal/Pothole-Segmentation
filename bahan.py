import keras
import os
import tensorflow as tf
from keras.losses import binary_crossentropy
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.callbacks import ReduceLROnPlateau, LearningRateScheduler
import tensorflow.keras.layers as l
from tensorflow.keras.metrics import TruePositives, FalseNegatives, FalsePositives
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.layer_utils import count_params

from skimage import segmentation, color
from skimage import graph
from matplotlib import pyplot as plt

from tqdm import tqdm

from zipfile import ZipFile

class ActivationLayer(l.Layer):
    def __init__(self, activation):
        super(ActivationLayer, self).__init__()
        
        if activation == 'leakyrelu':
            self.layer = l.LeakyReLU(alpha=0.01)
        elif activation == 'prelu':
            self.layer = l.PReLU()
        else:
            self.layer = l.Activation(activation)

    def call(self, t):
        return self.layer(t)
    
class ConvBlock(tf.keras.Model):
    def __init__(self, filters, activation, initializer, bias):
        super(ConvBlock, self).__init__()
        
        self.conv1 = l.Conv2D(filters=filters,
                              kernel_size=3,
                              padding='same',
                              kernel_initializer=initializer,
                              use_bias=bias)
        self.bn1 = l.BatchNormalization()
        self.act1 = ActivationLayer(activation=activation)
        
        self.conv2 = l.Conv2D(filters=filters,
                              kernel_size=3,
                              padding='same',
                              kernel_initializer=initializer,
                              use_bias=bias)
        self.act2 = ActivationLayer(activation=activation)
        self.bn2 = l.BatchNormalization()

    def call(self, t):
        t = self.conv1(t)
        t = self.bn1(t)
        t = self.act1(t)
        t = self.conv2(t)
        t = self.bn2(t)
        t = self.act2(t)
        
        return t
    
class UNet(tf.keras.Model):
    def __init__(self, filters, activation, initializer, bias):
        super(UNet, self).__init__()
        
        self.conv1 = ConvBlock(filters=filters,
                               activation=activation,
                               initializer=initializer,
                               bias=bias)
        self.conv2 = ConvBlock(filters=filters*2,
                               activation=activation,
                               initializer=initializer,
                               bias=bias)
        self.conv3 = ConvBlock(filters=filters*4,
                               activation=activation,
                               initializer=initializer,
                               bias=bias)
        self.conv4 = ConvBlock(filters=filters*8,
                               activation=activation,
                               initializer=initializer,
                               bias=bias)
        self.conv5 = ConvBlock(filters=filters*16,
                               activation=activation,
                               initializer=initializer,
                               bias=bias)
        
        self.convt1 = l.Conv2DTranspose(filters=filters*8,
                                        kernel_size=3,
                                        padding='same',
                                        strides=2,
                                        use_bias=bias,
                                        kernel_initializer=initializer)
        self.conv6 = ConvBlock(filters=filters*8,
                               activation=activation,
                                initializer=initializer,
                                bias=bias)
        
        self.convt2 = l.Conv2DTranspose(filters=filters*4,
                                        kernel_size=3,
                                        padding='same',
                                        strides=2,
                                        use_bias=bias,
                                        kernel_initializer=initializer)
        self.conv7 = ConvBlock(filters=filters*4,
                               activation=activation,
                                initializer=initializer,
                                bias=bias)
        
        self.convt3 = l.Conv2DTranspose(filters=filters*2,
                                        kernel_size=3,
                                        padding='same',
                                        strides=2,
                                        use_bias=bias,
                                        kernel_initializer=initializer)
        self.conv8 = ConvBlock(filters=filters*2,
                               activation=activation,
                                initializer=initializer,
                                bias=bias)
        
        self.convt4 = l.Conv2DTranspose(filters=filters,
                                        kernel_size=3,
                                        padding='same',
                                        strides=2,
                                        use_bias=bias,
                                        kernel_initializer=initializer)
        self.conv9 = ConvBlock(filters=filters,
                               activation=activation,
                                initializer=initializer,
                                bias=bias)
        
        self.conv10 = l.Conv2D(filters=2,
                               kernel_size=1,
                               use_bias=bias,
                               padding='same',
                               kernel_initializer=initializer)
        
        self.dropout = l.Dropout(0.1)
        
    def call(self, t):
        t = self.conv2(t)
        
        t1 = l.MaxPooling2D()(t)
        t1 = self.dropout(t1)
        t1 = self.conv3(t1)
        
        t2 = l.MaxPooling2D()(t1)
        t2 = self.dropout(t2)
        t2 = self.conv4(t2)
        
        t3 = l.MaxPooling2D()(t2)
        t3 = self.dropout(t3)
        t3 = self.conv6(t3)
        
#         t4 = l.MaxPooling2D()(t3)
#         t4 = self.conv5(t4)
        
#         t4 = self.convt1(t4)
#         t4 = l.Concatenate()([t4, t3])
#         t4 = self.conv6(t4)
        
        t4 = self.convt2(t3)
        t4 = l.Concatenate()([t4, t2])
        t4 = self.dropout(t4)
        t4 = self.conv7(t4)
        
        t4 = self.convt3(t4)
        t4 = l.Concatenate()([t4, t1])
        t4 = self.dropout(t4)
        t4 = self.conv8(t4)
        
        t4 = self.convt4(t4)
        t4 = l.Concatenate()([t4, t])
        t4 = self.dropout(t4)
        t4 = self.conv9(t4)
        
        t = self.conv10(t4)
        
        return t

def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred)
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice


# Fungsi untuk membuat elevation map & markers
def create_elevation_map(img,b_min=30,b_max=150):
    markers = np.zeros_like(img)
    markers[img < b_min] = 1
    markers[img > b_max] = 2

    elemap = sobel(img)

    plt.imshow(elemap,cmap=plt.cm.gray)
    plt.title("elevation map")

    return elemap,markers
# Fungsi segmentasi RAG-Watershed
def RAGWs_Segment(img,threshold=30):
    labels1= segmentation.slic(img,compactness=30,n_segments=400,start_label=1)
    out1= color.label2rgb(labels1,img, kind='avg',bg_label=0)

    g = graph.rag_mean_color(img,labels1)
    labels2= graph.cut_threshold(labels1,g,threshold)
    out2= color.label2rgb(labels2,img,kind='avg',bg_label=0)

    grayout2 = cv2.cvtColor(out2, cv2.COLOR_BGR2GRAY)

    elemap,markers = create_elevation_map(grayout2)

    segmentation_result = segmentation.watershed(elemap,markers)

    return segmentation_result