# -*- encoding=utf-8 -*-

import tensorflow as tf
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D

def VGG16(input_tensor):

    net = {}
    net['input'] = input_tensor

    net['conv1_1'] = Conv2D(64, kernel_size=(3,3), activation=tf.nn.relu, padding='same', name='conv1_1') \
        (net['input'])
    net['conv1_2'] = Conv2D(64, kernel_size=(3,3), activation=tf.nn.relu, padding='same', name='conv1_2') \
        (net['conv1_1'])
    net['pool1'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool1') \
        (net['conv1_2'])

    net['conv2_1'] = Conv2D(128, kernel_size=(3,3), activation=tf.nn.relu, padding='same', name='conv2_1')\
        (net['pool1'])
    net['conv2_2'] = Conv2D(128, kernel_size=(3,3), activation=tf.nn.relu, padding='same', name='conv2_2')\
        (net['conv2_1'])
    net['pool2'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool2') \
        (net['conv2_2'])

    net['conv3_1'] = Conv2D(256, kernel_size=(3, 3), activation=tf.nn.relu, padding='same', name='conv3_1') \
        (net['pool2'])
    net['conv3_2'] = Conv2D(256, kernel_size=(3, 3), activation=tf.nn.relu, padding='same', name='conv3_2') \
        (net['conv3_1'])
    net['conv3_3'] = Conv2D(256, kernel_size=(3, 3), activation=tf.nn.relu, padding='same', name='conv3_3') \
        (net['conv3_2'])
    net['pool3'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool3') \
        (net['conv3_3'])

    net['conv4_1'] = Conv2D(512, kernel_size=(3, 3), activation=tf.nn.relu, padding='same', name='conv4_1') \
        (net['pool3'])
    net['conv4_2'] = Conv2D(512, kernel_size=(3, 3), activation=tf.nn.relu, padding='same', name='conv4_2') \
        (net['conv4_1'])
    net['conv4_3'] = Conv2D(512, kernel_size=(3, 3), activation=tf.nn.relu, padding='same', name='conv4_3') \
        (net['conv4_2'])
    net['pool4'] = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool4') \
        (net['conv4_3'])

    net['conv5_1'] = Conv2D(512, kernel_size=(3, 3), activation=tf.nn.relu, padding='same', name='conv5_1') \
        (net['pool4'])
    net['conv5_2'] = Conv2D(512, kernel_size=(3, 3), activation=tf.nn.relu, padding='same', name='conv5_2') \
        (net['conv5_1'])
    net['conv5_3'] = Conv2D(512, kernel_size=(3, 3), activation=tf.nn.relu, padding='same', name='conv5_3')\
        (net['conv5_2'])
    net['pool5'] = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='pool5') \
        (net['conv5_3'])

    net['fc6'] = Conv2D(1024, kernel_size=(3, 3), padding='same', activation=tf.nn.relu, dilation_rate=(6, 6), name='fc6') \
        (net['pool5'])
    net['fc7'] = Conv2D(1024, kernel_size=(1, 1), activation=tf.nn.relu, padding='same', name='fc7') \
        (net['fc6'])

    net['conv6_1'] = Conv2D(256, kernel_size=(1, 1), activation=tf.nn.relu, padding='same', name='conv6_1') \
        (net['fc7'])
    net['conv6_2'] = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding') (net['conv6_1'])
    net['conv6_2'] = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), activation=tf.nn.relu, name='conv6_2') \
        (net['conv6_2'])

    net['conv7_1'] = Conv2D(128, kernel_size=(1, 1), activation=tf.nn.relu, padding='same', name='conv7_1') \
        (net['conv6_2'])

    net['conv7_2'] = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding') \
        (net['conv7_1'])
    net['conv7_2'] = Conv2D(256, kernel_size=(3, 3), strides=(2, 2), activation=tf.nn.relu, padding='valid', name='conv7_2') \
        (net['conv7_2'])

    net['conv8_1'] = Conv2D(128, kernel_size=(1, 1), activation=tf.nn.relu, padding='same', name='conv8_1') \
        (net['conv7_2'])
    net['conv8_2'] = Conv2D(256, kernel_size=(3, 3), activation=tf.nn.relu, padding='valid', name='conv8_2') \
        (net['conv8_1'])

    net['conv9_1'] = Conv2D(128, kernel_size=(1, 1), activation=tf.nn.relu, padding='same', name='conv9_1') \
        (net['conv8_2'])
    net['conv9_2'] = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu, padding='valid', name='conv9_2') \
        (net['conv9_1'])

    return net