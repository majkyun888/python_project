# -*- encoding=utf-8 -*-
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input
from tensorflow.keras.layers import Concatenate
from nets.VGG16 import VGG16
from nets.Layers import Normalize, PriorBox


def SSD300(input_shape, num_classes):
    input_tensor = Input(shape=input_shape)
    img_size = (input_shape[1], input_shape[0]) # last channel

    net = VGG16(input_tensor)

    # 38x38x512
    net['conv4_3_norm'] = Normalize(20, name='conv4_3_norm') (net['conv4_3'])
    num_priors = 4
    # 38x38x16 first featureMap
    net['conv4_3_norm_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv4_3_norm_mbox_loc') (net['conv4_3_norm'])
    net['conv4_3_norm_mbox_loc_flat'] = Flatten(name='conv4_3_norm_mbox_loc_flat') (net['conv4_3_norm_mbox_loc'])
    # [38, 38, num_classes * num_priors] first featureMap
    net['conv4_3_norm_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name='conv4_3_norm_mbox_conf') (net['conv4_3_norm'])
    net['conv4_3_norm_mbox_conf_flat'] = Flatten(name='onv4_3_norm_mbox_conf_flat') (net['conv4_3_norm_mbox_conf'])
    priorbox = PriorBox(img_size, 30.0, max_size=60.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2], name='conv4_3_norm_mbox_priorbox')
    # 38 * 38 * 4 = 5576
    net['conv4_3_norm_mbox_priorbox'] = priorbox.call(net['conv4_3_norm'])

    num_priors = 6
    # [19, 19, 24]
    net['fc7_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='fc7_mbox_loc') (net['fc7'])
    net['fc7_mbox_loc_flat'] = Flatten(name='fc7_mbox_loc_flat') (net['fc7_mbox_loc'])
    # [19, 19, num_priors * num_classes] second featureMap
    net['fc7_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name='fc7_mbox_conf') (net['fc7'])
    net['fc7_mbox_conf_flat'] = Flatten(name='fc7_mbox_conf_flat') (net['fc7_mbox_conf'])
    priorbox = PriorBox(img_size, 60.0, max_size=111.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='fc7_mbox_priorbox')
    # 19 * 19 * 6 = 2166
    net['fc7_mbox_priorbox'] = priorbox(net['fc7'])

    num_priors = 6
    # [10, 10, 24]
    net['conv6_2_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv6_2_mbox_loc') (net['conv6_2'])
    net['conv6_2_mbox_loc_flat'] = Flatten(name='conv6_2_mbox_loc_flat') (net['conv6_2_mbox_loc'])
    net['conv6_2_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name='conv6_2_mbox_conf') (net['conv6_2'])
    net['conv6_2_mbox_conf_flat'] = Flatten(name='conv6_2_mbox_conf_flat') (net['conv6_2_mbox_conf'])
    priorbox = PriorBox(img_size, 111.0, max_size=162.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv6_2_mbox_priorbox')
    # 10x10x6 = 600
    net['conv6_2_mbox_priorbox'] = priorbox(net['conv6_2'])

    num_priors = 6
    # [5, 5, 24]
    net['conv7_2_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv7_2_mbox_loc') (net['conv7_2'])
    net['conv7_2_mbox_loc_flat'] = Flatten(name='conv7_2_mbox_loc_flat') (net['conv7_2_mbox_loc'])
    net['conv7_2_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name='conv7_2_mbox_conf') (net['conv7_2'])
    net['conv7_2_mbox_conf_flat'] = Flatten(name='conv7_2_mbox_conf_flat') (net['conv7_2_mbox_conf'])
    priorbox = PriorBox(img_size, 162.0, max_size=213.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv7_2_mbox_priorbox')
    # 5x5x6 = 150
    net['conv7_2_mbox_priorbox'] = priorbox(net['conv7_2'])

    num_priors = 4
    net['conv8_2_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv8_2_mbox_loc') (net['conv8_2'])
    net['conv8_2_mbox_loc_flat'] = Flatten(name='conv8_2_mbox_loc_flat') (net['conv8_2_mbox_loc'])
    net['conv8_2_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name='conv8_2_mbox_conf') (net['conv8_2'])
    net['conv8_2_mbox_conf_flat'] = Flatten(name='conv8_2_mbox_conf_flat') (net['conv8_2_mbox_conf'])
    # 3x3x4 = 36
    priorbox = PriorBox(img_size, 213.0, max_size=264.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2], name='conv8_2_mbox_priorbox')
    net['conv8_2_mbox_priorbox'] = priorbox(net['conv8_2'])

    num_priors = 4
    net['conv9_2_mbox_loc'] = Conv2D(num_priors * 4, kernel_size=(3, 3), padding='same', name='conv9_2_mbox_loc') (net['conv9_2'])
    net['conv9_2_mbox_loc_flat'] = Flatten(name='conv9_2_mbox_loc_flat') (net['conv9_2_mbox_loc'])
    net['conv9_2_mbox_conf'] = Conv2D(num_priors * num_classes, kernel_size=(3, 3), padding='same', name='conv9_2_mbox_conf') (net['conv9_2'])
    net['conv9_2_mbox_conf_flat'] = Flatten(name='conv9_2_mbox_conf_flat') (net['conv9_2_mbox_conf'])
    # 1x1x4
    priorbox = PriorBox(img_size, 264.0, max_size=315.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2], name='conv9_2_mbox_priorbox')
    net['conv9_2_mbox_priorbox'] = priorbox(net['conv9_2'])

    # a = net['conv4_3_norm_mbox_loc_flat']
    # b = net['fc7_mbox_loc_flat']
    # d = net['conv6_2_mbox_loc_flat']
    # e = net['conv7_2_mbox_loc_flat']
    # f = net['conv8_2_mbox_loc_flat']
    # g = net['conv9_2_mbox_loc_flat']
    # v = net['mbox_loc']

    net['mbox_loc'] = Concatenate(axis=1, name='mbox_loc') ([
        net['conv4_3_norm_mbox_loc_flat'],
        net['fc7_mbox_loc_flat'],
        net['conv6_2_mbox_loc_flat'],
        net['conv7_2_mbox_loc_flat'],
        net['conv8_2_mbox_loc_flat'],
        net['conv9_2_mbox_loc_flat']
        ])

    net['mbox_conf'] = Concatenate(axis=1, name='mbox_conf') ([
        net['conv4_3_norm_mbox_conf_flat'],
        net['fc7_mbox_conf_flat'],
        net['conv6_2_mbox_conf_flat'],
        net['conv7_2_mbox_conf_flat'],
        net['conv8_2_mbox_conf_flat'],
        net['conv9_2_mbox_conf_flat']
    ])

    net['mbox_priorbox'] = Concatenate(axis=1, name='mbox_priorbox') ([
        net['conv4_3_norm_mbox_priorbox'],
        net['fc7_mbox_priorbox'],
        net['conv6_2_mbox_priorbox'],
        net['conv7_2_mbox_priorbox'],
        net['conv8_2_mbox_priorbox'],
        net['conv9_2_mbox_priorbox']
    ])

    # net['mbox_loc'] = tf.reshape((-1, 4), name='mbox_loc_final') (net['mbox_loc'])
    # 最终8732个预测框, 回归问题
    net['mbox_loc'] = tf.keras.layers.Reshape((-1, 4), name='mbox_loc_final') (net['mbox_loc'])

    # finally [8732, num_classes]
    net['mbox_conf'] = tf.keras.layers.Reshape((-1, num_classes), name='mbox_conf_logits') (net['mbox_conf'])
    net['mbox_conf'] = tf.keras.layers.Activation('softmax', name='mbox_conf_final') (net['mbox_conf'])
    net['predictions'] = Concatenate(axis=2, name='predictions') ([
        net['mbox_loc'],
        net['mbox_conf'],
        net['mbox_priorbox']
    ])

    model = tf.keras.Model(net['input'], net['predictions'])    # 闭包
    return model

