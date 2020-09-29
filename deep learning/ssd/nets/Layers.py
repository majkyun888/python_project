# -*- encoding=utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import InputSpec
from tensorflow.keras.layers import Layer

class Normalize(Layer):
    def __init__(self, scale, **kwargs):
        self.axis = 3
        self.scale = scale
        super(Normalize, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        shape = (input_shape[self.axis],)
        init_gamma = self.scale * np.ones(shape)
        self.gamma = K.variable(init_gamma, name='{}_gamma'.format(self.name))

    def call(self, x, mask=None):
        output = K.l2_normalize(x, self.axis)
        output *= self.gamma
        return output


class PriorBox(Layer):
    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, variances=[0.1], clip=True, **kwargs):

        self.waxis = 2
        self.haxis = 1

        self.img_size = img_size
        if min_size <= 0:
            raise Exception('min_size must be positive.')

        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]
        if max_size:
            if max_size < min_size:
                raise Exception('max_size must be greater than min_size.')
            self.aspect_ratios.append(1.0)
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances)
        self.clip = True
        super(PriorBox, self).__init__(**kwargs)

    def call(self, x, mask=None):
        if hasattr(x, '_keras_shape'):
            input_shape = x._keras_shape
        elif hasattr(K, 'int_shape'):
            input_shape = K.int_shape(x)

        layer_width = input_shape[self.waxis]
        layer_height = input_shape[self.haxis]

        img_width = self.img_size[0]
        img_height = self.img_size[1]

        box_widths = []
        box_heights = []
        for ratio in self.aspect_ratios:
            if ratio == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ratio == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ratio != 1:
                box_widths.append(self.min_size * np.sqrt(ratio))
                box_heights.append(self.min_size / np.sqrt(ratio))
        box_widths = .5 * np.array(box_widths)
        box_heights = .5 * np.array(box_heights)
        step_x = img_width / layer_width
        step_y = img_height / layer_height
        Linex = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width)
        Liney = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)
        center_x, center_y = np.meshgrid(Linex, Liney)
        center_x = center_x.reshape(-1, 1)
        center_y = center_y.reshape(-1, 1)
        num_priors = len(self.aspect_ratios)
        priorBoxs = np.concatenate((center_x, center_y), axis=1)
        priorBoxs = np.tile(priorBoxs, (1, 2 * num_priors))
        priorBoxs[:, ::4] -= box_widths
        priorBoxs[:, 1::4] -= box_heights
        priorBoxs[:, 2::4] += box_widths
        priorBoxs[:, 3::4] += box_heights
        priorBoxs[:, ::2] /= img_width
        priorBoxs[:, 1::2] /= img_height
        priorBoxs = priorBoxs.reshape(-1, 4)

        priorBoxs = np.minimum(np.maximum(priorBoxs, 0.0), 1.0)
        num_boxes = len(priorBoxs)

        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')

        priorBoxs = np.concatenate((priorBoxs, variances), axis=1)
        prior_boxes_tensor = K.expand_dims(tf.cast(priorBoxs, dtype=tf.float32), 0)

        pattern = [tf.shape(x)[0], 1, 1]
        prior_boxes_tensor = tf.tile(prior_boxes_tensor, pattern)

        return prior_boxes_tensor
