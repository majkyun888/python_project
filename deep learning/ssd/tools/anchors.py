# -*- encoding=utf-8 -*-
import numpy as np

class PriorBox(object):

    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None, flip=True, variances=[0.1], clip=True, **kwargs):
        self.waxis = 1
        self.haxis = 0

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

    def call(self, input_shape, mask=None):
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
        x = np.linspace(.5 * step_x, img_width - .5 * step_x, layer_width)
        y = np.linspace(.5 * step_y, img_height - .5 * step_y, layer_height)
        center_x, center_y = np.meshgrid(x, y)
        center_x = center_x.reshape(-1, 1)
        center_y = center_y.reshape(-1, 1)
        num_priors = len(self.aspect_ratios)
        prior_boxs = np.concatenate((center_x, center_y), axis=1)
        prior_boxs = np.tile(prior_boxs, (1, 2 * num_priors))

        prior_boxs[:, ::4] -= box_widths
        prior_boxs[:, 1::4] -= box_heights
        prior_boxs[:, 2::4] += box_widths
        prior_boxs[:, 3::4] += box_heights

        prior_boxs[:, ::2] /= img_width
        prior_boxs[:, 1::2] /= img_height
        prior_boxs = prior_boxs.reshape(-1, 4)

        prior_boxs = np.minimum(np.maximum(prior_boxs, 0.0), 1.0)
        num_boxes = len(prior_boxs)

        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances.')

        prior_boxs = np.concatenate((prior_boxs, variances), axis=1)
        return prior_boxs


def get_anchors(img_size=(300, 300)):
    net = {}
    priorbox = PriorBox(img_size, min_size=30.0, max_size=60.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2], name='conv4_3_norm_mbox_priorbox')
    net['conv4_3_norm_mbox_priorbox'] = priorbox.call([38, 38])

    priorbox = PriorBox(img_size, min_size=60.0, max_size=111.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='fc7_mbox_priorbox')
    net['fc7_mbox_priorbox'] = priorbox.call([19, 19])

    priorbox = PriorBox(img_size, min_size=111.0, max_size=162.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv6_2_mbox_priorbox')
    net['conv6_2_mbox_priorbox'] = priorbox.call([10, 10])

    priorbox = PriorBox(img_size, min_size=162.0, max_size=213.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2], name='conv7_2_mbox_priorbox')
    net['conv7_2_mbox_priorbox'] = priorbox.call([5, 5])

    priorbox = PriorBox(img_size, min_size=213.0, max_size=264.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2], name='conv8_2_mbox_priorbox')
    net['conv8_2_mbox_priorbox'] = priorbox.call([3, 3])

    priorbox = PriorBox(img_size, min_size=264.0, max_size=315.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2], name='pool6_mbox_priorbox')
    net['pool6_mbox_priorbox'] = priorbox.call([1, 1])

    net['mbox_priorbox'] = np.concatenate([
        net['conv4_3_norm_mbox_priorbox'],
        net['fc7_mbox_priorbox'],
        net['conv6_2_mbox_priorbox'],
        net['conv7_2_mbox_priorbox'],
        net['conv8_2_mbox_priorbox'],
        net['pool6_mbox_priorbox']
    ], axis=0)

    return net['mbox_priorbox']