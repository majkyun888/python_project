# -*- encoding=utf-8 -*-

import numpy as np
from tools.anchors import get_anchors
from nets.ssd import SSD300
from collections import namedtuple
from tensorflow.keras.optimizers import Adam, RMSprop
from nets.training import Generator, MultiboxLoss
from tools.util import ModelCheckpoint, BBoxUtility
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
np.random.seed(6666)


SSD_PARAMS = namedtuple('SSD_PARAMS',
                        [
                            'num_classes',
                            'img_shape',
                            'val_split'
                        ])

if __name__ == "__main__":
    log_dir = "logs/" # use for tensorboard
    annotation_path = 'train.txt'
    ssd_params = SSD_PARAMS(
        img_shape=(300, 300, 3), # last channel,
        num_classes=2,
        val_split=0.1
    )

    priors = get_anchors()
    bbox_util = BBoxUtility(ssd_params.num_classes, priors)
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.shuffle(lines)
    num_val = int(len(lines) * ssd_params.val_split)
    num_train = len(lines) - num_val

    model = SSD300(ssd_params.img_shape, num_classes=ssd_params.num_classes)
    #model.load_weights('model_data/ssd_weights.h5', by_name=True, skip_mismatch=True)

    logging = TensorBoard(log_dir=log_dir) # user for visdom
    checkPoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5', monitor='val_loss',
                                 save_weights_only=True, save_best_only=True, period=10)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=1)


    if True:
        BATCH_SIZE = 32
        Lr = 5e-4
        InitStart = 0
        Epoch = 1000
        gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:], (ssd_params.img_shape[0], ssd_params.img_shape[1]), ssd_params.num_classes)
        model.compile(optimizer=Adam(lr=Lr), loss=MultiboxLoss(ssd_params.num_classes, neg_pos_ratio=3.0).compute_loss)
        model.fit(gen.generate(True),
                  initial_epoch=InitStart,
                  epochs=Epoch,
                  steps_per_epoch=num_train / BATCH_SIZE,
                  validation_data=gen.generate(False),
                  validation_steps=num_val / BATCH_SIZE,
                  callbacks=[logging, checkPoint, reduce_lr, early_stopping]
                  )

    if True:
        BATCH_SIZE = 8
        Lr = 1e-4
        InitStart = 0
        Epoch = 5000
        gen = Generator(bbox_util, BATCH_SIZE, lines[:num_train], lines[num_train:], (ssd_params.img_shape[0], ssd_params.img_shape[1]), ssd_params.num_classes)
        model.compile(optimizer=RMSprop(lr=Lr), loss=MultiboxLoss(ssd_params.num_classes, neg_pos_ratio=3.0).compute_loss)
        model.fit(gen.generate(True),
                  initial_epoch=InitStart,
                  epochs=Epoch,
                  steps_per_epoch=num_train / BATCH_SIZE,
                  validation_data=gen.generate(False),
                  validation_steps=num_val / BATCH_SIZE,
                  callbacks=[logging, checkPoint, reduce_lr]
                  )



