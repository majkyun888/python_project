# -*- encoding=utf-8 -*-
from ssd import SSD
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import time

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

ssd = SSD()
imgPath = input("Input image filename:")
imgPath += "\\%d.jpg"
capture = cv2.VideoCapture(imgPath)

while (True):
    t1 = time.time()
    ref, frame = capture.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(np.uint8(frame))
    frame = np.array(ssd.detect_image(frame))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("Sequence", frame)
    c = cv2.waitKey(150) & 0xff

