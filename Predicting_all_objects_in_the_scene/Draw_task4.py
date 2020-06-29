import numpy as np
from PIL import Image, ImageDraw
import imageio
import os as os
from math import pi
from utils import makeRectangle, draw_ball, draw_bar

# path to where the numpy arrays of generated dataset are stored
dataset_path = './Generated_dataset/Task-4'

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

for filename in os.listdir(dataset_path):
    if filename.startswith('Task'):
        continue
    counter = 0
    print("Laded file: ", filename)
    data = os.path.join(dataset_path, filename)
    data = np.load(data)
    print(data.shape)

    for scene in data:
        idx = 0
        channel1 = None
        channel2 = None
        channel3 = None
        for obj in scene:
            cx = obj[0] * IMAGE_WIDTH
            cy = IMAGE_HEIGHT - (obj[1] * IMAGE_WIDTH)
            theta = obj[2] * 2 * pi
            diam = obj[3] * IMAGE_WIDTH

            typ = None
            if obj[4] == 1 and (idx == 1 or idx == 3):
                typ = 'ball-green'
            elif obj[4] == 1 and idx == 6:
                typ = 'ball-red'
            elif obj[5] == 1 and idx == 2:
                typ = 'bar-dynamic'
            elif obj[5] == 1:
                typ = 'bar-fixed'

            if typ == 'ball-green':
                channel3 = draw_ball(channel3, cx, cy, diam)

            elif typ == 'ball-red':
                channel1 = draw_ball(channel1, cx, cy, diam)

            elif typ == 'bar-fixed':
                channel2 = draw_bar(channel2, cx, cy, diam, 5, theta)

            elif typ == 'bar-dynamic':
                channel3 = draw_bar(channel3, cx, cy, diam, 5, theta)

            idx += 1

        channel1 = np.expand_dims(channel1, -1)
        channel2 = np.expand_dims(channel2, -1)
        channel3 = np.expand_dims(channel3, -1)

        bitmap_image = np.concatenate([channel1, channel2, channel3], axis=-1).astype(np.uint8)
        subtask_no = filename.split('_')[1].split('.')[0]
        try:
            os.mkdir(dataset_path + '/Task-' + filename[9] + '-' + subtask_no)
        except:
            pass

        imageio.imwrite(
            os.path.join(dataset_path + '/Task-' + filename[9] + '-' + str(subtask_no), str(counter) + '.png'),
            bitmap_image)
        counter += 1
