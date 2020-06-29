import numpy as np
from PIL import Image
import imageio
import os as os
from utils import get_bin, draw_bar, draw_ball
from math import pi

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
channels = 3
batch_size = 16
num_objects = 4

data_y = np.load('./Transformed_dataset/data_y.npy')

num_classes = 100

vx_bins = []
vy_bins = []
theta_bins = []

obj_data_mapping = {0: -1, 1: 1, 2: 3, 3: 2}

for obj in range(4):
    bin_vx = get_bin(data_y[:, obj, 0], num_classes)
    bin_vy = get_bin(data_y[:, obj, 1], num_classes)
    theta = get_bin(data_y[:, obj, 2], num_classes)

    vx_bins.append(bin_vx)
    vy_bins.append(bin_vy)
    theta_bins.append(theta)

dataset_path = './Generated_dataset/Task-4/'
test_subtask_no = str(2)

subtask = 'Task-4-' + test_subtask_no
subtask_folder = os.path.join(dataset_path, subtask)

task_no = subtask.split('-')[1]
subtask_no = subtask.split('-')[2]

file = 'task-0000' + str(task_no) + '_' + str(subtask_no) + '.npy'
data = np.load(os.path.join(dataset_path, file))
print(data.shape)

#################################################### STARTING HERE #####################################################
seq = 4

test_x_images = []
test_x_vel = []
pred = []

img_path = os.path.join(subtask_folder, str(seq) + '.png')
image = Image.open(img_path)
test_x_images.append(np.asarray(np.asarray(image) / 255, dtype=np.uint8))
image.close()
test_x_images = np.array(test_x_images)

all_posx = []
all_posy = []
all_theta = []

for obj in range(num_objects):
    posx = data[seq][obj_data_mapping[obj]][0] * IMAGE_WIDTH  # RED-BALL
    posy = (1. - data[seq][obj_data_mapping[obj]][1]) * IMAGE_HEIGHT
    theta = data[seq][obj_data_mapping[obj]][2] * 2 * pi
    all_posx.append(posx)
    all_posy.append(posy)
    all_theta.append(theta)

one_hot_encoded_data_y = np.load('./Transformed_dataset/one_hot-data_y.npy')
one_hot_encoded_data_y = one_hot_encoded_data_y.reshape(one_hot_encoded_data_y.shape[0],
                                                        num_objects, -1)

while seq <= data.shape[0] - 1:

    all_pred_x = []
    all_pred_y = []
    all_pred_theta = []

    for obj in range(4):
        pred_x = one_hot_encoded_data_y[141+seq - 4, obj, :num_classes]
        pred_y = one_hot_encoded_data_y[141+seq - 4, obj, num_classes: 2 * num_classes]
        theta = one_hot_encoded_data_y[141+seq - 4, obj, 2 * num_classes:]

        pred_x = np.argmax(pred_x, axis=-1)
        pred_y = np.argmax(pred_y, axis=-1)
        theta = np.argmax(theta, axis=-1)

        pred_x = np.array([(vx_bins[obj][pred_x + 1] + vx_bins[obj][pred_x]) / 2.])
        pred_y = np.array([(vy_bins[obj][pred_y + 1] + vy_bins[obj][pred_y]) / 2.])
        theta = np.array([(theta_bins[obj][theta + 1] + theta_bins[obj][theta]) / 2.])

        all_pred_x.append(pred_x)
        all_pred_y.append(pred_y)
        all_pred_theta.append(theta)

        # all_pred_x.append([data_y[seq-4][obj][0]])
        # all_pred_y.append([data_y[seq - 4][obj][1]])
        # all_pred_theta.append([data_y[seq - 4][obj][2]])

    scene = data[seq]
    idx = 0
    channel1 = None
    channel2 = None
    channel3 = None
    for obj in scene:
        cx = obj[0] * IMAGE_WIDTH
        cy = IMAGE_HEIGHT - (obj[1] * IMAGE_HEIGHT)
        diam = obj[3] * IMAGE_WIDTH
        theta = 2 * pi * (obj[2])

        typ = None
        if obj[4] == 1 and (idx == 1 or idx == 3):
            typ = 'ball-green'
        elif obj[4] == 1 and idx == 6:
            typ = 'ball-red'
        elif obj[5] == 1 and idx == 2:
            typ = 'bar-dynamic'
        elif obj[5] == 1:
            typ = 'bar-fixed'

        if typ == 'ball-green' or typ == 'ball-red' or typ == 'bar-dynamic':
            pass

        elif typ == 'bar-fixed':
            channel2 = draw_bar(channel2, cx, cy, diam, 5, theta)

        idx += 1

    filename = 'task-0000' + str(task_no) + '_' + test_subtask_no + '.npy'

    ################################################# Draw next state predictions ###################################

    for obj in range(4):
        diam = data[seq][obj_data_mapping[obj]][3] * IMAGE_WIDTH
        width = 5.
        next_cx_pred = all_posx[obj] + all_pred_x[obj][0] * IMAGE_WIDTH
        next_cy_pred = all_posy[obj] + all_pred_y[obj][0] * IMAGE_HEIGHT
        next_theta_pred = all_theta[obj] + all_pred_theta[obj][0] * 2 * pi

        all_posx[obj] = next_cx_pred
        all_posy[obj] = next_cy_pred
        all_theta[obj] = next_theta_pred

        while next_theta_pred > 2 * pi:
            next_theta_pred -= 2 * pi
        while next_theta_pred < 0:
            next_theta_pred += 2*pi

        if obj == 0:  # RED-BALL
            channel1 = draw_ball(channel1, next_cx_pred, next_cy_pred, diam)

        elif obj == 1 or obj == 2: # BLUE-BALLS
            channel3 = draw_ball(channel3, next_cx_pred, next_cy_pred, diam)

        elif obj == 3:  # BAR-DYNAMIC
            channel3 = draw_bar(channel3, next_cx_pred, next_cy_pred, diam, 5, next_theta_pred)

    # ####################################################################################################################

    channel1 = np.expand_dims(channel1, -1)
    channel2 = np.expand_dims(channel2, -1)
    channel3 = np.expand_dims(channel3, -1)

    bitmap_image = np.concatenate([channel1, channel2, channel3], axis=-1).astype(np.uint8)
    subtask_no = filename.split('_')[1].split('.')[0]
    try:
        os.mkdir('./Testing_train_data' + '/Task-' + filename[9] + '-' + subtask_no + '_prediction')
    except:
        pass

    imageio.imwrite(
        os.path.join('./Testing_train_data' + '/Task-' + filename[9] + '-' + str(subtask_no) + '_prediction',
                     str(seq) + '.png'), bitmap_image)

    test_x_images = [bitmap_image]

    seq += 1
