import numpy as np
from model import get_model
import os as os
from PIL import Image, ImageDraw
import imageio
from utils import get_all_bins, delta_pos, delta_theta, draw_ball, draw_bar
from math import pi
from keras.models import load_model

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256
num_objects = 4
channels = 5
input_feature_dims = 14
num_classes = 100

dataset_path = './Generated_dataset/Task-4/'
test_subtask_no = str(6)

subtask = 'Task-4-' + test_subtask_no
subtask_folder = os.path.join(dataset_path, subtask)

task_no = subtask.split('-')[1]
subtask_no = subtask.split('-')[2]

file = 'task-0000' + str(task_no) + '_' + str(subtask_no) + '.npy'
data = np.load(os.path.join(dataset_path, file))
print(data.shape)


########################################################################################################################
model = get_model()
model.load_weights('./Saved_Models_and_Weights/saved-model-30-nan.hdf5')
########################################################################################################################

data_y = np.load('./Transformed_dataset/data_y.npy')
obj_data_mapping = {0: -1, 1: 1, 2: 3, 3: 2}
vx_bins, vy_bins, theta_bins = get_all_bins(data_y, num_classes)

#################################################### STARTING HERE #####################################################

seq = 4

test_x_images = []
test_x_features = []

img_path_t0 = os.path.join(subtask_folder, str(seq) + '.png')
img_path_t1 = os.path.join(subtask_folder, str(seq - 1) + '.png')

image_t0 = np.asarray(np.asarray(Image.open(img_path_t0)) / 255., dtype=np.uint8)
image_t1 = np.asarray(np.asarray(Image.open(img_path_t1)) / 255., dtype=np.uint8)

# all three channels of t-0 frame and 2 channels of t-1 frame (skip the fixed objects channel)
channel_1_t1 = np.expand_dims(image_t1[:, :, 0], axis=-1)
channel_3_t1 = np.expand_dims(image_t1[:, :, 2], axis=-1)
input_image = np.concatenate([image_t0, channel_1_t1, channel_3_t1], axis=-1)

test_x_images.append(input_image)
test_x_images = np.array(test_x_images)

# FEATURES OF PREV 4 TIMESTEPS of 1st OBJECT (RED-BALL)
feature1 = []
for timestep in range(1, 5):
    vx = delta_pos(data, seq - timestep, -1, 0, False)
    vy = delta_pos(data, seq - timestep, -1, 1, True)
    theta = delta_theta(data, seq - timestep, -1, 2)
    feature1.append(vx)
    feature1.append(vy)
    feature1.append(theta)

shape = data[seq][-1][4:6]  # either ball or bar in this task
feature1.extend(shape)

# FEATURES OF PREV 4 TIMESTEPS of 2nd OBJECT (BLUE-BALL-1)
feature2 = []
for timestep in range(1, 5):
    vx = delta_pos(data, seq - timestep, 1, 0, False)
    vy = delta_pos(data, seq - timestep, 1, 1, True)
    theta = delta_theta(data, seq - timestep, 1, 2)
    feature2.append(vx)
    feature2.append(vy)
    feature2.append(theta)

shape = data[seq][1][4:6]  # either ball or bar in this task
feature2.extend(shape)

# FEATURES OF PREV 4 TIMESTEPS of 3rd OBJECT (BLUE-BALL-2)
feature3 = []
for timestep in range(1, 5):
    vx = delta_pos(data, seq - timestep, 3, 0, False)
    vy = delta_pos(data, seq - timestep, 3, 1, True)
    theta = delta_theta(data, seq - timestep, 3, 2)
    feature3.append(vx)
    feature3.append(vy)
    feature3.append(theta)

shape = data[seq][3][4:6]  # either ball or bar in this task
feature3.extend(shape)

# FEATURES OF PREV 4 TIMESTEPS of 4th OBJECT (BAR-DYNAMIC)
feature4 = []
for timestep in range(1, 5):
    vx = delta_pos(data, seq - timestep, 2, 0, False)
    vy = delta_pos(data, seq - timestep, 2, 1, True)
    theta = delta_theta(data, seq - timestep, 2, 2)
    feature4.append(vx)
    feature4.append(vy)
    feature4.append(theta)

shape = data[seq][2][4:6]  # either ball or bar in this task
feature4.extend(shape)

test_x_features.append([feature1, feature2, feature3, feature4])

# INITIAL POSITION OF DYNAMIC OBJECTS
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

while seq <= data.shape[0] - 1:

    test_input = []
    for i in range(num_objects):
        test_input.append(test_x_images)
        test_input.append(np.array([test_x_features[0][i]]))

    pred = model.predict(test_input)
    pred = pred.reshape(1, num_objects, -1)

    all_pred_x = []
    all_pred_y = []
    all_pred_theta = []

    for obj in range(4):
        pred_x = pred[0, obj, :num_classes]
        pred_y = pred[0, obj, num_classes: 2 * num_classes]
        theta = pred[0, obj, 2 * num_classes:]

        pred_x = np.argmax(pred_x, axis=-1)
        pred_y = np.argmax(pred_y, axis=-1)
        theta = np.argmax(theta, axis=-1)

        pred_x = np.array([(vx_bins[obj][pred_x + 1] + vx_bins[obj][pred_x]) / 2.])
        pred_y = np.array([(vy_bins[obj][pred_y + 1] + vy_bins[obj][pred_y]) / 2.])
        theta = np.array([(theta_bins[obj][theta + 1] + theta_bins[obj][theta]) / 2.])

        all_pred_x.append(pred_x)
        all_pred_y.append(pred_y)
        all_pred_theta.append(theta)

        # UPDATING
        for timestep in range(3, 12, 3):
            test_x_features[0][obj][timestep] = test_x_features[0][obj][timestep-3]
            test_x_features[0][obj][timestep + 1] = test_x_features[0][obj][timestep-2]
            test_x_features[0][obj][timestep + 2] = test_x_features[0][obj][timestep-1]

        test_x_features[0][obj][0] = pred_x[0]
        test_x_features[0][obj][1] = pred_y[0]
        test_x_features[0][obj][2] = theta[0]

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

        # if(obj==0):
        #     print(seq, all_pred_y[obj][0], next_cy_pred)

        all_posx[obj] = next_cx_pred
        all_posy[obj] = next_cy_pred
        all_theta[obj] = next_theta_pred

        while next_theta_pred > 2 * pi:
            next_theta_pred -= 2 * pi
        while next_theta_pred < 0:
            next_theta_pred += 2 * pi

        if obj == 0:  # RED-BALL
            channel1 = draw_ball(channel1, next_cx_pred, next_cy_pred, diam)

        elif obj == 1 or obj == 2:  # BLUE-BALLS
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
        os.mkdir('./Predictions_single_step' + '/Task-' + filename[9] + '-' + subtask_no + '_prediction')
    except:
        pass

    imageio.imwrite(
        os.path.join('./Predictions_single_step' + '/Task-' + filename[9] + '-' + str(subtask_no) + '_prediction',
                     str(seq) + '.png'), bitmap_image)

    # bitmap_image = bitmap_image.astype(np.float32)
    bitmap_image = bitmap_image // 255
    image_t0 = bitmap_image
    image_t1 = test_x_images[0][:, :, :3]

    channel_1_t1 = np.expand_dims(image_t1[:, :, 0], axis=-1)
    channel_3_t1 = np.expand_dims(image_t1[:, :, 2], axis=-1)
    input_image = np.concatenate([image_t0, channel_1_t1, channel_3_t1], axis=-1)

    test_x_images = np.array([input_image])

    seq += 1
