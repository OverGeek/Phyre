import numpy as np
import os as os
from PIL import Image
from utils import get_bin, convert_to_one_hot_encoded_bin

# Data Loading
dataset_path = './Generated_dataset/Task-4/'
num_objects = 4
data_x_images = []
data_x_features = []  # (frames, num_objects, features) - features = (vel_x, vel_y, angle, shape)
data_y = []


def delta_pos(data, seq, obj_index, feature_index, rev=False):
    if not rev:
        return data[seq + 1][obj_index][feature_index] - data[seq][obj_index][feature_index]
    else:
        return (1. - data[seq + 1][obj_index][feature_index]) - (1. - data[seq][obj_index][feature_index])


def delta_theta(data, seq, obj_index, feature_index):
    t1 = data[seq + 1][obj_index][feature_index]
    t0 = data[seq][obj_index][feature_index]

    return t1 - t0


for subtask_no in range(1, 51):
    subtask = 'Task-4-' + str(subtask_no)
    print(subtask)
    subtask_folder = os.path.join(dataset_path, subtask)

    task_no = subtask.split('-')[1]
    subtask_no = subtask.split('-')[2]

    file = 'task-0000' + str(task_no) + '_' + str(subtask_no) + '.npy'
    data = np.load(os.path.join(dataset_path, file))
    print(data.shape)

    for seq in range(4, len(os.listdir(subtask_folder)) - 1):
        img_path_t0 = os.path.join(subtask_folder, str(seq) + '.png')
        img_path_t1 = os.path.join(subtask_folder, str(seq - 1) + '.png')

        image_t0 = np.asarray(np.asarray(Image.open(img_path_t0)) / 255., dtype=np.float32)
        image_t1 = np.asarray(np.asarray(Image.open(img_path_t1)) / 255., dtype=np.float32)

        # all three channels of t-0 frame and 2 channels of t-1 frame (skip the fixed objects channel)
        channel_1_t1 = np.expand_dims(image_t1[:, :, 0], axis=-1)
        channel_3_t1 = np.expand_dims(image_t1[:, :, 2], axis=-1)
        input_image = np.concatenate([image_t0, channel_1_t1, channel_3_t1], axis=-1)
        data_x_images.append(input_image)

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

        data_x_features.append([feature1, feature2, feature3, feature4])

        # Target features for first object (RED-BALL)
        target1 = []
        vx1 = delta_pos(data, seq, -1, 0, False)
        vy1 = delta_pos(data, seq, -1, 1, True)
        theta1 = delta_theta(data, seq, -1, 2)
        target1.append(vx1)
        target1.append(vy1)
        target1.append(theta1)

        # Target features for second object (BLUE-BALL-1)
        target2 = []
        vx2 = delta_pos(data, seq, 1, 0, False)
        vy2 = delta_pos(data, seq, 1, 1, True)
        theta2 = delta_theta(data, seq, 1, 2)
        target2.append(vx2)
        target2.append(vy2)
        target2.append(theta2)

        # Target features for third object (BLUE-BALL-2)
        target3 = []
        vx3 = delta_pos(data, seq, 3, 0, False)
        vy3 = delta_pos(data, seq, 3, 1, True)
        theta3 = delta_theta(data, seq, 3, 2)
        target3.append(vx3)
        target3.append(vy3)
        target3.append(theta3)

        # Target features for fourth object (BAR-DYNAMIC)
        target4 = []
        vx4 = delta_pos(data, seq, 2, 0, False)
        vy4 = delta_pos(data, seq, 2, 1, True)
        theta4 = delta_theta(data, seq, 2, 2)
        target4.append(vx4)
        target4.append(vy4)
        target4.append(theta4)

        data_y.append([target1, target2, target3, target4])

        seq += 1

data_x_images = np.array(data_x_images)
data_x_features = np.asarray(data_x_features).astype(np.float32)
data_y = np.asarray(data_y).astype(np.float32)

np.save('./Transformed_dataset/data_x_images.npy', data_x_images)
np.save('./Transformed_dataset/data_x_features.npy', data_x_features)
np.save('./Transformed_dataset/data_y.npy', data_y)

data_y = np.load('./Transformed_dataset/data_y.npy')

num_classes = 100

vx_bins = []
vy_bins = []
theta_bins = []

for obj in range(4):
    bin_vx = get_bin(data_y[:, obj, 0], num_classes)
    bin_vy = get_bin(data_y[:, obj, 1], num_classes)
    theta = get_bin(data_y[:, obj, 2], num_classes)

    vx_bins.append(bin_vx)
    vy_bins.append(bin_vy)
    theta_bins.append(theta)

# ONE-HOT ENCODING DATA_Y

one_hot_encoded_data_y = []

for frame in range(data_y.shape[0]):
    target = []
    for obj in range(data_y.shape[1]):
        one_hot_vx = convert_to_one_hot_encoded_bin(data_y[frame, obj, 0], vx_bins[obj], num_classes)
        one_hot_vy = convert_to_one_hot_encoded_bin(data_y[frame, obj, 1], vy_bins[obj], num_classes)
        one_hot_theta = convert_to_one_hot_encoded_bin(data_y[frame, obj, 2], theta_bins[obj], num_classes)

        target.append(np.concatenate([one_hot_vx, one_hot_vy, one_hot_theta], axis=-1).reshape(3*num_classes,))
    one_hot_encoded_data_y.append(target)

one_hot_encoded_data_y = np.array(one_hot_encoded_data_y)
one_hot_encoded_data_y = one_hot_encoded_data_y.reshape(one_hot_encoded_data_y.shape[0], -1)
np.save('./Transformed_dataset/one_hot-data_y.npy', one_hot_encoded_data_y)
