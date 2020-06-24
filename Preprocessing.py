import numpy as np
import os as os
from PIL import Image
from utils import get_bin, convert_to_one_hot_encoded_bin

# Data Loading
dataset_path = './Generated_dataset/Task-2/'

data_x_images = []
data_x_vel = []
data_y = []

# for subtask_no in range(46, 91):
#     subtask = 'Task-2-' + str(subtask_no)
#     print(subtask)
#     subtask_folder = os.path.join(dataset_path, subtask)
#
#     task_no = subtask.split('-')[1]
#     subtask_no = subtask.split('-')[2]
#
#     file = 'task-0000' + str(task_no) + '_' + str(subtask_no) + '.npy'
#     data = np.load(os.path.join(dataset_path, file))
#     print(data.shape)
#
#     for seq in range(4, len(os.listdir(subtask_folder)) - 1):
#         img_path_t0 = os.path.join(subtask_folder, str(seq) + '.png')
#         img_path_t1 = os.path.join(subtask_folder, str(seq - 1) + '.png')
#
#         image_t0 = np.asarray(np.asarray(Image.open(img_path_t0)) / 255., dtype=np.float32)
#         image_t1 = np.asarray(np.asarray(Image.open(img_path_t1)) / 255., dtype=np.float32)
#
#         # all three channels of t-0 frame and 2 channels of t-1 frame (skip the fixed objects channel)
#         channel_1_t1 = np.expand_dims(image_t1[:, :, 0], axis=-1)
#         channel_3_t1 = np.expand_dims(image_t1[:, :, 2], axis=-1)
#         input_image = np.concatenate([image_t0, channel_1_t1, channel_3_t1], axis=-1)
#         data_x_images.append(input_image)
#
#         # VEL OF PREV 4 TIMESTEPS
#         vxt1 = data[seq][-1][0] - data[seq - 1][-1][0]  # timestep t-1
#         vyt1 = (1. - data[seq][-1][1]) - (1. - data[seq - 1][-1][1])
#
#         vxt2 = data[seq - 1][-1][0] - data[seq - 2][-1][0]  # timestep t-2
#         vyt2 = (1. - data[seq - 1][-1][1]) - (1. - data[seq - 2][-1][1])
#
#         vxt3 = data[seq - 2][-1][0] - data[seq - 3][-1][0]  # timestep t-3
#         vyt3 = (1. - data[seq - 2][-1][1]) - (1. - data[seq - 3][-1][1])
#
#         vxt4 = data[seq - 3][-1][0] - data[seq - 4][-1][0]  # timestep t-4
#         vyt4 = (1. - data[seq - 3][-1][1]) - (1. - data[seq - 4][-1][1])
#
#         data_x_vel.append([vxt1, vyt1, vxt2, vyt2, vxt3, vyt3, vxt4, vyt4])
#
#         # interested in velocity  of only last object (added object)
#         vx = data[seq + 1][-1][0] - data[seq][-1][0]
#         vy = (1. - data[seq + 1][-1][1]) - (1. - data[seq][-1][1])
#
#         data_y.append([vx, vy])
#
#         seq += 1
#
# data_x_images = np.array(data_x_images)
# data_x_vel = np.array(data_x_vel).astype(np.float32)
# data_y = np.array(data_y).astype(np.float32)
#
# np.save('./data_in_float/data_x_images_2.npy', data_x_images)
# np.save('./data_in_float/data_x_vel_2.npy', data_x_vel)
# np.save('./data_in_float/data_y_2.npy', data_y)

data_y = np.load('./data_in_float/data_y_2.npy')

num_classes = 100

bin_vx = get_bin(data_y[:, 0], num_classes)
bin_vy = get_bin(data_y[:, 1], num_classes)

# ONE-HOT ENCODING DATA_Y
one_hot_vx = convert_to_one_hot_encoded_bin(data_y[:, 0], bin_vx, num_classes)
one_hot_vy = convert_to_one_hot_encoded_bin(data_y[:, 1], bin_vy, num_classes)

data_y = np.concatenate([one_hot_vx, one_hot_vy], axis=-1)
np.save('./data_in_float/one_hot-data_y_2.npy', data_y)
