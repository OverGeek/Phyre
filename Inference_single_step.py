import numpy as np
from keras.models import load_model
import os as os
from PIL import Image, ImageDraw
import imageio
from utils import get_bin
from loss_functions import loss_fun_task2

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

########################################################################################################################
model = load_model('Model_Task2', custom_objects={'loss_fun_task2': loss_fun_task2})
model.load_weights('./10-epochs.h5')
########################################################################################################################

data_x_vel = np.load('data_x_vel.npy')
data_y = np.load('data_y.npy')

num_classes = 100

bin_vx = get_bin(data_y[:, 0], num_classes)
bin_vy = get_bin(data_y[:, 1], num_classes)

dataset_path = './Generated_dataset/Task-2/'
test_subtask_no = str(98)

subtask = 'Task-2-' + test_subtask_no
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

# VEL OF PREV 4 TIMESTEPS
vxt1 = data[seq][-1][0] - data[seq - 1][-1][0]  # timestep t-1
vyt1 = (1. - data[seq][-1][1]) - (1. - data[seq - 1][-1][1])

vxt2 = data[seq - 1][-1][0] - data[seq - 2][-1][0]  # timestep t-2
vyt2 = (1. - data[seq - 1][-1][1]) - (1. - data[seq - 2][-1][1])

vxt3 = data[seq - 2][-1][0] - data[seq - 3][-1][0]  # timestep t-3
vyt3 = (1. - data[seq - 2][-1][1]) - (1. - data[seq - 3][-1][1])

vxt4 = data[seq - 3][-1][0] - data[seq - 4][-1][0]  # timestep t-4
vyt4 = (1. - data[seq - 3][-1][1]) - (1. - data[seq - 4][-1][1])

# INITIAL POSITION OF RED BALL
posx = data[seq][-1][0] * IMAGE_WIDTH
posy = (1 - data[seq][-1][1]) * IMAGE_HEIGHT

while seq <= data.shape[0] - 1:

    test_x_vel = np.array([[vxt1, vyt1, vxt2, vyt2, vxt3, vyt3, vxt4, vyt4]])

    pred = model.predict([test_x_images, test_x_vel])

    # converting one-hot encoded velocities tp actual velocities
    pred_x = pred[:, :num_classes]
    pred_y = pred[:, num_classes:]

    pred_x = np.argmax(pred_x, axis=1)
    pred_y = np.argmax(pred_y, axis=1)

    pred_x = np.array([(bin_vx[i + 1] + bin_vx[i]) / 2. for i in pred_x])
    pred_y = np.array([(bin_vy[i + 1] + bin_vy[i]) / 2. for i in pred_y])

    print(pred_x[0], pred_y[0])

    vxt4 = vxt3
    vyt4 = vyt3
    vxt3 = vxt2
    vyt3 = vyt2
    vxt2 = vxt1
    vyt2 = vyt1
    vxt1 = pred_x[0]
    vyt1 = pred_y[0]

    scene = data[seq]
    idx = 0
    channel1 = None
    channel2 = None
    channel3 = None
    for obj in scene:
        cx = obj[0] * IMAGE_WIDTH
        cy = IMAGE_HEIGHT - (obj[1] * IMAGE_WIDTH)
        diam = obj[3] * IMAGE_WIDTH

        typ = None
        if obj[4] == 1 and idx == 1:
            typ = 'ball-green'
        elif obj[4] == 1 and idx == 3:
            typ = 'ball-red'
        elif obj[5] == 1:
            typ = 'bar'

        image = Image.new('1', (IMAGE_WIDTH, IMAGE_HEIGHT))
        if typ == 'ball-green':
            draw = ImageDraw.Draw(image)
            x1 = cx - diam / 2.
            x2 = cx + diam / 2.
            y1 = cy - diam / 2.
            y2 = cy + diam / 2.

            draw.ellipse((x1, y1, x2, y2), fill='white')
            l = list(image.getdata())
            channel3 = np.array(l).reshape((IMAGE_WIDTH, IMAGE_HEIGHT))

        elif typ == 'ball-red':
            pass

        elif typ == 'bar':
            if channel2 is None:
                draw = ImageDraw.Draw(image)
            else:
                im = Image.fromarray(channel2)
                draw = ImageDraw.Draw(im)
            x1 = cx - diam / 2.
            x2 = cx + diam / 2.
            y1 = cy - 5 / 2.
            y2 = cy + 5 / 2.

            draw.rectangle((x1, y1, x2, y2), fill='white')
            if channel2 is None:
                l = list(image.getdata())
            else:
                l = list(im.getdata())

            channel2 = np.array(l).reshape((IMAGE_WIDTH, IMAGE_HEIGHT))

        idx += 1

    filename = 'task-0000' + str(task_no) + '_' + test_subtask_no + '.npy'

    # Drawing next state prediction
    next_cx_pred = posx + pred_x[0] * IMAGE_WIDTH
    next_cy_pred = posy + pred_y[0] * IMAGE_HEIGHT

    posx = next_cx_pred
    posy = next_cy_pred

    next_x1_pred = next_cx_pred - diam / 2
    next_x2_pred = next_cx_pred + diam / 2
    next_y1_pred = next_cy_pred - diam / 2
    next_y2_pred = next_cy_pred + diam / 2

    image = Image.new('1', (IMAGE_WIDTH, IMAGE_HEIGHT))
    draw = ImageDraw.Draw(image)
    draw.ellipse((next_x1_pred, next_y1_pred, next_x2_pred, next_y2_pred), fill='white')
    l = list(image.getdata())
    channel1 = np.array(l).reshape((IMAGE_WIDTH, IMAGE_HEIGHT))

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
                     str(seq) + '.png'),
        bitmap_image)

    # bitmap_image = bitmap_image.astype(np.float32)
    bitmap_image = bitmap_image // 255
    image_t0 = bitmap_image
    image_t1 = test_x_images[0][:, :, :3]

    channel_1_t1 = np.expand_dims(image_t1[:, :, 0], axis=-1)
    channel_3_t1 = np.expand_dims(image_t1[:, :, 2], axis=-1)
    input_image = np.concatenate([image_t0, channel_1_t1, channel_3_t1], axis=-1)

    test_x_images = np.array([input_image])

    seq += 1
