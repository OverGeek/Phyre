import numpy as np
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import  Dropout, Flatten, Dense, Input, concatenate
import keras
import tensorflow as tf
import os as os
from PIL import Image, ImageDraw
import imageio
import os as os

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
channels = 3
batch_size = 16

input_image = Input((IMAGE_WIDTH, IMAGE_HEIGHT, channels))
input_vel_1 = Input((10,))
input_vel_2 = Input((10,))

x = Conv2D(32, (3, 3), input_shape=(256, 256, 3), activation='relu')(input_image)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(32, (3, 3), input_shape=(256, 256, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Conv2D(1, (3, 3), input_shape=(256, 256, 3), activation='relu')(x)
x = MaxPooling2D(pool_size=(2, 2))(x)

x = Flatten()(x)

x = concatenate([x, input_vel_1, input_vel_2])

x = Dense(512, activation='relu')(x)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.4)(x)
vx = Dense(5, activation='softmax')(x)
vy = Dense(5, activation='softmax')(x)

output = concatenate([vx, vy])

model = Model(inputs=[input_image, input_vel_1, input_vel_2], outputs=output)


def loss_fun(y_true, y_pred):
    vx_true = y_true[:, :5]
    vy_true = y_true[:, 5:]
    vx_pred = y_pred[:, :5]
    vy_pred = y_pred[:, 5:]

    loss_x = keras.losses.categorical_crossentropy(vx_true, vx_pred)
    loss_y = keras.losses.categorical_crossentropy(vy_true, vy_pred)

    loss = tf.reduce_mean(tf.add(loss_x, loss_y), axis=0)

    return loss


model.compile(loss=loss_fun,
              optimizer='adam',
              metrics=['accuracy'])

model.load_weights('7-epochs.h5')

########################################################################################################################
dataset_path = './Generated_dataset/Task-2/'
test_subtask_no = str(99)

test_x_images = []
test_x_vel = []
pred = []

subtask = 'Task-2-'+test_subtask_no
subtask_folder = os.path.join(dataset_path, subtask)

task_no = subtask.split('-')[1]
subtask_no = subtask.split('-')[2]

file = 'task-0000' + str(task_no) + '_' + str(subtask_no) + '.npy'
data = np.load(os.path.join(dataset_path, file))
print(data.shape)

for seq in range(2, len(os.listdir(subtask_folder)) - 1):
    img_path = os.path.join(subtask_folder, str(seq) + '.png')
    image = Image.open(img_path)
    test_x_images.append(np.asarray(np.asarray(image) / 255, dtype=np.uint8))
    image.close()

    # VEL OF PREV 2 STEPS
    vx1 = data[seq-1][-1][0] - data[seq-2][-1][0]  # seq t-2
    vy1 = (1. - data[seq-1][-1][1]) - (1. - data[seq-2][-1][1])
    vx2 = data[seq][-1][0] - data[seq - 1][-1][0]  # seq t-1
    vy2 = (1. - data[seq][-1][1]) - (1. - data[seq - 1][-1][1])

    test_x_vel.append([vx1, vy1, vx2, vy2])

    seq += 1

test_x_images = np.array(test_x_images)
test_x_vel = np.array(test_x_vel)

data_x_vel = np.load('data_x_vel.npy')
data_y = np.load('data_y.npy')

max_vx = max(np.max(data_y[:, 0]), np.max(data_x_vel[:, 0]), np.max(data_x_vel[:, 2]))
min_vx = min(np.min(data_y[:, 0]), np.min(data_x_vel[:, 0]), np.min(data_x_vel[:, 2]))
max_vy = max(np.max(data_y[:, 1]), np.max(data_x_vel[:, 1]), np.max(data_x_vel[:, 3]))
min_vy = min(np.min(data_y[:, 1]), np.min(data_x_vel[:, 1]), np.min(data_x_vel[:, 3]))

step_vx = (max_vx - min_vx) / 4.
bin_vx = np.arange(min_vx, max_vx + 2 * step_vx, step_vx)
step_vy = (max_vy - min_vy) / 4.
bin_vy = np.arange(min_vy, max_vy + 2 * step_vy, step_vy)

digitized_vx1 = np.digitize(test_x_vel[:, 0], bin_vx) - 1  # 0-indexing
digitized_vy1 = np.digitize(test_x_vel[:, 1], bin_vy) - 1  # 0-indexing
digitized_vx2 = np.digitize(test_x_vel[:, 2], bin_vx) - 1  # 0-indexing
digitized_vy2 = np.digitize(test_x_vel[:, 3], bin_vy) - 1  # 0-indexing

one_hot_vx1 = np.zeros((digitized_vx1.size, 5))
one_hot_vx1[np.arange(digitized_vx1.size), digitized_vx1] = 1
one_hot_vy1 = np.zeros((digitized_vy1.size, 5))
one_hot_vy1[np.arange(digitized_vy1.size), digitized_vy1] = 1

one_hot_vx2 = np.zeros((digitized_vx2.size, 5))
one_hot_vx2[np.arange(digitized_vx2.size), digitized_vx2] = 1
one_hot_vy2 = np.zeros((digitized_vy2.size, 5))
one_hot_vy2[np.arange(digitized_vy2.size), digitized_vy2] = 1

test_x_vel = np.concatenate([one_hot_vx1, one_hot_vy1, one_hot_vx2, one_hot_vy2], axis=-1)
pred = model.predict([test_x_images, test_x_vel[:, :10], test_x_vel[:, 10:]])


# converting one-hot encoded velocities tp actual velocities
pred_x = pred[:, :5]
pred_y = pred[:, 5:]

pred_x = np.argmax(pred_x, axis=1)
pred_y = np.argmax(pred_y, axis=1)

pred_x = np.array([(bin_vx[x+1]-bin_vx[x])/2. for x in pred_x])
pred_y = np.array([(bin_vy[x+1]-bin_vy[x])/2. for x in pred_y])

########################################################################################################################
dataset_path = './Generated_dataset/Task-2'

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256

filename = file
counter = 0
print("Laded file: ", filename)
data = os.path.join(dataset_path, filename)
data = np.load(data)
print(data.shape)

for ix in range(2, data.shape[0] - 1):
    idx=0
    scene = data[ix]
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
            draw = ImageDraw.Draw(image)
            x1 = cx - diam / 2.
            x2 = cx + diam / 2.
            y1 = cy - diam / 2.
            y2 = cy + diam / 2.

            draw.ellipse((x1, y1, x2, y2), fill='white')
            l = list(image.getdata())
            channel1 = np.array(l).reshape((IMAGE_WIDTH, IMAGE_HEIGHT))

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
            # plt.imshow(image, cmap='gray')
            # plt.show()

        idx += 1

    #####################################################################################################################

    # Drawing next state ground truth
    # next_cx_gt = data[ix + 1][-1][0] * IMAGE_WIDTH
    # next_cy_gt = IMAGE_HEIGHT - data[ix + 1][-1][1] * IMAGE_HEIGHT
    # diam = data[ix + 1][-1][3] * IMAGE_WIDTH
    #
    # next_x1_gt = next_cx_gt - diam / 2
    # next_x2_gt = next_cx_gt + diam / 2
    # next_y1_gt = next_cy_gt - diam / 2
    # next_y2_gt = next_cy_gt + diam / 2
    #
    # im = Image.fromarray(channel1)
    # draw = ImageDraw.Draw(im)
    # draw.ellipse((next_x1_gt, next_y1_gt, next_x2_gt, next_y2_gt), fill=64)
    # l = list(im.getdata())
    # channel1 = np.array(l).reshape((IMAGE_WIDTH, IMAGE_HEIGHT))

    # Drawing next state prediction
    next_cx_pred = (data[ix][-1][0] + pred_x[ix-2]) * IMAGE_WIDTH
    next_cy_pred = (1 - data[ix][-1][1] + pred_y[ix-2]) * IMAGE_HEIGHT

    next_x1_pred = next_cx_pred - diam / 2
    next_x2_pred = next_cx_pred + diam / 2
    next_y1_pred = next_cy_pred - diam / 2
    next_y2_pred = next_cy_pred + diam / 2

    im = Image.fromarray(channel1)
    draw = ImageDraw.Draw(im)
    draw.ellipse((next_x1_pred, next_y1_pred, next_x2_pred, next_y2_pred), fill=128)
    l = list(im.getdata())
    channel1 = np.array(l).reshape((IMAGE_WIDTH, IMAGE_HEIGHT))

    #####################################################################################################################

    channel1 = np.expand_dims(channel1, -1)
    channel2 = np.expand_dims(channel2, -1)
    channel3 = np.expand_dims(channel3, -1)

    bitmap_image = np.concatenate([channel1, channel2, channel3], axis=-1).astype(np.uint8)
    subtask_no = filename.split('_')[1].split('.')[0]
    try:
        os.mkdir('./Predictions' + '/Task-' + filename[9] + '-' + subtask_no + '_prediction')
    except:
        pass

    imageio.imwrite(os.path.join('./Predictions' + '/Task-' + filename[9] + '-' + str(subtask_no)  + '_prediction', str(counter) + '.png'),
                    bitmap_image)
    counter += 1
