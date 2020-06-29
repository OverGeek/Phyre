import numpy as np
import os as os
from PIL import Image, ImageDraw
import imageio
import os as os
from utils import get_bin

IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
channels = 3
batch_size = 16

data_x_vel = np.load('data_x_vel.npy')
data_y = np.load('data_y.npy')

num_classes = 100

bin_vx = get_bin(data_y[:, 0], num_classes)
bin_vy = get_bin(data_y[:, 1], num_classes)


def convert_to_one_hot(test_x_vel):
    test_x_vel = np.array(test_x_vel)
    digitized_vx1 = np.digitize(test_x_vel[:, 0], bin_vx) - 1  # 0-indexing
    digitized_vy1 = np.digitize(test_x_vel[:, 1], bin_vy) - 1  # 0-indexing
    digitized_vx2 = np.digitize(test_x_vel[:, 2], bin_vx) - 1  # 0-indexing
    digitized_vy2 = np.digitize(test_x_vel[:, 3], bin_vy) - 1  # 0-indexing

    one_hot_vx1 = np.zeros((digitized_vx1.size, num_classes))
    one_hot_vx1[np.arange(digitized_vx1.size), digitized_vx1] = 1
    one_hot_vy1 = np.zeros((digitized_vy1.size, num_classes))
    one_hot_vy1[np.arange(digitized_vy1.size), digitized_vy1] = 1

    one_hot_vx2 = np.zeros((digitized_vx2.size, num_classes))
    one_hot_vx2[np.arange(digitized_vx2.size), digitized_vx2] = 1
    one_hot_vy2 = np.zeros((digitized_vy2.size, num_classes))
    one_hot_vy2[np.arange(digitized_vy2.size), digitized_vy2] = 1

    test_x_vel = np.concatenate([one_hot_vx1, one_hot_vy1, one_hot_vx2, one_hot_vy2], axis=-1)

    return test_x_vel


dataset_path = './Generated_dataset/Task-2/'
test_subtask_no = str(1)

subtask = 'Task-2-' + test_subtask_no
subtask_folder = os.path.join(dataset_path, subtask)

task_no = subtask.split('-')[1]
subtask_no = subtask.split('-')[2]

file = 'task-0000' + str(task_no) + '_' + str(subtask_no) + '.npy'
data = np.load(os.path.join(dataset_path, file))
print(data.shape)

#################################################### STARTING HERE #####################################################
seq = 2

test_x_images = []
test_x_vel = []
pred = []

img_path = os.path.join(subtask_folder, str(seq) + '.png')
image = Image.open(img_path)
test_x_images.append(np.asarray(np.asarray(image) / 255, dtype=np.uint8))
image.close()
test_x_images = np.array(test_x_images)


posx = data[seq][-1][0] * IMAGE_WIDTH
posy = (1 - data[seq][-1][1]) * IMAGE_HEIGHT

one_hot_encoded_data_y = np.load('one_hot-data_y.npy')

while seq <= data.shape[0] - 1:

    # digitized_vx = np.digitize(data_y[688+seq-2, 0], bin_vx) - 1  # 0-indexing
    # digitized_vy = np.digitize(data_y[688+seq-2, 1], bin_vy) - 1  # 0-indexing
    #
    # one_hot_vx = np.zeros((digitized_vx.size, 5))
    # one_hot_vx[np.arange(digitized_vx.size), digitized_vx] = 1
    # one_hot_vy = np.zeros((digitized_vy.size, 5))
    # one_hot_vy[np.arange(digitized_vy.size), digitized_vy] = 1
    #
    # pred = np.concatenate([one_hot_vx, one_hot_vy], axis=-1)

    # converting one-hot encoded velocities tp actual velocities
    pred_x = one_hot_encoded_data_y[seq-2, :num_classes]
    pred_y = one_hot_encoded_data_y[seq-2, num_classes:]

    pred_x = np.argmax(pred_x, axis=-1)
    pred_y = np.argmax(pred_y, axis=-1)

    pred_x = np.array([(bin_vx[pred_x + 1] + bin_vx[pred_x]) / 2.])
    pred_y = np.array([(bin_vy[pred_y + 1] + bin_vy[pred_y]) / 2.])

    # pred_x = [data_y[seq-2][0]]
    # pred_y = [data_y[seq-2][1]]

    IMAGE_WIDTH = 256
    IMAGE_HEIGHT = 256

    scene = data[seq]
    idx = 0
    channel1 = None
    channel2 = None
    channel3 = None
    for obj in scene:
        cx = obj[0] * IMAGE_WIDTH
        cy = IMAGE_HEIGHT - (obj[1] * IMAGE_HEIGHT)
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

    #####################################################################################################################

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
                     str(seq) + '.png'),
        bitmap_image)

    test_x_images = [bitmap_image]

    seq += 1
