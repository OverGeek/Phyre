import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import imageio
import os as os

# path to where the numpy arrays of generated dataset are stored
dataset_path = './Generated_dataset/Task-2'

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
                # plt.imshow(image, cmap='gray')
                # plt.show()

            elif typ == 'ball-red':
                draw = ImageDraw.Draw(image)
                x1 = cx - diam / 2.
                x2 = cx + diam / 2.
                y1 = cy - diam / 2.
                y2 = cy + diam / 2.

                draw.ellipse((x1, y1, x2, y2), fill='white')
                l = list(image.getdata())
                channel1 = np.array(l).reshape((IMAGE_WIDTH, IMAGE_HEIGHT))
                # plt.imshow(image, cmap='gray')
                # plt.show()

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
