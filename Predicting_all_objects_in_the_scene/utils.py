import numpy as np
from math import sin, cos, pi
from PIL import Image, ImageDraw

IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


def makeRectangle(l, w, theta, offset=(0, 0)):
    theta = 2 * pi - theta
    c, s = cos(theta), sin(theta)
    rectCoords = [(l / 2.0, w / 2.0), (l / 2.0, -w / 2.0), (-l / 2.0, -w / 2.0), (-l / 2.0, w / 2.0)]
    return [(c * x - s * y + offset[0], s * x + c * y + offset[1]) for (x, y) in rectCoords]


def convert_to_one_hot_encoded_bin(arr, bin, num_classes):
    digitized_arr = np.digitize(arr, bin) - 1
    one_hot_encoded_arr = np.zeros((digitized_arr.size, num_classes))
    one_hot_encoded_arr[np.arange(digitized_arr.size), digitized_arr] = 1
    return one_hot_encoded_arr


def get_bin(arr, num_bins):
    arr_sort_copy = list(arr)
    arr_sort_copy.sort()
    # print(arr_sort_copy)
    step = len(arr_sort_copy)// num_bins
    bins = []
    i = 0
    while i < len(arr_sort_copy) - step:
        bins.append(arr_sort_copy[i])
        i += step

    bins.append(max(arr_sort_copy) + 1e-4)

    return np.array(bins)


def draw_ball(channel, cx, cy, diam):
    image = Image.new('1', (IMAGE_WIDTH, IMAGE_HEIGHT))
    if channel is None:
        draw = ImageDraw.Draw(image)
    else:
        im = Image.fromarray(channel)
        draw = ImageDraw.Draw(im)

    x1 = cx - diam / 2.
    x2 = cx + diam / 2.
    y1 = cy - diam / 2.
    y2 = cy + diam / 2.

    draw.ellipse((x1, y1, x2, y2), fill='white')
    if channel is None:
        l = list(image.getdata())
    else:
        l = list(im.getdata())

    channel = np.array(l).reshape((IMAGE_WIDTH, IMAGE_HEIGHT))
    return channel


def draw_bar(channel, cx, cy, length, width, theta):
    image = Image.new('1', (IMAGE_WIDTH, IMAGE_HEIGHT))
    if channel is None:
        draw = ImageDraw.Draw(image)
    else:
        im = Image.fromarray(channel)
        draw = ImageDraw.Draw(im)

    vertices = makeRectangle(length, width, theta, offset=(cx, cy))
    draw.polygon(vertices, fill='white')
    if channel is None:
        l = list(image.getdata())
    else:
        l = list(im.getdata())

    channel = np.array(l).reshape((IMAGE_WIDTH, IMAGE_HEIGHT))
    return channel


def delta_pos(data, seq, obj_index, feature_index, rev=False):
    if not rev:
        return data[seq + 1][obj_index][feature_index] - data[seq][obj_index][feature_index]
    else:
        return (1. - data[seq + 1][obj_index][feature_index]) - (1. - data[seq][obj_index][feature_index])


def delta_theta(data, seq, obj_index, feature_index):
    t1 = data[seq + 1][obj_index][feature_index]
    t0 = data[seq][obj_index][feature_index]

    return t1 - t0


def get_all_bins(data_y, num_classes):
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

    return vx_bins, vy_bins, theta_bins


test = np.zeros((6027,))
test_bin = get_bin(test, 100)
# digitized_arr = np.digitize(test, test_bin) - 1
