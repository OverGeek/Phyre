import numpy as np


def convert_to_one_hot_encoded_bin(arr, bin, num_classes):
    digitized_arr = np.digitize(arr, bin) - 1
    one_hot_encoded_arr = np.zeros((digitized_arr.size, num_classes))
    one_hot_encoded_arr[np.arange(digitized_arr.size), digitized_arr] = 1
    return one_hot_encoded_arr


def get_bin(arr, num_bins):
    arr_sort_copy = list(arr)
    arr_sort_copy.sort()
    # print(arr_sort_copy)
    step = len(arr_sort_copy)//(num_bins-1)
    bins = []
    i = 0
    while i < len(arr_sort_copy):
        bins.append(arr_sort_copy[i])
        i += step

    bins.append(max(arr_sort_copy)+1e-4)

    return np.array(bins)


# test = np.random.rand(14328)
# test_bin = get_bin(test, 100)