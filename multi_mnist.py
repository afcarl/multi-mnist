# Code to combine MNIST SIZExSIZE images into images containing MIN to MAX MNIST images within
# a single IMAGE_LENGTHxSIZE image. Need to create label for each image.
# If possible, keep same format as original MNIST data.

# Initial MNIST to numpy code adapted from Gustav Sweyla (http://g.sweyla.com/blog/2012/mnist-numpy/)

import os
import struct
import numpy as np
import load
import random
from numpy import array, int8, uint8, zeros
from scipy.misc import imresize, imsave
import pickle

MIN = 2
MAX = 4
#size of frame
SIZE = 64
#border
BORDER = 2
#downsample
ISIZE = 20
#n. of digits
ND = 2
MNIST_SAMPLES = 60000
DATA_PATH = "./mnist/data"
train_x, valid_x, test_x, train_y, valid_y, test_y = load.load_mnist(DATA_PATH)


def create_rand_multi_mnist(data, labels, samples=60000):
    """ Create a dataset where multiple (MIN to MAX) random MNIST digits are randomly located in a long image. """
    new_images = []
    new_labels = []
    img_by_labels = {}
    for i, (x, l) in enumerate(zip(data, labels)):
        img_by_labels[l] = img_by_labels.get(l, []) + [i]
    while len(new_images) != samples:
        if len(new_images) % 1000 == 0:
            print('done {}'.format(len(new_images)))
        pos = []
        mask = np.zeros((SIZE, SIZE))
        while len(pos) != ND:
            # try to find a non-overlapping free position
            valid = False
            n_trials = 0
            (rows, cols) = np.where(mask == 0.)
            coords = list(zip(*[rows, cols]))
            while not valid and n_trials < 5:
                start_x, start_y = coords[np.random.randint(len(coords))]
                m = mask[start_x + BORDER:start_x + ISIZE + BORDER,
                         start_y + BORDER:start_y + ISIZE + BORDER]
                if np.sum(m) > 0. or np.prod(m.shape) < (ISIZE * ISIZE):
                    valid = False
                    n_trials += 1
                else:
                    pos.append((start_x + BORDER, start_y + BORDER))
                    mask[start_x:start_x + ISIZE,start_y:start_y + ISIZE] = 1.
                    valid = True

        new_image = np.zeros([SIZE, SIZE])
        classes = np.random.permutation(range(10))
        for i, (start_x, start_y) in enumerate(pos):
            cdata = img_by_labels[classes[i]]
            idx = np.random.randint(0, len(cdata))
            img = data[cdata[idx]].reshape((28, 28))
            img = imresize(img, (ISIZE, ISIZE))
            new_image[start_x:start_x+ISIZE,start_y:start_y+ISIZE] = img
        new_labels.append(np.asarray(classes[:ND]))
        new_images.append(np.ravel(new_image))
    new_images = np.asarray(new_images)
    new_labels = np.asarray(new_labels)
    return new_images, new_labels

train_x = np.concatenate([train_x, valid_x], 0)
train_y = np.concatenate([train_y, valid_y], 0)

# test
train_img, train_lbl = create_rand_multi_mnist(train_x, train_y, samples=10)
imsave('img.png', train_img[:10].reshape((10 * SIZE, SIZE)))
#
train_img, train_lbl = create_rand_multi_mnist(train_x, train_y, samples=60000)
test_img, test_lbl = create_rand_multi_mnist(test_x, test_y, samples=10000)

pickle.dump({
    'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y,
    'multi_train_x': train_img, 'multi_train_y': train_lbl,
    'multi_test_x': test_img, 'multi_test_y': test_lbl},
            open('multi_mnist.pkl', 'w'))
