#
# File: libnoise.py
# Description: This file includes auxiliar functions to be used in the experiments performed for the papers.
# Author: Jose A. Rodriguez-Rodriguez
# Date: 21/09/2019
#

import numpy as np
import scipy.io
import cv2


# 8 - bits resolution RGB and monochrome
# generate_p_g_img - Generate Poisson + Gaussian Noise
# a = conversion gain
# b = sigma gaussian noise
def generate_p_g_img(img, a, b, seed):
    if seed > 0:
        np.random.seed(seed)
    img = img.astype(int)
    noisy_img = np.random.poisson((1 / a) * img, img.shape)
    noisy_img = a * noisy_img + np.random.normal(0, b, noisy_img.shape)
    noisy_img[noisy_img > 255] = 255
    noisy_img[noisy_img < 0] = 0
    return noisy_img


# generate_p_img - Generate Poisson Noise
# a = conversion gain
def generate_p_img(img, a, seed):
    if seed > 0:
        np.random.seed(seed)
    img = img.astype(int)
    noisy_img = np.random.poisson((1 / a) * img, img.shape)
    noisy_img = a * noisy_img
    noisy_img[noisy_img > 255] = 255
    noisy_img[noisy_img < 0] = 0
    return noisy_img


# generate_g_img - Generate Gaussian Noise
def generate_g_img(img, b, seed):
    if seed > 0:
        np.random.seed(seed)
    img = img.astype(int)
    noisy_img = img + np.random.normal(0, b, img.shape)
    noisy_img[noisy_img > 255] = 255
    noisy_img[noisy_img < 0] = 0
    return noisy_img


# Histogram Equalization
def hist_equal(img, flag):
    img = img.astype(np.uint8)

    if flag:
        img_gbr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_yuv = cv2.cvtColor(img_gbr, cv2.COLOR_BGR2YUV)
        img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
        img_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    else:
        img_rgb = img

    return img_rgb


# Get word mapping for Imagenet2012 Challenge
def get_word_map(filename):
    meta_mat = scipy.io.loadmat(filename, struct_as_record=False)
    synsets = np.squeeze(meta_mat['synsets'])
    # ids = np.squeeze(np.array([s.ILSVRC2012_ID for s in synsets]))
    wnids = np.squeeze(np.array([s.WNID for s in synsets]))
    words = np.squeeze(np.array([s.words for s in synsets]))
    num_children = np.squeeze(np.array([s.num_children for s in synsets]))
    index_children0 = np.nonzero(num_children == 0)
    wnids0 = wnids
    wnids = wnids[index_children0]
    words = words[index_children0]
    index_reorder = np.argsort(wnids)
    wnids.sort()
    words = words[index_reorder]
    return words, index_reorder


# Convert image mapping
def convert_word_map(y, index_reorder):
    yc = np.zeros(y.shape[0], dtype=int)
    for i in range(0, y.shape[0]):
        aux = np.nonzero(index_reorder == ((np.asscalar(np.squeeze(y[i]))) - 1))
        yc[i] = np.asscalar(np.squeeze(aux)) + 1
    return yc


# Convert integer to one-hot
def int2onehot(y, num_outputs):
    ysize = y.shape[0];
    yc = np.zeros((num_outputs, ysize), dtype=int)

    for i in range(0, ysize):
        index = y[i] - 1
        yc[index, i] = 1
    return yc


# Create Grid Meshing for Frequency Domain filters
# param: M = Number of rows
#        N = Number of columns
def meshuv(M, N):
    u = np.arange(0, N)
    v = np.arange(0, M)
    v = np.flip(v)
    idx = np.argwhere(u > N / 2)
    u[idx] = u[idx] - N
    idy = np.argwhere(v > M / 2)
    v[idy] = v[idy] - M

    [U, V] = np.meshgrid(u, v)

    return U, V


# Filter in frequency domain
# param: img = image in space domain without padding (uint8)
#        H = Filter un frequency domain extended
def fftfilt(img, H):
    F = np.fft.fft2(img, [H.shape[0], H.shape[1]])
    g = np.real(np.fft.ifft2(H * F))
    g_filt = g[0:img.shape[0], 0:img.shape[1]]

    return g_filt


# Get a and b parameters from theta and Length for Motion Blur filter
# param: vector = [[length1, theta1],[length2, theta2],...,[lengthn, thetan]]
def get_mb_ab_params(vector):

    vector_shape = np.shape(vector)

    if len(vector_shape) == 1:
        theta_rad = np.deg2rad(vector[1])
        a = vector[0] / np.sqrt(1 + (np.tan(theta_rad)) ** 2)

        if 90 < vector[1] < 270:
            a = -a

        b = a * np.tan(theta_rad)
        if vector[1] == 270 and b > 0:
            b = -b

        vector_out = [a, b]
    else:
        vector_out = np.zeros((2, 2), dtype=float)

        for i in range(0, 2):
            theta_rad = np.deg2rad(vector[i][1])
            vector_out[i][0] = vector[i][0] / np.sqrt(1 + (np.tan(theta_rad)) ** 2)

            if 90 < vector[i][1] < 270:
                vector_out[i][0] = -vector_out[i][0]

            vector_out[i][1] = vector_out[i][0] * np.tan(theta_rad)
            if vector[i][1] == 270 and vector_out[i][1] > 0:
                vector_out[i][1] = -vector_out[i][1]

    vector_out = np.round(vector_out, 5)

    return vector_out


# Motion blur filter
# param: M: number of rows (image)
#        N: number of columns (image)
#        T: relative exposure time
#        vector: [[a1, b1],[a2, b2],...,[an, bn]]
def get_mb_filter(M, N, T, vector):
    # U, V mesh grid
    U, V = meshuv(M, N)

    # Check vector size
    vector_shape = np.shape(vector)
    if len(vector_shape) == 1:
        a = vector[0]
        b = vector[1]
        H = T * np.sinc(U * a + V * b) * np.exp(complex(0, -1) * np.pi * (U * a + V * b))
    else:
        # TODO: generalize filter size - 1 and 2 sizes allowed at the moment
        a1 = vector[0][0]
        b1 = vector[0][1]
        H = (T / 2) * (np.sinc(U * a1 + V * b1)) * np.exp(complex(0, -1) * np.pi * (U * a1 + V * b1))
        a2 = vector[1][0]
        b2 = vector[1][1]
        H_aux = (T / 2) * np.exp(complex(0, -1) * np.pi * (2 * (U * (a1 - a2) + V * (b1 - b2)) +
                                 (3 * (U * a2 + V * b2)))) * np.sinc((U * a2 + V * b2))
        H = H + H_aux

    return H


# Generate Motion Blur
# param: img = image in space domain without padding (uint8)
#        T = relative exposure time
#        vector = [[length1, theta1],[length2, theta2],...,[lengthn, thetan]]
def get_mb_image(img, T, vector):
    size = np.shape(img)

    PQ = [2 * img.shape[0], 2 * img.shape[1]]

    vector_filter = get_mb_ab_params(vector)

    H = get_mb_filter(PQ[0], PQ[1], T, vector_filter)

    # RGB image
    if len(size) == 3:
        g = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype='double')
        for i in range(0, img.shape[2]):
            g[:, :, i] = fftfilt(img[:, :, i], H)
    # Grayscale image
    else:
        g = np.zeros((img.shape[0], img.shape[1]), dtype='double')
        g = fftfilt(img, H)

    return g, H
