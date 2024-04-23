from scipy.optimize import minimize_scalar
import numpy as np
import cv2
from PIL import Image
from scipy.optimize import minimize_scalar
import numpy as np
import cv2
import os


#https://github.com/pengyan510/glcae/tree/master
#this version is modified for 3d grayscale


def glcae_grayscale(path, name):
    def linearStretching(x_c, x_max, x_min, l):
        return (l - 1) * (x_c - x_min) / (x_max - x_min)

    def mapping(h, l):
        cum_sum = 0
        t = np.zeros_like(h, dtype=int)
        for i in range(l):
            cum_sum += h[i]
            t[i] = np.ceil((l - 1) * cum_sum + 0.5)

        return t

    def f(lam, h_i, h_u, l):
        h_tilde = 1 / (1 + lam) * h_i + lam / (1 + lam) * h_u
        t = mapping(h_tilde, l)
        d = 0
        for i in range(l):
            for j in range(i + 1):
                if h_tilde[i] > 0 and h_tilde[j] > 0 and t[i] == t[j]:
                    d = max(d, i - j)

        return d

    def fusion(i):
        lap = cv2.Laplacian(i.astype(np.uint8), cv2.CV_16S, ksize=3)
        c_d = np.array(cv2.convertScaleAbs(lap))
        c_d = c_d / np.max(np.max(c_d)) + 0.00001
        i_scaled = (i - np.min(np.min(i))) / (np.max(np.max(i)) - np.min(np.min(i)))
        b_d = np.apply_along_axis(lambda x: np.exp(- (x - 0.5) ** 2 / (2 * 0.2 ** 2)), 0, i_scaled.flatten()).reshape(
            i.shape)
        w_d = np.minimum(c_d, b_d)

        return w_d

    x = np.array(Image.open(path).convert('L')).astype(np.float64)
    x_max = np.max(np.max(x))
    x_min = np.min(np.min(x))

    l = 256
    x_hat = linearStretching(x, x_max, x_min, l)
    i = x_hat.astype(np.uint8)

    h_i = np.bincount(i.flatten())
    h_i = np.concatenate((h_i, np.zeros(l - h_i.shape[0]))) / (i.shape[0] * i.shape[1])
    h_u = np.ones_like(h_i) * 1 / l

    result = minimize_scalar(f, method="brent", args=(h_i, h_u, l))
    h_tilde = 1 / (1 + result.x) * h_i + result.x / (1 + result.x) * h_u
    t = mapping(h_tilde, l)
    g_i = np.apply_along_axis(lambda x: t[x], 0, i.flatten()).reshape(i.shape)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_i = clahe.apply(i)

    w_g = fusion(g_i)
    w_l = fusion(l_i)
    w_hat_g = w_g / (w_g + w_l)
    w_hat_l = w_l / (w_g + w_l)
    y = w_hat_g * g_i + w_hat_l * l_i
    y = y.astype(np.uint8)

    img = Image.fromarray(y)
    img.save(name + '-en.png')


def global_local_contrast_3d(x):
    def linearStretching(x_c, x_max, x_min, l):
        return (l - 1) * (x_c - x_min) / (x_max - x_min)

    def mapping(h, l):
        cum_sum = 0
        t = np.zeros_like(h, dtype=int)
        for i in range(l):
            cum_sum += h[i]
            t[i] = np.ceil((l - 1) * cum_sum + 0.5)
        return t

    def f(lam, h_i, h_u, l):
        h_tilde = 1 / (1 + lam) * h_i + lam / (1 + lam) * h_u
        t = mapping(h_tilde, l)
        d = 0
        for i in range(l):
            for j in range(i + 1):
                if h_tilde[i] > 0 and h_tilde[j] > 0 and t[i] == t[j]:
                    d = max(d, i - j)
        return d

    def huePreservation(g_i, i, x_hat, l):
        g_i_f = g_i.flatten()
        i_f = i.flatten()
        x_hat_f = x_hat.flatten()
        g = np.zeros(g_i_f.shape)
        g[g_i_f <= i_f] = (g_i_f / (i_f + 1e-8) * x_hat_f)[g_i_f <= i_f]
        g[g_i_f > i_f] = ((l - 1 - g_i_f) / (l - 1 - i_f + 1e-8) * (x_hat_f - i_f) + g_i_f)[g_i_f > i_f]
        return g.reshape(i.shape)

    def fusion(i):
        lap = cv2.Laplacian(i.astype(np.uint8), cv2.CV_16S, ksize=3)
        c_d = np.array(cv2.convertScaleAbs(lap))
        c_d = c_d / np.max(np.max(c_d)) + 0.00001
        i_scaled = (i - np.min(np.min(i))) / (np.max(np.max(i)) - np.min(np.min(i)))
        b_d = np.apply_along_axis(lambda x: np.exp(- (x - 0.5) ** 2 / (2 * 0.2 ** 2)), 0, i_scaled.flatten()).reshape(
            i.shape)
        w_d = np.minimum(c_d, b_d)
        return w_d

    x_max = np.max(x)
    x_min = np.min(x)

    l = 256
    x_hat = linearStretching(x, x_max, x_min, l)
    i = x_hat.astype(np.uint8)

    h_i = np.bincount(i.flatten())
    h_i = np.concatenate((h_i, np.zeros(l - h_i.shape[0]))) / (i.size)
    h_u = np.ones_like(h_i) * 1 / l

    result = minimize_scalar(f, method="brent", args=(h_i, h_u, l))
    h_tilde = 1 / (1 + result.x) * h_i + result.x / (1 + result.x) * h_u
    t = mapping(h_tilde, l)
    g_i = np.apply_along_axis(lambda x: t[x], 0, i.flatten()).reshape(i.shape)

    g = huePreservation(g_i, i, x_hat, l)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_i = np.zeros_like(i)
    for slice_idx in range(i.shape[0]):
        l_i[slice_idx] = clahe.apply(i[slice_idx])
    l = huePreservation(l_i, i, x_hat, l)

    w_g = fusion(g_i)
    w_l = fusion(l_i)
    w_hat_g = w_g / (w_g + w_l)
    w_hat_l = w_l / (w_g + w_l)
    y = w_hat_g * g + w_hat_l * l
    return y.astype(np.uint8)

if __name__ == '__main__':
    glcae_grayscale(r'02162.png','new')