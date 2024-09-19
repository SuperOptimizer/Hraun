import numpy as np
from scipy.optimize import minimize_scalar
import cv2

from common import timing_decorator

#https://github.com/pengyan510/glcae/tree/master
#this version is modified for 3d grayscale
def linearStretching(x_c, x_max, x_min, l):
    return (l - 1) * (x_c - x_min) / (x_max - x_min)


def mapping(h, l):
    cum_sum = np.cumsum(h)
    t = np.ceil((l - 1) * cum_sum + 0.5).astype(np.int64)
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


def huePreservation(g_i, i, x_hat_c, l):
    g_i_f = g_i.flatten()
    i_f = i.flatten()
    x_hat_c_f = x_hat_c.flatten()
    g_c = np.zeros(g_i_f.shape)
    g_c[g_i_f <= i_f] = (g_i_f / (i_f + 1e-8) * x_hat_c_f)[g_i_f <= i_f]
    g_c[g_i_f > i_f] = ((l - 1 - g_i_f) / (l - 1 - i_f + 1e-8) * (x_hat_c_f - i_f) + g_i_f)[g_i_f > i_f]
    return g_c.reshape(i.shape)


def fusion(i):
    lap = cv2.Laplacian(i.astype(np.uint8), cv2.CV_16S, ksize=3)
    c_d = np.array(cv2.convertScaleAbs(lap))
    c_d = c_d / (np.max(c_d) + 1e-8) + 0.00001
    i_scaled = (i - np.min(i)) / (np.max(i) - np.min(i) + 1e-8)
    b_d = np.exp(- (i_scaled - 0.5) ** 2 / (2 * 0.2 ** 2))
    w_d = np.minimum(c_d, b_d)
    return w_d

@timing_decorator
def global_local_contrast_3d(x):
    # Convert input to float64
    x = x.astype(np.float64)
    x_max = np.max(x)
    x_min = np.min(x)

    l = 256
    x_hat = (l - 1) * (x - x_min) / (x_max - x_min + 1e-8)
    i = np.clip(x_hat, 0, 255).astype(np.uint8)

    h_i = np.bincount(i.flatten(), minlength=l) / i.size
    h_u = np.ones(l) / l

    result = minimize_scalar(f, method="brent", args=(h_i, h_u, l))
    h_tilde = 1 / (1 + result.x) * h_i + result.x / (1 + result.x) * h_u
    t = mapping(h_tilde, l)
    g_i = np.take(t, i)

    g = huePreservation(g_i, i, x_hat, l)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_i = np.zeros_like(i)
    for slice_idx in range(i.shape[0]):
        l_i[slice_idx] = clahe.apply(i[slice_idx])
    l = huePreservation(l_i, i, x_hat, l)

    w_g = fusion(g_i)
    w_l = fusion(l_i)
    w_hat_g = w_g / (w_g + w_l + 1e-8)
    w_hat_l = w_l / (w_g + w_l + 1e-8)
    y = w_hat_g * g + w_hat_l * l

    # Handle NaN and infinity values, then rescale back to uint8
    y = np.nan_to_num(y, nan=0, posinf=255, neginf=0)
    y = np.clip(y, 0, 255).astype(np.uint8)


    return y


