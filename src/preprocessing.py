from PIL import Image
from scipy.optimize import minimize_scalar
import numpy as np
import cv2

#https://github.com/awangenh/fastaniso
def anisodiff3(stack,niter=1,kappa=50,gamma=0.1,step=(1.,1.,1.),option=1):
    """
    3D Anisotropic diffusion.

    Usage:
    stackout = anisodiff(stack, niter, kappa, gamma, option)

    Arguments:
            stack  - input stack
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (z,y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the middle z-plane will be plotted on every
                 iteration

    Returns:
            stackout   - diffused stack.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x,y and/or z axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """

    # initialize output array
    stack = stack.astype('float32')
    stackout = stack.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(stackout)
    deltaE = deltaS.copy()
    deltaD = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    UD = deltaS.copy()
    gS = np.ones_like(stackout)
    gE = gS.copy()
    gD = gS.copy()

    for ii in range(niter):

        # calculate the diffs
        deltaD[:-1,: ,:  ] = np.diff(stackout,axis=0)
        deltaS[:  ,:-1,: ] = np.diff(stackout,axis=1)
        deltaE[:  ,: ,:-1] = np.diff(stackout,axis=2)

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gD = np.exp(-(deltaD/kappa)**2.)/step[0]
            gS = np.exp(-(deltaS/kappa)**2.)/step[1]
            gE = np.exp(-(deltaE/kappa)**2.)/step[2]
        elif option == 2:
            gD = 1./(1.+(deltaD/kappa)**2.)/step[0]
            gS = 1./(1.+(deltaS/kappa)**2.)/step[1]
            gE = 1./(1.+(deltaE/kappa)**2.)/step[2]

        # update matrices
        D = gD*deltaD
        E = gE*deltaE
        S = gS*deltaS

        # subtract a copy that has been shifted 'Up/North/West' by one
        # pixel. don't as questions. just do it. trust me.
        UD[:] = D
        NS[:] = S
        EW[:] = E
        UD[1:,: ,: ] -= D[:-1,:  ,:  ]
        NS[: ,1:,: ] -= S[:  ,:-1,:  ]
        EW[: ,: ,1:] -= E[:  ,:  ,:-1]

        # update the image
        stackout += gamma*(UD+NS+EW)

    return stackout



#https://github.com/pengyan510/glcae/tree/master
#this version is modified for 3d grayscale

import numpy as np
from scipy.optimize import minimize_scalar
import cv2


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


# Example usage:
# voxel_data = ... # Your 3D uint8 voxel data
# enhanced_voxel_data = global_local_contrast_3d(voxel_data)




def clip(chunk):
    flat_arr = chunk.flatten()
    hist, bins = np.histogram(flat_arr, bins=256, range=(0, 256))
    cum_sum = np.cumsum(hist)
    total_pixels = flat_arr.shape[0]
    lower_idx = np.argmax(cum_sum >= 0.025 * total_pixels)
    upper_idx = np.argmax(cum_sum >= 0.975 * total_pixels)
    arr_capped = np.clip(chunk, bins[lower_idx], bins[upper_idx])
    arr_rescaled = ((arr_capped - bins[lower_idx]) * 255 / (bins[upper_idx] - bins[lower_idx])).astype(np.uint8)
    return arr_rescaled

def rescale(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def do_mask(combined_chunk, labels, superpixels, threshold):
    mask = np.zeros_like(combined_chunk, dtype=bool)

    for i, superpixel in enumerate(superpixels):
        if i == 0:
            continue  # Skip the background superpixel (label 0)

        if superpixel.c >= threshold:
            mask |= (labels == i)

    masked_chunk = np.zeros_like(combined_chunk)
    masked_chunk[mask] = combined_chunk[mask]
    return masked_chunk


def avg_pool_3d(input, pool_size, stride):
    input_depth, input_height, input_width = input.shape

    output = np.zeros_like(input)

    for d in range(0, input_depth, stride):
        for i in range(0, input_height, stride):
            for j in range(0, input_width, stride):
                start_d = d
                start_i = i
                start_j = j
                end_d = min(start_d + pool_size[0], input_depth)
                end_i = min(start_i + pool_size[1], input_height)
                end_j = min(start_j + pool_size[2], input_width)

                pool_region = input[start_d:end_d, start_i:end_i, start_j:end_j]
                output[start_d:end_d, start_i:end_i, start_j:end_j] = np.mean(pool_region)

    return output


def project_mask_to_volume(mask_path, voxel_volume, crop_start, brightness_adjust):
    Image.MAX_IMAGE_PIXELS = None

    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask) > 0

    cropped_mask = mask[crop_start[0]:crop_start[0]+voxel_volume.shape[0],
                         crop_start[1]:crop_start[1]+voxel_volume.shape[1]]

    mask_3d = np.repeat(cropped_mask[:, :, np.newaxis], voxel_volume.shape[2], axis=2)

    modified_volume = np.copy(voxel_volume)

    modified_volume[mask_3d] = np.minimum(modified_volume[mask_3d] + brightness_adjust, 255)

    return modified_volume