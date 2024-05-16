import math
import tifffile
import numpy as np
import cv2
import cv2.optflow

def calculate_angle(x1, y1, x2, y2, x3, y3):
    v1 = (x1 - x2, y1 - y2)
    v2 = (x3 - x2, y3 - y2)

    mag_v1 = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    mag_v2 = math.sqrt((x3 - x2)**2 + (y3 - y2)**2)

    if mag_v1 == 0 or mag_v2 == 0:
        return 0  # Return 0 if either vector has zero magnitude

    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    cos_angle = dot_product / (mag_v1 * mag_v2)
    angle_rad = math.acos(cos_angle)
    angle_deg = math.degrees(angle_rad)

    if v1[0] * v2[1] - v1[1] * v2[0] < 0:
        angle_deg = 360 - angle_deg

    return angle_deg


def unwrap_scroll(image_path):
    # Load the input image
    image = tifffile.imread(image_path)

    # Replace 0 pixels with 1
    image[image == 0] = 1

    center = (image.shape[1] // 2, image.shape[0] // 2)

    h, w = image.shape
    ch, cw = center
    radius = int(math.sqrt((ch - 0) ** 2 + (cw - 0) ** 2))

    height = int(radius)
    width = int(radius * 2 * math.pi)
    unwrapped = np.zeros((height, width), dtype=np.uint8)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if i == ch and j == cw:
                continue
            y = int(math.sqrt((ch - i) ** 2 + (cw - j) ** 2))
            if y == 0:
                continue  # Skip the point if it coincides with the center
            angle = math.atan2(i - ch, j - cw)
            if angle < 0:
                angle += 2 * math.pi
            x = int(angle / (2 * math.pi) * width)
            unwrapped[height - y, x] = image[i, j]

    # Create a new array to store the shifted pixels
    shifted_unwrapped = np.zeros_like(unwrapped)

    # Shift pixels to the left to fill in black pixels and space left by other shifted pixels
    for y in range(unwrapped.shape[0]):
        row = unwrapped[y]
        non_zero_indices = np.nonzero(row)[0]
        if len(non_zero_indices) > 0:
            shift = non_zero_indices[0]
            data_pixels = row[non_zero_indices]
            shifted_unwrapped[y, :len(data_pixels)] = data_pixels

    cv2.imwrite('unwrapped.tif', shifted_unwrapped)

def snake():
    import cv2
    import numpy as np

    # Load and preprocess the image
    img = cv2.imread('01979.tif', cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Apply Gaussian blur for smoothing

    # Define the initial contour
    initial_contour = np.array([[[1711,1223]], [[1653, 1366]], ])

    # Set the optical flow parameters
    flow_params = {
        'alpha': 0.012,
        'beta': 0.75,
        'gamma': 0.5,
        'delta': 0.25,
        'sigma': 0.6,
        'min_size': 10,
        'num_inner_iter': 3,
        'num_iter': 4
    }

    # Create a mask for the initial contour
    mask = np.zeros_like(img)
    cv2.drawContours(mask, [initial_contour.astype(int)], 0, 255, 1)

    # Perform deep flow optical flow
    flow = 	cv2.optflow.createOptFlow_DeepFlow()

    contour = initial_contour.copy()
    for _ in range(flow_params['num_iter']):
        flow_field = flow.calc(img, img, None)
        contour = np.round(contour + flow_field[contour[:, 0, 1].astype(int), contour[:, 0, 0].astype(int)]).astype(np.int32)

    # Retrieve the final contour
    final_contour = contour.reshape(-1, 2)

    # Visualize the result
    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_color, [final_contour.astype(int)], 0, (0, 255, 0), 2)
    img_color = cv2.resize(img_color, (900,900))
    cv2.imshow('Papyrus Scroll with Contour', img_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    #unwrap_scroll('./dl.ash2txt.org/full-scrolls/PHerc1667.volpkg/volumes/20231117161658/01000.tif')
    #image = cv2.imread('./dl.ash2txt.org/full-scrolls/PHerc1667.volpkg/volumes/20231117161658/01000.tif')
    import cv2
    import numpy as np
    snake()
