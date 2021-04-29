import cv2
import numpy as np
import math


def pyramid(image):
    image = np.array(image, dtype=np.float32)
    lps = []
    ds = []
    dhats = []
    fs = []
    k = 4

    for i in range(k):
        sigma_c = 1.0  # 0.1?
        sigma_s = math.pow(2, i + 2)

        if i == k - 1:
            r = cv2.GaussianBlur(image[:, :, 0], (0, 0), sigma_s)
            g = cv2.GaussianBlur(image[:, :, 1], (0, 0), sigma_s)
            b = cv2.GaussianBlur(image[:, :, 2], (0, 0), sigma_s)
        else:
            r = cv2.ximgproc.dtFilter(image[:, :, 0], image[:, :, 0], sigma_s, sigma_c, numIters=5)
            g = cv2.ximgproc.dtFilter(image[:, :, 1], image[:, :, 0], sigma_s, sigma_c, numIters=5)
            b = cv2.ximgproc.dtFilter(image[:, :, 2], image[:, :, 0], sigma_s, sigma_c, numIters=5)

        if i == 0:
            last_r = image[:, :, 0]
            last_g = image[:, :, 1]
            last_b = image[:, :, 2]
        else:
            last_r = lps[-1][0]
            last_g = lps[-1][1]
            last_b = lps[-1][2]

        d = np.sqrt(np.square(r - last_r) + np.square(g - last_g) + np.square(b - last_b))
        dhat = cv2.ximgproc.dtFilter(image, d, 20.0, sigma_c, numIters=5)
        # dhat[dhat < 0.4] = 0.0

        lps.append((r, g, b))
        ds.append(d)
        dhats.append(dhat)

    m = np.ones_like(ds[0])
    for i in range(k - 1):
        thresh = (1.0 - i / (k - 3)) * 0.04

        f = dhats[i] * ((dhats[i] > thresh) * m * (dhats[i] > dhats[i + 1]))
        m = m * (f == 0.0)

        fs.append(f)

    return lps, ds, dhats, fs
