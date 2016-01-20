import cv2
import numpy as np
from operator import itemgetter


def get_sample_image(filename='example.jpg'):
    return cv2.imread(filename, 0)


def binarize(img=get_sample_image()):
    thresh = cv2.adaptiveThreshold(img, 255,
                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 4)
    return thresh


def show(img):
    import matplotlib.pyplot as plt
    plt.imshow(img, cmap=plt.cm.binary)


def find_digits(binary_img):
    inv = cv2.bitwise_not(binary_img)
    contours, hierarchy = cv2.findContours(inv,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    digits = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            [x, y, w, h] = cv2.boundingRect(cnt)
            margin = 20
            x -= margin
            y -= margin
            w += margin*2
            h += margin*2

            figure = binary_img[y: y + h, x: x + w]
            if figure.size > 0:
                digits.append({
                    'image': figure,
                    'x': x,
                    'y': y,
                    'w': w,
                    'h': h,
                })

    return digits


def resize_digits(digits):
    digits = map(itemgetter('image'), sorted(digits, key=itemgetter('x')))
    blur_kernel = np.ones((4, 4), np.float32)/(4*4)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return [
        cv2.resize(
            cv2.bitwise_not(
                cv2.filter2D(
                    cv2.erode(digit, erode_kernel, iterations=1),
                    -1, blur_kernel)
            ),
            (20, 20))
        for digit in digits]


def insert_into_center(resized_digits):
    results = []
    for img in resized_digits:
        i = np.zeros((28, 28))
        # calculate center of mass of the pixels
        M = cv2.moments(img)
        try:
            xc = M['m10'] / M['m00']
            yc = M['m01'] / M['m00']
        except ZeroDivisionError:
            xc = 10
            yc = 10

        # translating the image so as to position
        # this point at the center of the 28x28 field.
        start_a = max(min(4 + (10 - int(yc)), 8), 0)
        start_b = max(min(4 + (10 - int(xc)), 8), 0)
        i[start_a:start_a+20, start_b:start_b+20] = img

        results.append(i)
    return results
