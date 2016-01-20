import cv2
import matplotlib.pyplot as plt
import numpy as np

from image import (
    binarize,
    find_digits,
    resize_digits,
    insert_into_center,
    get_sample_image,
)
from train import get_model


def draw_contours(frame, contours):
    for img in contours:
        cv2.rectangle(
            frame,
            (img['x'], img['y']),
            (img['x'] + img['w'], img['y'] + img['h']),
            (0, 0, 0),
            4
        )


def preprocess(digits):
    return np.vstack([digit.reshape(28*28).astype(np.float)/255
                      for digit in digits])


def video():
    model = get_model()
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('OCR')
    last_seen = "Number: NaN"

    while(True):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thresh = binarize(gray)
        key = cv2.waitKey(1)

        contours = find_digits(thresh)
        draw_contours(frame, contours)
        if key == ord('p'):
            digits = insert_into_center(resize_digits(contours))
            if digits:
                X = preprocess(digits)

                prediction = np.argmax(model.predict(X), axis=1)
                last_seen = "Number: " + "".join(map(str, prediction))

                # print last_seen
                # plt.imshow(np.hstack(tuple(digits)), cmap=plt.cm.binary)
                # plt.show()

        cv2.putText(frame, last_seen, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, 0)
        cv2.imshow('OCR', frame)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def static_image():
    model = get_model()
    frame = get_sample_image()
    contours = find_digits(binarize(frame.copy()))
    draw_contours(frame, contours)
    digits = insert_into_center(resize_digits(contours))
    X = preprocess(digits)
    print np.argmax(model.predict(X), axis=1)
    plt.imshow(np.hstack(tuple(digits)), cmap=plt.cm.binary)
    plt.show()


if __name__ == '__main__':
    # video()
    static_image()
