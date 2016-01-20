from collections import defaultdict
from itertools import (
    islice,
)
from random import randint

import matplotlib.pyplot as plt
import numpy as np


from train import read
from main import get_digits


digits = defaultdict(list)
images = read()
for i, img in islice(images, None, 200):
    digits[i].append(img)

test_digits = get_digits()
real_values = [1, 0, 2, 3, 4, 5, 6, 7, 8, 9]


for i, img in zip(real_values, test_digits):
    fig = plt.figure()
    plt.title("Testset vs my")
    ax = fig.add_subplot(2, 1, 1)

    examples = []
    k = randint(0, min([len(digits[key]) for key in digits]))
    for e in digits[i][k:k+1]:
        examples.append(e)
        examples.append(np.ones((28, 2)))

    ax.imshow(np.hstack(tuple(examples)), cmap=plt.cm.binary)
    ax = fig.add_subplot(2, 1, 2)
    ax.imshow(img, cmap=plt.cm.binary)
