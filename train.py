import numpy as np
import os
import struct

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.utils import np_utils


MODEL_FILENAME = 'digit.pkl'
WEIGHTS_FILENAME = 'weights1.h5'


def read(dataset="training", path="."):
    """
    Python function for importing the MNIST data set.  It returns an iterator
    of 2-tuples with the first element being the label and the second element
    being a numpy.uint8 2D array of pixel data for the given image.
    """

    if dataset is "training":
        fname_img = os.path.join(path, 'train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
    elif dataset is "testing":
        fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in xrange(len(lbl)):
        yield get_img(i)


def toMatrix(data, maxItems=30000):
    datalist = [t for t in data]
    m = maxItems
    n = 28 * 28
    X = np.zeros((m, n))
    Y = np.zeros(m)
    for i, (label, image) in enumerate(datalist[:m]):
        X[i, :] = image.reshape(28*28,)
        Y[i] = label

    idx = np.arange(len(X))
    np.random.shuffle(idx)
    return X[idx].astype(np.float)/255, Y[idx]


def cv(model, X, y, n_iter=5, test_size=0.3):
    split = cross_validation.ShuffleSplit(
        len(X), n_iter=n_iter, test_size=test_size,
    )
    return cross_validation.cross_val_score(model, X, y, cv=split,
                                            scoring='accuracy', n_jobs=-1)


def train_sk_model():
    model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
    X_train, y_train = toMatrix(read(), maxItems=60000)
    X_test, y_test = toMatrix(read(), maxItems=10000)
    print "Performing CV..."
    model.fit(X_train, y_train)
    print "Test accuracy:", accuracy_score(y_test, model.predict(X_test))

    joblib.dump(model, MODEL_FILENAME)


def get_ANN():
    model = Sequential()
    model.add(Dense(512, input_shape=(784,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms)
    return model


def train_ANN():
    X_train, y_train = toMatrix(read(), maxItems=60000)
    X_test, y_test = toMatrix(read(), maxItems=10000)

    nb_classes = 10
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    model = get_ANN()

    model.fit(X_train, Y_train,
              batch_size=128, nb_epoch=20,
              show_accuracy=True, verbose=2)
    score = model.evaluate(X_test, Y_test,
                           show_accuracy=True, verbose=0)
    print 'Test score:', score[0]
    print 'Test accuracy:', score[1]
    model.save_weights(WEIGHTS_FILENAME)


def get_model_sk():
    return joblib.load(MODEL_FILENAME)


def get_model():
    model = get_ANN()
    model.load_weights(WEIGHTS_FILENAME)
    return model

if __name__ == '__main__':
    # train_ANN()
    model = get_model()
    X_test, y_test = toMatrix(read(), maxItems=10000)
    print accuracy_score(
        np.argmax(model.predict(X_test), axis=1), y_test)
