from keras.layers import *
from keras.backend import int_shape
from keras.models import Sequential

import numpy as np


def cnn2d(input):

    conv1       = Conv2D(32,5)(input)
    activ1      = Activation('relu', name="activation_a")(conv1)
    bool1       = MaxPool2D(pool_size=(2,2))(activ1)
    conv2       = Conv2D(64,5)(bool1)
    activ2      = Activation('relu', name="activation_b")(conv2)
    bool2       = MaxPool2D(pool_size=(2,2))(activ2)
    flat2       = Flatten()(bool2)
    dense2      = Dense(1024, name="dense_a")(flat2)
    activ3      = Activation('relu', name="activation_c")(dense2)
    drop1       = Dropout(0.2)(activ3)
    output      = Dense(2, activation='sigmoid', name="dense_last")(drop1)

    return output
