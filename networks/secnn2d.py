from keras.layers import *
from keras.models import Model
from keras.backend import int_shape

import numpy as np

def se_block(block_input, num_filters, ratio=8, name="se0"):  # Squeeze and excitation block

    '''
		Args:
			block_input: input tensor to the squeeze and excitation block
			num_filters: no. of filters/channels in block_input
			ratio: a hyperparameter that denotes the ratio by which no. of channels will be reduced

		Returns:
			scale: scaled tensor after getting multiplied by new channel weights
	'''

    pool1 = GlobalAveragePooling2D()(block_input)
    flat = Reshape((1, 1, num_filters))(pool1)
    dense1 = Dense(num_filters // ratio, activation='relu', name=name+"_dense_a")(flat)
    dense2 = Dense(num_filters, activation='sigmoid', name=name+"_dense_b")(dense1)
    scale = multiply([block_input, dense2], name=name+"_multiply")

    return scale


def secnn2d(input):
    '''
		Squeeze and excitation blocks applied on an 14-layer adapted version of ResNet18.
		Adapted for MNIST dataset.
		Input size is 28x28x1 representing images in MNIST.
		Output size is 10 representing classes to which images belong.
	'''

    conv1 = Conv2D(32, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(input)
    pool1 = MaxPooling2D((2, 2), strides=2)(conv1)

    block1 = se_block(pool1, 32, name="se1")
    conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(block1)
    conv3 = Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=2)(conv3)
    block2 = se_block(pool2, 64, name="se2")

    flat = Flatten()(block2)
    drop = Dropout(0.20)(flat)
    fc1 = Dense(512, activation='relu', name="dense_a" )(drop)
    output = Dense(2, activation='sigmoid', name="dense_last")(fc1)

    return output
