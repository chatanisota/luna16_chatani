from keras.layers import *
from keras.models import Model
from keras.backend import int_shape

import numpy as np

def resnet_block(block_input, num_filters, name="res0"):  # Single ResNet block

    '''
		Args:
			block_input: input tensor to the ResNet block
			num_filters: no. of filters/channels in block_input

		Returns:
			relu2: activated tensor after addition with original input
	'''
    if int_shape(block_input)[3] != num_filters:
        block_input = Conv2D(num_filters, kernel_size=(1, 1))(block_input)

    conv1 = Conv2D(num_filters, kernel_size=(2, 2), padding='same')(block_input)
    norm1 = BatchNormalization()(conv1)
    relu1 = Activation('relu', name=name+"_activation1")(norm1)
    conv2 = Conv2D(num_filters, kernel_size=(2, 2), padding='same')(relu1)
    norm2 = BatchNormalization()(conv2)

    #se = se_block(norm2, num_filters=num_filters, name=name)

    sum = Add(name=name+"_add")([block_input, norm2])
    relu2 = Activation('relu', name=name+"_activation2")(sum)

    return relu2


def cnnresnet2d(input):
    '''
		Squeeze and excitation blocks applied on an 14-layer adapted version of ResNet18.
		Adapted for MNIST dataset.
		Input size is 28x28x1 representing images in MNIST.
		Output size is 10 representing classes to which images belong.
	'''

    conv1 = Conv2D(32, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(input)
    pool1 = MaxPooling2D((2, 2), strides=2)(conv1)

    block1 = resnet_block(pool1, 32, name="res1")
    conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(block1)
    conv3 = Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=2)(conv3)
    block2 = resnet_block(pool2, 64, name="res2")

    # drop out https://medium.com/axinc/dropout%E3%81%AB%E3%82%88%E3%82%8B%E9%81%8E%E5%AD%A6%E7%BF%92%E3%81%AE%E6%8A%91%E5%88%B6-be5b9bba7e89
    flat = Flatten()(block2)
    drop = Dropout(0.20)(flat)
    fc1 = Dense(512, activation='relu', name="dense_a")(drop)
    output = Dense(2, activation='sigmoid', name="dense_last")(fc1)

    return output
