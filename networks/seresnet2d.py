

from sklearn.metrics import roc_curve

from skimage.transform import resize

from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

from keras.layers import *
from keras.models import Model
from keras.backend import int_shape

import numpy as np

def se_block(block_input, num_filters, ratio=8):  # Squeeze and excitation block

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
    dense1 = Dense(num_filters // ratio, activation='relu')(flat)
    dense2 = Dense(num_filters, activation='sigmoid')(dense1)
    scale = multiply([block_input, dense2])

    return scale


def resnet_block(block_input, num_filters):  # Single ResNet block

    '''
		Args:
			block_input: input tensor to the ResNet block
			num_filters: no. of filters/channels in block_input

		Returns:
			relu2: activated tensor after addition with original input
	'''
    if int_shape(block_input)[3] != num_filters:
        block_input = Conv2D(num_filters, kernel_size=(1, 1))(block_input)

    conv1 = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(block_input)
    norm1 = BatchNormalization()(conv1)
    relu1 = Activation('relu')(norm1)
    conv2 = Conv2D(num_filters, kernel_size=(3, 3), padding='same')(relu1)
    norm2 = BatchNormalization()(conv2)

    se = se_block(norm2, num_filters=num_filters)

    sum = Add()([block_input, se])
    relu2 = Activation('relu')(sum)

    return relu2


def seresnet2d(input, input_size):
    '''
		Squeeze and excitation blocks applied on an 14-layer adapted version of ResNet18.
		Adapted for MNIST dataset.
		Input size is 28x28x1 representing images in MNIST.
		Output size is 10 representing classes to which images belong.
	'''

    input_r = Reshape((input_size[1], input_size[2], 1), input_shape=(input_size[1],input_size[2]))(input)
    conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(input_r)
    
    block1 = resnet_block(conv1, 16)

    block2 = resnet_block(block1, 16)

    block3= resnet_block(block2, 16)

    pool1 = MaxPooling2D((2, 2), strides=2)(block3)

    block4 = resnet_block(pool1, 32)

    block5 = resnet_block(block4, 32)

    block6 = resnet_block(block5, 32)

    block7 = resnet_block(block6, 32)

    pool2 = MaxPooling2D((2, 2), strides=2)(block7)

    block8 = resnet_block(pool2, 64)

    block9 = resnet_block(block8, 64)

    block10 = resnet_block(block9, 64)

    block11 = resnet_block(block10, 64)

    block12 = resnet_block(block11, 64)

    block13 = resnet_block(block12, 64)

    pool3 = MaxPooling2D((2, 2), strides=2)(block13)

    block14 = resnet_block(pool3, 128)

    block15 = resnet_block(block14, 128)

    block16 = resnet_block(block15, 128)

    pool4 = MaxPooling2D((2, 2), strides=2)(block16)


    flat = Flatten()(pool4)

    fc1 = Dense(1024, activation='relu')(flat)
    fc2 = Dense(512, activation='relu')(fc1)
    output = Dense(2, activation='sigmoid')(fc2)

    return output
