from keras.layers import *
from keras.models import Model
from keras.backend import int_shape

import numpy as np

def cbam_block(cbam_feature, ratio=8, name="cbam0"):
	"""Contains the implementation of Convolutional Block Attention Module(CBAM) block.
	As described in https://arxiv.org/abs/1807.06521.
	"""

	cbam_feature = channel_attention(cbam_feature, ratio, name+'_attention_ch')
	cbam_feature = spatial_attention(cbam_feature, name+'_attention_sp')
	return cbam_feature

def channel_attention(input_feature, ratio=8, name="ch0"):

	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]

	shared_layer_one = Dense(channel//ratio,
							 activation='relu',
							 kernel_initializer='he_normal',
							 use_bias=True,
							 name=name+"_dense1",
							 bias_initializer='zeros')
	shared_layer_two = Dense(channel,
							 kernel_initializer='he_normal',
							 use_bias=True,
							 name=name+"_dense2",
							 bias_initializer='zeros')

	avg_pool = GlobalAveragePooling2D()(input_feature)
	avg_pool = Reshape((1,1,channel))(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)
	avg_pool = shared_layer_one(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel//ratio)
	avg_pool = shared_layer_two(avg_pool)
	assert avg_pool._keras_shape[1:] == (1,1,channel)

	max_pool = GlobalMaxPooling2D()(input_feature)
	max_pool = Reshape((1,1,channel))(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)
	max_pool = shared_layer_one(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel//ratio)
	max_pool = shared_layer_two(max_pool)
	assert max_pool._keras_shape[1:] == (1,1,channel)

	cbam_feature = Add(name=name+"_add1")([avg_pool,max_pool])
	cbam_feature = Activation('sigmoid', name=name+"_activation1")(cbam_feature)

	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)

	return multiply([input_feature, cbam_feature],name=name+"_mul1")

def spatial_attention(input_feature, name="sp0"):
	kernel_size = 3

	if K.image_data_format() == "channels_first":
		channel = input_feature._keras_shape[1]
		cbam_feature = Permute((2,3,1))(input_feature)
	else:
		channel = input_feature._keras_shape[-1]
		cbam_feature = input_feature

	avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True), name=name+"_dense1")(cbam_feature)
	assert avg_pool._keras_shape[-1] == 1
	max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True), name=name+"_dense12")(cbam_feature)
	assert max_pool._keras_shape[-1] == 1
	concat = Concatenate(axis=3)([avg_pool, max_pool])
	assert concat._keras_shape[-1] == 2
	cbam_feature = Conv2D(filters = 1,
					kernel_size=kernel_size,
					strides=1,
					padding='same',
					activation='sigmoid',
					kernel_initializer='he_normal',
					use_bias=False)(concat)
	assert cbam_feature._keras_shape[-1] == 1

	if K.image_data_format() == "channels_first":
		cbam_feature = Permute((3, 1, 2))(cbam_feature)

	return multiply([input_feature, cbam_feature],name=name+"_mul1")



def cbamcnn2d(input):
    '''
		Squeeze and excitation blocks applied on an 14-layer adapted version of ResNet18.
		Adapted for MNIST dataset.
		Input size is 28x28x1 representing images in MNIST.
		Output size is 10 representing classes to which images belong.
	'''

    conv1 = Conv2D(32, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(input)
    pool1 = MaxPooling2D((2, 2), strides=2)(conv1)

    block1 = cbam_block(pool1, 4, name="cbam1")
    conv2 = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(block1)
    conv3 = Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D((2, 2), strides=2)(conv3)
    block2 = cbam_block(pool2, 4, name="cbam2")

    flat = Flatten()(block2)
    drop = Dropout(0.20)(flat)
    fc1 = Dense(512, activation='relu', name="dense_a")(drop)
    output = Dense(2, activation='sigmoid', name="dense_last")(fc1)

    return output
