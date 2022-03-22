from networks.cnn2d import cnn2d
from networks.cnnresnet2d import cnnresnet2d
from networks.secnn2d import secnn2d
from networks.secnnresnet2d import secnnresnet2d
from networks.cbamcnnresnet2d import cbamcnnresnet2d
from networks.cbamcnn2d import cbamcnn2d

from utils.train.generator import *

from keras.layers import *
from keras.models import Model


def multi_size_input(experiment_index):

    input      = Input(shape=(image_size,image_size))
    input3     = Reshape((image_size3, image_size3, 1), input_shape=(image_size3, image_size3))(input)
    input2   = Cropping3D(cropping=(int((image_size3-image_size2)/2)))(input3)
    input1   = Cropping3D(cropping=(int((image_size3-image_size1)/2)))(input3)

    if(experiment_index==1):
        net1 = cnn2d(input1)
        net2 = cnn2d(input2)
        net3 = cnn2d(input3)
    elif(experiment_index==2):
        net1 = resnet2d(input1)
        net2 = resnet2d(input2)
        net3 = resnet2d(input3)
    elif(experiment_index==3):
        net1 = secnnresnet2d(input1)
        net2 = secnnresnet2d(input2)
        net3 = secnnresnet2d(input3)
    elif(experiment_index==4):
        net1 = secnn2d(input1)
        net2 = secnn2d(input2)
        net3 = secnn2d(input3)

    net1_c = Lambda(lambda x: x * 0.3)(net1)
    net2_c = Lambda(lambda x: x * 0.3)(net2)
    net3_c = Lambda(lambda x: x * 0.4)(net3)
    net1_net2_c = Add()([net1_c, net2_c])
    net_out = Add()([net1_net2_c, net3_c])

    model = Model(inputs=[input], outputs=net_out)
    return model

def single_size_network(experiment_index):

    input      = Input(shape=(image_size,image_size))
    input1     = Reshape((image_size, image_size, 1), input_shape=(image_size, image_size))(input)

    if(experiment_index==1):
        net1 = cnn2d(input1)
    elif(experiment_index==2):
        net1 = cnnresnet2d(input1)
    elif(experiment_index==3):
        net1 = secnnresnet2d(input1)
    elif(experiment_index==4):
        net1 = secnn2d(input1)
    elif(experiment_index==5):
        net1 = cbamcnnresnet2d(input1)
    elif(experiment_index==6):
        net1 = cbamcnn2d(input1)

    model = Model(inputs=[input], outputs=net1)
    return model
