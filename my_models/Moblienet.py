import tensorflow
import keras
from keras.layers import Dropout,Conv2D,DepthwiseConv2D,Dense,AveragePooling2D,Input,Flatten,BatchNormalization
from keras import Model
from keras.layers.advanced_activations import ReLU
from keras.optimizers import adam_v2


def depth_point_conv2d(x, s=[1, 1, 2, 1], channel=[64, 128]):
    """
    s:the strides of the conv
    channel: the depth of pointwiseconvolutions
    """

    dw1 = DepthwiseConv2D((3, 3), strides=s[0], padding='same')(x)
    bn1 = BatchNormalization()(dw1)
    relu1 = ReLU()(bn1)
    pw1 = Conv2D(filters=channel[0], kernel_size=(1, 1), strides=s[1], padding='same')(relu1)
    bn2 = BatchNormalization()(pw1)
    relu2 = ReLU()(bn2)
    dw2 = DepthwiseConv2D(kernel_size = (3, 3), strides=s[2], padding='same')(relu2)
    bn3 = BatchNormalization()(dw2)
    relu3 = ReLU()(bn3)
    pw2 = Conv2D(filters=channel[1], kernel_size=(1, 1), strides=s[3], padding='same')(relu3)
    bn4 = BatchNormalization()(pw2)
    relu4 = ReLU()(bn4)

    return relu4


def repeat_conv(x, s=[1, 1], channel=512):
    dw1 = DepthwiseConv2D(kernel_size=(3, 3), strides=s[0], padding='same')(x)
    bn1 = BatchNormalization()(dw1)
    relu1 = ReLU()(bn1)
    pw1 = Conv2D(filters=channel, kernel_size=(1, 1), strides=s[1], padding='same')(relu1)
    bn2 = BatchNormalization()(pw1)
    relu2 = ReLU()(bn2)

    return relu2


def MobileNet():
    h0 = Input(shape=(224, 224, 3))
    h1 = Conv2D(filters=32, kernel_size=(3, 3), strides=2, padding="same")(h0)
    h2 = BatchNormalization()(h1)
    h3 = ReLU()(h2)
    h4 = depth_point_conv2d(h3, s=[1, 1, 2, 1], channel=[64, 128])
    h5 = depth_point_conv2d(h4, s=[1, 1, 2, 1], channel=[128, 256])
    h6 = depth_point_conv2d(h5, s=[1, 1, 2, 1], channel=[256, 512])
    h7 = repeat_conv(h6)
    h8 = repeat_conv(h7)
    h9 = repeat_conv(h8)
    h10 = repeat_conv(h9)
    h11 = depth_point_conv2d(h10, s=[1, 1, 2, 1], channel=[512, 1024])
    h12 = repeat_conv(h11, channel=1024)
    h13 = AveragePooling2D((7, 7))(h12)
    h14 = Flatten()(h13)
    h15 = Dense(1000, activation='relu')(h14)
    h16 = Dropout(0.5)(h15)
    h17 = Dense(10, activation='softmax')(h16)
    model = Model(inputs=h0, outputs=h17)

    # adm = adam_v2.Adam(learning_rate=0.001)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
