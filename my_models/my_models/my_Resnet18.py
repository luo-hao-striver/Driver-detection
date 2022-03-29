from keras import Input
from keras.models import Model
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D, AveragePooling2D, Dropout, \
    Flatten, BatchNormalization, Add, GlobalAveragePooling2D, Softmax
from keras.optimizers import adam_v2

# ResNet18网络对应的残差模块a和残差模块b
def resiidual_a_or_b(input_x,filters, flag):
    if flag == 'a':
        # 主路
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(input_x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(x)
        x = BatchNormalization()(x)

        y = Add()([x, input_x])
        y = Activation('relu')(y)

        return y

    elif flag == 'b':
        # 主路
        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=2, padding='same')(input_x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=filters, kernel_size=(3, 3), strides=1, padding='same')(x)
        x = BatchNormalization()(x)

        # 支路下采样
        input_x = Conv2D(filters=filters, kernel_size=(1, 1), strides=2, padding='same')(input_x)
        input_x = BatchNormalization()(input_x)

        # 输出
        y = Add()([x, input_x])
        y = Activation('relu')(y)

        return y


def Resnet_18():
    # 第一层
    input_layer = Input(shape=(224, 224, 3))
    x = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same')(input_layer)
    conv1_Maxpooling = MaxPooling2D(pool_size=(3,3),strides=2)(x)

    # conv2_x
    x = resiidual_a_or_b(conv1_Maxpooling, 64, 'b')
    x = resiidual_a_or_b(x, 64, 'a')

    # conv3_x
    x = resiidual_a_or_b(x, 128, 'b')
    x = resiidual_a_or_b(x, 128, 'a')

    # conv4_x
    x = resiidual_a_or_b(x, 256, 'b')
    x = resiidual_a_or_b(x, 256, 'a')

    # conv5_x
    x = resiidual_a_or_b(x, 512, 'b')
    x = resiidual_a_or_b(x, 512, 'a')

    # 最后一层
    x = MaxPooling2D(pool_size=(3,3),strides=2)(x)
    # x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x= Dense(1000, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(10,activation='softmax')(x)

    model = Model([input_layer], [y])

    adm = adam_v2.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
    return model


