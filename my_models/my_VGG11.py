from keras.models import Sequential
from keras.layers import Conv2D, Dense, Activation, MaxPooling2D, \
    Dropout, Flatten,BatchNormalization
from keras.optimizers import adam_v2

def VGG_11():
    model = Sequential()
    '''第1个卷积块'''
    model.add(Conv2D(filters=64, kernel_size=3, padding='same', input_shape=(224, 224, 3), name='block1_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    '''第2个卷积块'''
    model.add(Conv2D(filters=128, kernel_size=3, padding='same', name='block2_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    '''第3个卷积块'''
    model.add(Conv2D(filters=256, kernel_size=3, padding='same', name='block3_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=256, kernel_size=3, padding='same', name='block3_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    '''第4个卷积块'''
    model.add(Conv2D(filters=512, kernel_size=3, padding='same', name='block4_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=512, kernel_size=3, padding='same', name='block4_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    '''第5个卷积块'''
    model.add(Conv2D(filters=512, kernel_size=3, padding='same', name='block5_conv1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Conv2D(filters=512, kernel_size=3, padding='same', name='block5_conv2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    '''展平'''
    model.add(Flatten())

    model.add(Dense(4096, activation='relu', name='fc1'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096, activation='relu', name='fc2'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10, activation='relu', name='output'))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    adm = adam_v2.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])
    return model
