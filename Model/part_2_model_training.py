from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Activation, MaxPooling2D, BatchNormalization, Activation, Conv2D


def tfl_model():
    input_shape = (81, 81, 3)

    model = Sequential()

    def conv_bn_relu(filters, **conv_kw):
        model.add(Conv2D(filters,  use_bias=False, kernel_initializer='he_normal', **conv_kw))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def dense_bn_relu(units):
        model.add(Dense(units, use_bias=False, kernel_initializer='he_normal'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))

    def spatial_layer(count, filters):
        for i in range(count):
            conv_bn_relu(filters, kernel_size=(3, 3))
        conv_bn_relu(filters, kernel_size=(3, 3), strides=(2, 2))

    conv_bn_relu(32, kernel_size=(3, 3), input_shape=input_shape)
    spatial_layer(1, 32)
    spatial_layer(2, 64)
    spatial_layer(2, 96)

    model.add(Flatten())
    dense_bn_relu(96)
    model.add(Dense(2, activation='softmax'))
    return model


m = tfl_model()
m.summary()
