from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.layers import Conv2D, ZeroPadding2D

def layers(input_shape):
    print("dense layer: {}".format(input_shape[0]*input_shape[1]))
    return [
#        ZeroPadding2D(padding=3, input_shape=input_shape,
#                      data_format='channels_first'),
#        Conv2D(48, (7,7), data_format='channels_first'),
        ZeroPadding2D(padding=3, input_shape=input_shape,
                      data_format='channels_last'),
        Conv2D(48, (7,7), data_format='channels_last'),
        Activation('relu'),

#        ZeroPadding2D(padding=2, data_format='channels_first'),
#        Conv2D(32, (5,5), data_format='channels_first'),
        ZeroPadding2D(padding=2, data_format='channels_last'),
        Conv2D(32, (5,5), data_format='channels_last'),
        Activation('relu'),

#        ZeroPadding2D(padding=2, data_format='channels_first'),
#        Conv2D(32, (5,5), data_format='channels_first'),
        ZeroPadding2D(padding=2, data_format='channels_last'),
        Conv2D(32, (5,5), data_format='channels_last'),
        Activation('relu'),

#        ZeroPadding2D(padding=2, data_format='channels_first'),
#        Conv2D(32, (5,5), data_format='channels_first'),
        ZeroPadding2D(padding=2, data_format='channels_last'),
        Conv2D(32, (5,5), data_format='channels_last'),
        Activation('relu'),

        Flatten(),
        Dense(512),
        #Dense(19*19),
        #Dense(input_shape[0]*input_shape[1]),
        Activation('relu'),
    ]
