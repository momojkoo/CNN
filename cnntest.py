from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Model

NUM_CLASSES = 5

input_layer = Input(shape=(32,32,3))
x = Conv2D(filters = 32, kernel_size=3, strides = 1, padding='same')(input_layer)
x = BatchNormalization()(x)
x = LeaykyReLU()(x)

x = Conv2D(filters = 32, kernel_size=3, strides = 2, padding='same')(x)
x = BatchNormalization()(x)
x = LeaykyReLU()(x)

x = Conv2D(filters = 64, kernel_size=3, strides = 1, padding='same')(x)
x = BatchNormalization()(x)
x = LeaykyReLU()(x)

x = Conv2D(filters = 64, kernel_size=3, strides = 2, padding='same')(x)
x = BatchNormalization()(x)
x = LeaykyReLU()(x)

x = Flatten()(x)

x = Dense(128)(x)
x = BatchNormalization()(x)
x = LeaykyReLU()(x)

x = Dropout(rate=0.5)(x)

x = Dense(NUM_CLASSES)(x)
output_layer = Activation('softmax')(x)

model = Model(input_layer, output_layer)

model.summary()

