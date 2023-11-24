# Convolution
from keras import preprocessing
from keras import regularizers
from keras import initializers
from keras import activations
from keras import optimizers
from keras import metrics
from keras import layers
from keras import models
from keras import losses

import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


isRgb = True
inputWidth = 80
inputheight = 80


input_shape = (inputWidth, inputheight, 3 if isRgb else 1)

def generateModel():
    model = models.Sequential()
    model.add(
    layers.Conv2D(
        64,
        (5, 5), # Perde 4 pixels de borda
        activation='relu')
    )

    # Pooling
    model.add(
        layers.MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    model.add(
    layers.Conv2D(
        32,
        (5, 5), # Perde 4 pixels de borda
        input_shape=((input_shape[0] - 4) / 2, (input_shape[1] - 4) / 2, input_shape[2]),
        activation='relu')
    )

    # Pooling
    model.add(
        layers.MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    for i in range(20, 2, -1):
        quant = 7 * (i * i)
        # print(f'Neurônios: {quant}')
        model.add(layers.Dense(
            quant
        ))
        model.add(layers.Activation(
            activation=activations.relu
        ))


    # Last layer
    model.add(layers.Dense(7))
    model.add(layers.Activation(
        activation=activations.relu
    ))

    model.compile(
        optimizer=optimizers.Adam(), # adam e sgd
        loss = losses.SparseCategoricalCrossentropy(), # loss function, função de calcular o erro
        metrics = [ metrics.SparseCategoricalAccuracy()]
    )
    
    return model
