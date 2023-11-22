# Importando
from keras import models
from keras import layers
from keras import activations
from keras import regularizers
from keras import optimizers
from keras import losses
from keras import callbacks
from keras import preprocessing
from keras import initializers
from keras import metrics
from keras.preprocessing import image


model = models.Sequential()

isRgb = True
inputWidth = 64
inputheight = 64


input_shape = (inputWidth, inputheight, 3 if isRgb else 1)

# Convolution
model.add(
  layers.Conv2D(
    64,
    (5, 5), # Perde 4 pixels de borda
    input_shape=input_shape,
    activation='relu')
)

# Pooling
model.add(
    layers.MaxPooling2D(
        pool_size=(2, 2)
      )
)


model.add(
    layers.Flatten()
)

model.add(
    layers.Activation(
        activations.relu
    )
)

model.add(
    layers.Dense(
        256,
    )
)
model.add(layers.Activation(activations.softmax))

model.add(
    layers.Dense(
        128,
    )
)
model.add(layers.Activation(activations.tanh))


model.add(
    layers.Dropout(0.05)
)

model.add(
    layers.Dense(
        64,
    )
)
model.add(layers.Activation(activations.relu))

model.add(
    layers.Dense(
        16,
    )
)
model.add(layers.Activation(activations.elu))


# Last layer
model.add(
    layers.Dense(
        2,
    )
)
model.add(layers.Activation(activations.softmax))

model.compile(
    optimizer=optimizers.Adam(), # adam e sgd
    loss = losses.BinaryCrossentropy(), # loss function, função de calcular o erro
    metrics = [ metrics.Precision(), metrics.BinaryAccuracy(), metrics.CategoricalAccuracy()]
)

dataGen = image.ImageDataGenerator(
    rescale = 1.0 / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    vertical_flip = False,
    validation_split = 0.2
)

path = './PetImages'

batch_size = 16
X_train = dataGen.flow_from_directory(
    path,
    target_size = (64, 64), # resize
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'training'
)
train_data = len(X_train)
print('='*30)
print(train_data)
print('='*30)

X_test = dataGen.flow_from_directory(
    path,
    target_size = (64, 64),
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'validation'
)

import sys
# Treinamento do modelo
dataset_size = 10
model.fit(
    X_train,
    steps_per_epoch = train_data / batch_size, # batch_size * steps_per_epoch = quant_dados
    epochs = 50, # Quantidade máxima de epocas
    validation_steps = 50, # a cada 100 batch_sizes ele verifica
    callbacks = [
        callbacks.EarlyStopping(patience = 4), # para o treinamento de o erro não melhorar em 4 épocas
        callbacks.ModelCheckpoint(
            filepath = f'model.{sys.argv[1]}.h5' # Salvando cada época para poder voltar se crashar
        )
     ],
     validation_data=X_test
)

model.save('model')