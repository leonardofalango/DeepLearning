# Importando
from TrainTestSplit import train_test_split
from GenerateCnn import generateModel
from keras import callbacks
import sys

try:
    path = sys.argv[1]
except:
    raise Exception("Invalid argument train files")

save_as = ''

try:
    save_as = f'{sys.argv[1]}models/{sys.argv[2]}.h5'
except:
    save_as = ''


batch_size = 2**7

X_train, X_test = train_test_split(sys.argv[1], batch_size=batch_size)

train_data = len(X_train.filenames)
# Treinamento do modelo
model = generateModel()
steps_per_epoch = int(train_data / batch_size)

model.fit(
    X_train,
    steps_per_epoch = steps_per_epoch, # batch_size * steps_per_epoch = quant_dados
    epochs = 50, # Quantidade máxima de epocas
    validation_steps = int(steps_per_epoch / 10), # a cada alguns batch_sizes ele verifica
    callbacks = [
        callbacks.EarlyStopping(monitor='loss', patience = 4), # para o treinamento de o erro não melhorar em 4 épocas
        callbacks.ModelCheckpoint(
            filepath = save_as
                if save_as
                != '' else
                (sys.argv[1] + 'models/model.{epoch:02d}.keras') # Salvando cada época para poder voltar se crashar
        )
     ],
     validation_data=X_test
)

model.save('model')