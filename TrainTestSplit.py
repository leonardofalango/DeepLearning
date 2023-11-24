from keras.preprocessing import image


def train_test_split(path, batch_size=16, shear_range=.2, zoom_range=.2, horizontal_flip=True, vertical_flip=False, validation_split=.2):
    dataGen = image.ImageDataGenerator(
    rescale = 1.0 / 255,
    shear_range = shear_range,
    zoom_range = zoom_range,
    horizontal_flip = horizontal_flip,
    vertical_flip = vertical_flip,
    validation_split=validation_split
    )


    X_train = dataGen.flow_from_directory(
        path + '/assets',
        target_size = (80, 80), # resize
        batch_size = batch_size,
        class_mode = 'categorical',
        subset = 'training'
    )

    X_test = dataGen.flow_from_directory(
        path + '/assets',
        target_size = (80, 80),
        batch_size = batch_size,
        class_mode = 'categorical',
        subset = 'validation'
    )

    return (X_train, X_test)