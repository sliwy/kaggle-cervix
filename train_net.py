import keras
import numpy as np
import pandas as pd
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, Dropout, Conv2D, GlobalMaxPooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split


def train_model(model, batch_size, num_classes, epochs, data_augmentation,
                model_name, X_train_train, y_train_train, X_test_train,
                y_test_train):
    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = (X_train_train, y_train_train), (
    X_test_train, y_test_train)
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    callbacks = [
        keras.callbacks.ModelCheckpoint(model_name, save_best_only=True,
                                        verbose=True),
        keras.callbacks.EarlyStopping(patience=6, verbose=1,
                                      mode='auto')]
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  epochs=epochs,
                  validation_data=(x_test, y_test),
                  shuffle=True)
        print('Returning None instead datagen.')
        return keras.models.load_model(model_name), None
    else:
        print('Using real-time data augmentation.')

        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            zoom_range=0.2, )

        datagen.fit(x_train)

        model.fit_generator(datagen.flow(x_train, y_train,
                                         batch_size=batch_size),
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
                            validation_data=datagen.flow(x_test, y_test,
                                                         batch_size=batch_size),
                            validation_steps=x_test.shape[0] // batch_size,
                            callbacks=callbacks)
        return keras.models.load_model(model_name), datagen


def test_augmentation(model, datagen, X, n_iter=50):
    pred = []
    print('Predictions are being computed...')
    for _ in range(n_iter):
        pred.append(model.predict_generator(datagen.flow(X, shuffle=False),
                                            X.shape[0] / 32, pickle_safe=True))
    return np.array(pred).mean(axis=0)


if __name__ == '__main__':

    # Path to the folder with pickles get from data preprocessing
    FILE_PATH = '../data/'
    MODELS_PATH = '../models/'

    X_train_cut = pd.read_pickle(FILE_PATH + 'X_train_cut')
    y_train_cut = pd.read_pickle(FILE_PATH + 'y_train_cut')
    X_test_file_name = pd.read_pickle(FILE_PATH + 'X_test_file_name')
    X_test_cut = pd.read_pickle(FILE_PATH + 'X_test_cut')
    X_test_stage_2 = pd.read_pickle(FILE_PATH + 'X_test_stage_2.pkl')
    X_test_file_name_stage_2 = pd.read_pickle(
        FILE_PATH + 'X_test_file_name_stage_2.pkl')
    y_test_stage_1 = pd.read_csv(FILE_PATH + '/solution_stg1_release.csv')

    X_train_train, X_test_train, y_train_train, y_test_train = train_test_split(
        X_train_cut, y_train_cut, test_size=0.25, random_state=123)

    X_train_train = np.concatenate([X_train_train, X_test_cut])
    y_train_train = np.concatenate(
        [y_train_train, np.argmax(y_test_stage_1.iloc[:, 1:].values, 1)])

    base_model = InceptionV3(include_top=False, weights='imagenet',
                             input_tensor=None,
                             input_shape=(224, 224, 3), classes=3)

    x = base_model.output
    x = Conv2D(512, (3, 3))(x)
    x = GlobalMaxPooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.25)(x)
    predictions = Dense(3, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in model.layers[:-5]:
        layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.sgd(lr=0.001, momentum=0.9, nesterov=True),
        loss='categorical_crossentropy')

    model, datagen = train_model(model, 32, 3, 40, True, 'inception1.hdf5',
                                 X_train_train, y_train_train, X_test_train,
                                 y_test_train)

    for layer in model.layers[:-5]:
        layer.trainable = True

    model.compile(optimizer=keras.optimizers.adam(lr=0.0001),
                  loss='categorical_crossentropy')

    model, datagen = train_model(model, 32, 3, 40, True, 'inception1.hdf5',
                                 X_train_train, y_train_train, X_test_train,
                                 y_test_train)

    preds = test_augmentation(model, datagen, X_test_stage_2, n_iter=5)
