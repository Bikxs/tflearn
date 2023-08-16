import os

import tensorflow as tf
from keras.applications import InceptionResNetV2
from keras.callbacks import EarlyStopping, CSVLogger, ReduceLROnPlateau
from keras.layers import Input, Dense, Dropout
from keras.models import Sequential

from etl import batch_size, data_generators
from utils import get_steps_per_epoch, train_eval_save

title = 'pretrained_inception_resnet_v2'


def make_model():
    model = Sequential([
        Input(shape=(224, 224, 3)),
        InceptionResNetV2(include_top=False, pooling='avg'),
        Dropout(0.4),
        Dense(200, activation='softmax')
    ])

    loss = tf.keras.losses.CategoricalCrossentropy()
    adam = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss=loss, optimizer=adam, metrics=['accuracy'])
    return model


def train(model):
    n_epochs = 50
    train_gen_aux, valid_gen_aux, _ = data_generators(tripple_y=False,target_size=(224,224))
    # Create a directory which stores model performance
    if not os.path.exists(f'models/{title}'):
        os.makedirs(f'models/{title}')

    # Callbacks
    es_callback = EarlyStopping(monitor='val_loss', patience=25)
    csv_logger = CSVLogger(os.path.join('eval', '4_eval_resnet_pretrained.log'))
    lr_callback = ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto'
    )

    history = model.fit(
        train_gen_aux, validation_data=valid_gen_aux,
        steps_per_epoch=get_steps_per_epoch(int(0.9 * (500 * 200)), batch_size),
        validation_steps=get_steps_per_epoch(int(0.1 * (500 * 200)), batch_size),
        epochs=n_epochs, callbacks=[es_callback, csv_logger, lr_callback]
    )
    return history, model


def main():
    train_eval_save(title, make_model, train)
