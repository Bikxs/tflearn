import time

from keras.layers import Input, Conv2D, MaxPool2D, AvgPool2D, Dense, Concatenate, Flatten, Lambda, Dropout
from keras.models import Model
from keras.src.callbacks import EarlyStopping, CSVLogger

from utils import *

title = 'inception_v1'


def make_model(metrics):
    def inception(inp, n_filters):
        # 1x1 layer
        # init argument defaults to glorot_uniform
        out1 = Conv2D(n_filters[0][0], (1, 1), strides=(1, 1), activation='relu', padding='same')(inp)

        # 1x1 followed by 3x3
        out2_1 = Conv2D(n_filters[1][0], (1, 1), strides=(1, 1), activation='relu', padding='same')(inp)
        out2_2 = Conv2D(n_filters[1][1], (3, 3), strides=(1, 1), activation='relu', padding='same')(out2_1)

        # 1x1 followed by 5x5
        out3_1 = Conv2D(n_filters[2][0], (1, 1), strides=(1, 1), activation='relu', padding='same')(inp)
        out3_2 = Conv2D(n_filters[2][1], (5, 5), strides=(1, 1), activation='relu', padding='same')(out3_1)

        # 3x3 (pool) followed by 1x1
        out4_1 = MaxPool2D((3, 3), strides=(1, 1), padding='same')(inp)
        out4_2 = Conv2D(n_filters[3][0], (1, 1), strides=(1, 1), activation='relu', padding='same')(out4_1)

        out = Concatenate(axis=-1)([out1, out2_2, out3_2, out4_2])
        return out

    # Code listing 7.4
    def aux_out(inp, name=None):
        avgpool1 = AvgPool2D((5, 5), strides=(3, 3), padding='valid')(inp)
        conv1 = Conv2D(128, (1, 1), activation='relu', padding='same')(avgpool1)
        flat = Flatten()(conv1)
        dense1 = Dense(1024, activation='relu')(flat)
        dense1 = Dropout(0.7)(dense1)
        aux_out = Dense(200, activation='softmax', name=name)(dense1)
        return aux_out

    def stem(inp):
        conv1 = Conv2D(64, (7, 7), strides=(1, 1), activation='relu', padding='same')(inp)
        maxpool2 = MaxPool2D((3, 3), strides=(2, 2), padding='same')(conv1)
        lrn3 = Lambda(lambda x: tf.nn.local_response_normalization(x))(maxpool2)

        conv4 = Conv2D(64, (1, 1), strides=(1, 1), padding='same')(lrn3)
        conv5 = Conv2D(192, (3, 3), strides=(1, 1), activation='relu', padding='same')(conv4)
        lrn6 = Lambda(lambda x: tf.nn.local_response_normalization(x))(conv5)

        maxpool7 = MaxPool2D((3, 3), strides=(1, 1), padding='same')(lrn6)

        return maxpool7

    def inception_v1():
        K.clear_session()

        inp = Input(shape=(56, 56, 3))
        stem_out = stem(inp)
        inc_3a = inception(stem_out, [(64,), (96, 128), (16, 32), (32,)])
        inc_3b = inception(inc_3a, [(128,), (128, 192), (32, 96), (64,)])

        maxpool = MaxPool2D((3, 3), strides=(2, 2), padding='same')(inc_3b)

        inc_4a = inception(maxpool, [(192,), (96, 208), (16, 48), (64,)])
        inc_4b = inception(inc_4a, [(160,), (112, 224), (24, 64), (64,)])

        aux_out1 = aux_out(inc_4a, name='aux1')

        inc_4c = inception(inc_4b, [(128,), (128, 256), (24, 64), (64,)])
        inc_4d = inception(inc_4c, [(112,), (144, 288), (32, 64), (64,)])
        inc_4e = inception(inc_4d, [(256,), (160, 320), (32, 128), (128,)])

        maxpool = MaxPool2D((3, 3), strides=(2, 2), padding='same')(inc_4e)

        aux_out2 = aux_out(inc_4d, name='aux2')

        inc_5a = inception(maxpool, [(256,), (160, 320), (32, 128), (128,)])
        inc_5b = inception(inc_5a, [(384,), (192, 384), (48, 128), (128,)])
        avgpool1 = AvgPool2D((7, 7), strides=(1, 1), padding='valid')(inc_5b)

        flat_out = Flatten()(avgpool1)
        flat_out = Dropout(0.4)(flat_out)
        out_main = Dense(200, activation='softmax', name='final')(flat_out)

        model = Model(inputs=inp, outputs=[out_main, aux_out1, aux_out2])
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam', metrics=metrics)
        return model

    return inception_v1()


def train(model,epochs):
    train_gen_aux, valid_gen_aux, _ = data_generators()
    # Create a directory which stores model performance
    if not os.path.exists(f'models/{title}'):
        os.makedirs(f'models/{title}')

    # Early stopping callback
    es_callback = EarlyStopping(monitor='val_loss', patience=5)
    csv_logger = CSVLogger(os.path.join(f'models/{title}', 'early_stopping.log'))

    t1 = time.time()
    history = model.fit(
        train_gen_aux, validation_data=valid_gen_aux,
        steps_per_epoch=get_steps_per_epoch(int(0.9 * (500 * 200)), batch_size),
        validation_steps=get_steps_per_epoch(int(0.1 * (500 * 200)), batch_size),
        epochs=epochs, callbacks=[es_callback, csv_logger]
    )
    t2 = time.time()

    print(f"It took {t2 - t1} seconds to complete the training")
    return history, model


def main():
    train_eval_save(title, make_model, train)
