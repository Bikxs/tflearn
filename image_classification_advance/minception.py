import os
import time

import tensorflow as tf
from keras.callbacks import EarlyStopping, CSVLogger
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, AvgPool2D, Dense, Concatenate, Flatten, BatchNormalization, \
    Activation
from keras.layers.experimental.preprocessing import RandomCrop, RandomContrast
from keras.models import Model

from etl import random_seed, data_generators, batch_size
from utils import train_eval_save, get_steps_per_epoch

title = 'minception_resnet_v2'


def make_model():
    # Code listing 7.6
    def stem(inp, activation='relu', bn=True):

        conv1_1 = Conv2D(32, (3, 3), strides=(2, 2), activation=None, kernel_initializer=init, padding='same')(
            inp)  # 62x62
        if bn:
            conv1_1 = BatchNormalization()(conv1_1)
        conv1_1 = Activation(activation)(conv1_1)

        conv1_2 = Conv2D(32, (3, 3), strides=(1, 1), activation=None, kernel_initializer=init, padding='same')(
            conv1_1)  # 31x31
        if bn:
            conv1_2 = BatchNormalization()(conv1_2)
        conv1_2 = Activation(activation)(conv1_2)

        conv1_3 = Conv2D(64, (3, 3), strides=(1, 1), activation=None, kernel_initializer=init, padding='same')(
            conv1_2)  # 31x31
        if bn:
            conv1_3 = BatchNormalization()(conv1_3)
        conv1_3 = Activation(activation)(conv1_3)

        # Split to two branches
        # Branch 1
        maxpool2_1 = MaxPool2D((3, 3), strides=(2, 2), padding='same')(conv1_3)

        # Branch 2
        conv2_2 = Conv2D(96, (3, 3), strides=(2, 2), activation=None, kernel_initializer=init, padding='same')(conv1_3)
        if bn:
            conv2_2 = BatchNormalization()(conv2_2)
        conv2_2 = Activation(activation)(conv2_2)

        # Concat the results from two branches
        out2 = Concatenate(axis=-1)([maxpool2_1, conv2_2])

        # Split to two branches
        # Branch 1
        conv3_1 = Conv2D(64, (1, 1), strides=(1, 1), activation=None, kernel_initializer=init, padding='same')(out2)
        if bn:
            conv3_1 = BatchNormalization()(conv3_1)
        conv3_1 = Activation(activation)(conv3_1)

        conv3_2 = Conv2D(96, (3, 3), strides=(1, 1), activation=None, kernel_initializer=init, padding='same')(conv3_1)
        if bn:
            conv3_2 = BatchNormalization()(conv3_2)
        conv3_2 = Activation(activation)(conv3_2)

        # Branch 2
        conv4_1 = Conv2D(64, (1, 1), strides=(1, 1), activation=None, kernel_initializer=init, padding='same')(out2)
        if bn:
            conv4_1 = BatchNormalization()(conv4_1)
        conv4_1 = Activation(activation)(conv4_1)

        conv4_2 = Conv2D(64, (7, 1), strides=(1, 1), activation=None, kernel_initializer=init, padding='same')(conv4_1)
        if bn:
            conv4_2 = BatchNormalization()(conv4_2)

        conv4_3 = Conv2D(64, (1, 7), strides=(1, 1), activation=None, kernel_initializer=init, padding='same')(conv4_2)
        if bn:
            conv4_3 = BatchNormalization()(conv4_3)
        conv4_3 = Activation(activation)(conv4_3)

        conv4_4 = Conv2D(96, (3, 3), strides=(1, 1), activation=None, kernel_initializer=init, padding='same')(conv4_3)
        if bn:
            conv4_4 = BatchNormalization()(conv4_4)
        conv4_4 = Activation(activation)(conv4_4)

        # Concat the results from two branches
        out34 = Concatenate(axis=-1)([conv3_2, conv4_4])

        # Split to two branches
        # Branch 1
        maxpool5_1 = MaxPool2D((3, 3), strides=(2, 2), padding='same')(out34)
        # Branch 2
        conv6_1 = Conv2D(192, (3, 3), strides=(2, 2), activation=None, kernel_initializer=init, padding='same')(out34)
        if bn:
            conv6_1 = BatchNormalization()(conv6_1)
        conv6_1 = Activation(activation)(conv6_1)

        # Concat the results from two branches
        out56 = Concatenate(axis=-1)([maxpool5_1, conv6_1])

        return out56

    # Code listing 7.7
    def inception_resnet_a(inp, n_filters, initializer, activation='relu', bn=True, res_w=0.1):

        # Split to three branches
        # Branch 1
        out1_1 = Conv2D(n_filters[0][0], (1, 1), strides=(1, 1), activation=None,
                        kernel_initializer=initializer, padding='same')(inp)
        if bn:
            out1_1 = BatchNormalization()(out1_1)
        out1_1 = Activation(activation)(out1_1)

        # Branch 2
        out2_1 = Conv2D(n_filters[1][0], (1, 1), strides=(1, 1), activation=None,
                        kernel_initializer=initializer, padding='same')(inp)
        if bn:
            out2_1 = BatchNormalization()(out2_1)
        out2_1 = Activation(activation)(out2_1)

        out2_2 = Conv2D(n_filters[1][1], (1, 1), strides=(1, 1), activation=None,
                        kernel_initializer=initializer, padding='same')(out2_1)
        if bn:
            out2_2 = BatchNormalization()(out2_2)
        out2_2 = Activation(activation)(out2_2)

        # Branch 3
        out3_1 = Conv2D(n_filters[2][0], (1, 1), strides=(1, 1), activation=None,
                        kernel_initializer=initializer, padding='same')(inp)
        if bn:
            out3_1 = BatchNormalization()(out3_1)
        out3_1 = Activation(activation)(out3_1)

        out3_2 = Conv2D(n_filters[2][1], (3, 3), strides=(1, 1), activation=None,
                        kernel_initializer=initializer, padding='same')(out3_1)
        if bn:
            out3_2 = BatchNormalization()(out3_2)
        out3_2 = Activation(activation)(out3_2)

        out3_3 = Conv2D(n_filters[2][2], (3, 3), strides=(1, 1), activation=None,
                        kernel_initializer=initializer, padding='same')(out3_2)
        if bn:
            out3_3 = BatchNormalization()(out3_3)
        out3_3 = Activation(activation)(out3_3)

        out3_4 = Conv2D(n_filters[2][3], (1, 1), strides=(1, 1), activation=None,
                        kernel_initializer=initializer, padding='same')(out3_3)
        if bn:
            out3_4 = BatchNormalization()(out3_4)
        out3_4 = Activation(activation)(out3_4)

        # Concat the results from three branches
        out4_1 = Concatenate(axis=-1)([out1_1, out2_2, out3_4])
        out4_2 = Conv2D(n_filters[3][0], (1, 1), strides=(1, 1), activation=None,
                        kernel_initializer=initializer, padding='same')(out4_1)
        if bn:
            out4_2 = BatchNormalization()(out4_2)

        # Residual connection
        out4_2 += res_w * inp

        # Last activation
        out4_2 = Activation(activation)(out4_2)

        return out4_2

    # Code listing 7.8
    def inception_resnet_b(inp, n_filters, initializer, activation='relu', bn=True, res_w=0.1):

        # Split to two branches
        # Branch 1
        out1_1 = Conv2D(n_filters[0][0], (1, 1), strides=(1, 1), activation=None,
                        kernel_initializer=initializer, padding='same')(inp)
        if bn:
            out1_1 = BatchNormalization()(out1_1)
        out1_1 = Activation(activation)(out1_1)

        # Branch 2
        out2_1 = Conv2D(n_filters[1][0], (1, 1), strides=(1, 1), activation=activation,
                        kernel_initializer=initializer, padding='same')(inp)
        if bn:
            out2_1 = BatchNormalization()(out2_1)
        out2_1 = Activation(activation)(out2_1)

        out2_2 = Conv2D(n_filters[1][1], (1, 7), strides=(1, 1), activation=None,
                        kernel_initializer=initializer, padding='same')(out2_1)
        if bn:
            out2_2 = BatchNormalization()(out2_2)
        out2_2 = Activation(activation)(out2_2)

        out2_3 = Conv2D(n_filters[1][2], (7, 1), strides=(1, 1), activation=None,
                        kernel_initializer=initializer, padding='same')(out2_2)
        if bn:
            out2_3 = BatchNormalization()(out2_3)
        out2_3 = Activation(activation)(out2_3)

        # Concat the results from 2 branches
        out3_1 = Concatenate(axis=-1)([out1_1, out2_3])
        out3_2 = Conv2D(n_filters[2][0], (1, 1), strides=(1, 1), activation=None,
                        kernel_initializer=initializer, padding='same')(out3_1)
        if bn:
            out3_2 = BatchNormalization()(out3_2)

        # Residual connection
        out3_2 += res_w * inp

        # Last activation
        out3_2 = Activation(activation)(out3_2)

        return out3_2

    # Code listing 7.9
    def reduction(inp, n_filters, initializer, activation='relu', bn=True):
        # Split to three branches
        # Branch 1
        out1_1 = Conv2D(n_filters[0][0], (3, 3), strides=(2, 2), activation=activation,
                        kernel_initializer=initializer, padding='same')(inp)
        if bn:
            out1_1 = BatchNormalization()(out1_1)
        out1_2 = Conv2D(n_filters[0][1], (3, 3), strides=(1, 1), activation=activation,
                        kernel_initializer=initializer, padding='same')(out1_1)
        if bn:
            out1_2 = BatchNormalization()(out1_2)
        out1_3 = Conv2D(n_filters[0][2], (3, 3), strides=(1, 1), activation=activation,
                        kernel_initializer=initializer, padding='same')(out1_2)
        if bn:
            out1_3 = BatchNormalization()(out1_3)

        # Branch 2
        out2_1 = Conv2D(n_filters[1][0], (3, 3), strides=(2, 2), activation=activation,
                        kernel_initializer=initializer, padding='same')(inp)
        if bn:
            out2_1 = BatchNormalization()(out2_1)

        # Branch 3
        out3_1 = MaxPool2D((3, 3), strides=(2, 2), padding='same')(inp)

        # Concat the results from 3 branches
        out = Concatenate(axis=-1)([out1_3, out2_1, out3_1])

        return out

    activation = tf.nn.leaky_relu
    init = tf.keras.initializers.GlorotUniform()  # tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_in')

    bn = True

    # Code listing 7.10

    inp = Input(shape=(64, 64, 3))
    crop_inp = RandomCrop(56, 56, seed=random_seed)(inp)
    crop_inp = RandomContrast(0.3, seed=random_seed)(crop_inp)
    stem_out = stem(crop_inp)

    inc_a = inception_resnet_a(stem_out, [(32,), (32, 32), (32, 48, 64, 384), (384,)], initializer=init)

    red = reduction(inc_a, [(256, 256, 384), (384,)], initializer=init)

    inc_b1 = inception_resnet_b(red, [(192,), (128, 160, 192), (1152,)], initializer=init)
    inc_b2 = inception_resnet_b(inc_b1, [(192,), (128, 160, 192), (1152,)], initializer=init)

    avgpool1 = AvgPool2D((4, 4), strides=(1, 1), padding='valid')(inc_b2)
    flat_out = Flatten()(avgpool1)
    dropout1 = Dropout(0.5)(flat_out)
    out_main = Dense(200, activation='softmax', kernel_initializer=init, name='final')(dropout1)

    # Loss Weighing: https://github.com/tensorflow/models/blob/09d3c74a31d7e0c1742ae65025c249609b3c9d81/research/slim/train_image_classifier.py#L495
    minception_resnet_v2 = Model(inputs=inp, outputs=out_main)

    minception_resnet_v2.compile(loss=tf.keras.losses.CategoricalCrossentropy(), optimizer='adam', metrics=['accuracy'])

    return minception_resnet_v2

def train(model):
    train_gen_aux, valid_gen_aux, _ = data_generators(tripple_y=False)

    # Create a directory which stores model performance
    if not os.path.exists(f'models/{title}'):
        os.makedirs(f'models/{title}')

    es_callback = EarlyStopping(monitor='val_loss', patience=10)
    csv_logger = CSVLogger(os.path.join(f'models/{title}', 'early_stopping.log'))
    n_epochs = 50

    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto'
    )

    t1 = time.time()
    history = model.fit(
        train_gen_aux, validation_data=valid_gen_aux,
        steps_per_epoch=get_steps_per_epoch(int(0.9 * (500 * 200)), batch_size),
        validation_steps=get_steps_per_epoch(int(0.1 * (500 * 200)), batch_size),
        epochs=n_epochs, callbacks=[es_callback, csv_logger, lr_callback]
    )
    t2 = time.time()

    print(f"It took {t2 - t1} seconds to complete the training")
    return history, model


def main():
    train_eval_save(title, make_model, train)
