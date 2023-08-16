# import tensorflow_hub as hub

import os
import random
from functools import partial
from itertools import tee

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

batch_size = 128
random_seed = 4321


def fix_random_seed(seed):
    """ Setting the random seed of various libraries """
    try:
        np.random.seed(seed)
    except NameError:
        print("Warning: Numpy is not imported. Setting the seed for Numpy failed.")
    try:
        tf.random.set_seed(seed)
    except NameError:
        print("Warning: TensorFlow is not imported. Setting the seed for TensorFlow failed.")
    try:
        random.seed(seed)
    except NameError:
        print("Warning: random module is not imported. Setting the seed for random failed.")


def data_generators(target_size=(56, 56), tripple_y=True):
    def get_test_labels_df(test_labels_path):
        """ Reading the test data labels for all files in the test set as a data frame """
        test_df = pd.read_csv(test_labels_path, sep='\t', index_col=None, header=None)
        test_df = test_df.iloc[:, [0, 1]].rename({0: "filename", 1: "class"}, axis=1)
        return test_df

    def get_train_valid_test_data_generators(batch_size, target_size):
        """ Get the training/validation/testing data generators """

        # Code listing 7.1
        # Defining a data-augmenting image data generator and a standard image data generator
        image_gen_aug = ImageDataGenerator(
            samplewise_center=False, rotation_range=30, width_shift_range=0.2,
            height_shift_range=0.2, brightness_range=(0.5, 1.5), shear_range=5,
            zoom_range=0.2, horizontal_flip=True, fill_mode='reflect', validation_split=0.1
        )
        image_gen = ImageDataGenerator(samplewise_center=False)

        # Code listing 7.2
        # Define a training data generator
        partial_flow_func = partial(
            image_gen_aug.flow_from_directory,
            directory=os.path.join('data', 'tiny-imagenet-200', 'train'),
            target_size=target_size, classes=None,
            class_mode='categorical', interpolation='bilinear', batch_size=batch_size,
            shuffle=True, seed=random_seed)

        # Get the training data subset
        train_gen = partial_flow_func(subset='training')
        # Get the validation data subset
        valid_gen = partial_flow_func(subset='validation')

        # Defining the test data generator
        test_df = get_test_labels_df(os.path.join('data', 'tiny-imagenet-200', 'val', 'val_annotations.txt'))
        test_gen = image_gen.flow_from_dataframe(
            test_df, directory=os.path.join('data', 'tiny-imagenet-200', 'val', 'images'), target_size=target_size,
            classes=None,
            class_mode='categorical', interpolation='bilinear', batch_size=batch_size, shuffle=False
        )
        return train_gen, valid_gen, test_gen

    # Code listing 7.3
    def data_gen_augmented_inceptionnet_v1(gen, random_gamma=False, random_occlude=False):
        for x, y in gen:

            if x.ndim != 4:
                raise ValueError("This function is designed for a batch of images with 4 dims [b, h, w, c]")

            if random_gamma:
                # Gamma correction
                # Doing this in the image process fn doesn't help improve performance
                rand_gamma = np.random.uniform(0.9, 1.08, (x.shape[0], 1, 1, 1))
                x = x ** rand_gamma

            if random_occlude:
                # Randomly occluding sections in the image
                occ_size = 10
                occ_h, occ_w = np.random.randint(0, x.shape[1] - occ_size), np.random.randint(0, x.shape[2] - occ_size)
                x[::2, occ_h:occ_h + occ_size, occ_w:occ_w + occ_size, :] = np.random.choice([0., 128., 255.])

            # Image centering
            x -= np.mean(x, axis=(1, 2, 3), keepdims=True)

            # Making sure we replicate the target (y) three times
            if tripple_y:
                yield x, (y, y, y)
            else:
                yield x, y

    # Getting the train,valid, test data generators
    train_gen, valid_gen, test_gen = get_train_valid_test_data_generators(batch_size, target_size)
    # Modifying the data generators to fit the model targets
    # We augment data in the training set
    train_gen_aux = data_gen_augmented_inceptionnet_v1(train_gen, random_gamma=True, random_occlude=True)
    # We do not augment data in the validation/test datasets
    valid_gen_aux = data_gen_augmented_inceptionnet_v1(valid_gen)
    test_gen_aux = data_gen_augmented_inceptionnet_v1(test_gen)
    return train_gen_aux, valid_gen_aux, test_gen_aux


def validate_etl():
    all_labels = []
    n_trials = 10
    train_gen_aux, valid_gen_aux, test_gen_aux = data_generators()
    valid_gen_test = tee(train_gen_aux, n_trials)

    for i in range(n_trials):
        labels = []
        for j in range(5):
            _, ohe = next(valid_gen_test[i])
            # Convert one hot encoded to class labels
            labels.append(np.argmax(ohe, axis=-1))

        # Concat all labels
        labels = np.reshape(np.concatenate(labels, axis=0), (1, -1))
        all_labels.append(labels)

    # Concat all labels accross all trials
    all_labels = np.concatenate(all_labels, axis=0)

    # Assert the labels are equal across all trials
    assert np.all(np.all(all_labels == all_labels[0, :], axis=0)), "Labels across multiple trials were not equal"
    print("ETL validation successful! Labels across all trials were consistent.")

    data = []  # Holds both training and validation samples to be plotted

    # Getting training samples (20 samples)
    for i, (x, y) in enumerate(train_gen_aux):
        if i >= 20: break
        data.append((x[0, :, :, :] + 128).astype('int32'))

    # Getting validation samples (20 samples)
    for i, (x, y) in enumerate(valid_gen_aux):
        if i >= 20: break
        data.append((x[0, :, :, :] + 128).astype('int32'))

    # Creating a plot with 40 subplots (4 rows and 10 columns)
    n_rows = 4
    n_cols = 10
    f, axes = plt.subplots(n_rows, n_cols, figsize=(18, 9))

    # Plot the training and validation images
    # First 2 rows are training data
    # Second 2 rows are validation data
    for ri in range(n_rows):
        for ci in range(n_cols):
            # Plotting the correct image at ri,ci position in the plot
            i = ri * n_cols + ci
            axes[ri][ci].imshow(data[i])
            axes[ri][ci].axis('off')

            # Setting plot titles
            if ri == 0 and ci == n_cols // 2:
                axes[ri][ci].set_title("Training data", fontsize=20, pad=1, x=-0.15)
            elif ri == 2 and ci == n_cols / 2:
                axes[ri][ci].set_title("Validation data", fontsize=20, pad=1, x=-0.15)

    f.show()


if __name__ == '__main__':
    # Fixing the random seed

    fix_random_seed(random_seed)
    init_gpus()
    validate_etl()
