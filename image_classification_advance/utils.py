import json
import os

import keras.backend as K
import tensorflow as tf
from keras.src.saving.saving_api import load_model
from matplotlib import pyplot as plt

from etl import data_generators, batch_size


def init_gpus():
    print("TensorFlow version: {}".format(tf.__version__))
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                memory_limit = int(1024 * 6)
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
                print(f"Set memory limit for {gpu} to {memory_limit}MBs")

        except:
            print("Couldn't set memory_growth")


def plot_curves(history, title):
    folder = f'models/{title}'
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'{title} loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'{folder}/loss_plot.png')  # Save the figure as a PNG file

    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title(f'{title} accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f'{folder}/accuracy_plot.png')  # Save the figure as a PNG file

    history_json = json.dumps(history.history, indent=4)
    with open(f'{folder}/training_history.json', 'w') as json_file:
        json_file.write(history_json)


def model__is_trained(title):
    folder = f'models/{title}'
    filename = os.path.join(folder, f'model.h5')
    return os.path.exists(filename)


def save_model(model, title):
    folder = f'models/{title}'
    if not os.path.exists(folder):
        os.makedirs(folder)
    filename = os.path.join(folder, f'model.h5')
    model.save(filename)
    with open(f'{folder}/summary.txt', 'w') as summary_file:
        summary_file.write(model.summary())
    print(f"Saved model {filename}")


def eval_model(title):
    _, _, test_gen_aux = data_generators()
    # Load the model from disk
    folder = f'models/{title}'
    model = load_model(os.path.join(folder, 'model.h5'))

    # Evaluate the model
    test_res = model.evaluate(test_gen_aux, steps=get_steps_per_epoch(500 * 50, batch_size))

    # Print the results as a dictionary {<metric name>: <value>}
    test_res_dict = dict(zip(model.metrics_names, test_res))
    evaluation_json = json.dumps(test_res_dict.history, indent=4)
    with open(f'{folder}/evaluation.json', 'w') as json_file:
        json_file.write(evaluation_json)


def get_steps_per_epoch(n_data, batch_size):
    """ Given the data size and batch size, gives the number of steps to travers the full dataset """
    if n_data % batch_size == 0:
        return int(n_data / batch_size)
    else:
        return int(n_data * 1.0 / batch_size) + 1


def train_eval_save(title, make_model, train):
    try:
        K.clear_session()
        if model__is_trained(title):
            print(f"Model {title} already trained")
        else:
            model = make_model()
            history, model = train(model)
            plot_curves(history, title)
            save_model(model, title)
        eval_model(title)
    except Exception as e:
        print(f"Error on {title}:  {e}")
