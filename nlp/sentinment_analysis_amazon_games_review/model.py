import os
import time

import tensorflow.keras.backend as K
import tensorflow as tf

from nlp.sentinment_analysis_amazon_games_review.eda_etl import get_tf_pipeline, get_datasets, N_VOCAB

K.clear_session()


class OnehotEncoder(tf.keras.layers.Layer):
    def __init__(self, depth, **kwargs):
        super(OnehotEncoder, self).__init__(**kwargs)
        self.depth = depth

    def build(self, input_shape):
        pass

    def call(self, inputs):
        inputs = tf.cast(inputs, 'int32')

        if len(inputs.shape) == 3:
            inputs = inputs[:, :, 0]

        return tf.one_hot(inputs, depth=self.depth)

    def compute_mask(self, inputs, mask=None):
        return mask

    def get_config(self):
        config = super().get_config().copy()
        config.update({'depth': self.depth})
        return config


def make_model(n_vocab):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=(None,)),
        # Create a mask to mask out zero inputs
        tf.keras.layers.Masking(mask_value=0),
        # After creating the mask, convert inputs to onehot encoded inputs
        OnehotEncoder(depth=n_vocab),
        # Defining an LSTM layer
        tf.keras.layers.LSTM(128, return_state=False, return_sequences=False),
        # Defining a Dense layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train(model, batch_size):
    data = get_datasets(batch_size)
    train_ds = data['train_ds']
    valid_ds = data['valid_ds']
    # There is a class imbalance in the data therefore we are defining a weight for negative inputs
    neg_weight = (data['tr_y'] == 1).sum() / (data['tr_y'] == 0).sum()
    print(f"Will be using a weight of {neg_weight} for negative samples")

    os.makedirs('eval', exist_ok=True)

    # Logging the performance metrics to a CSV file
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join('eval', '1_sentiment_analysis.log'))

    monitor_metric = 'val_loss'
    mode = 'min'
    print("Using metric={} and mode={} for EarlyStopping".format(monitor_metric, mode))

    # Reduce LR callback
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric, factor=0.1, patience=3, mode=mode, min_lr=1e-8
    )

    # EarlyStopping itself increases the memory requirement
    # restore_best_weights will increase the memory req for large models
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric, patience=6, mode=mode, restore_best_weights=False
    )

    # Train the model
    t1 = time.time()

    history = model.fit(train_ds, validation_data=valid_ds, epochs=10, class_weight={0: neg_weight, 1: 1.0},
                        callbacks=[es_callback, lr_callback, csv_logger])
    t2 = time.time()

    print("It took {} seconds to complete the training".format(t2 - t1))
    return history, model


if __name__ == '__main__':
    model = make_model(N_VOCAB)
    print(model.summary())
    print("Defining data pipelines")

    # Using a batch size of 128
    batch_size = 128
    history, model = train(model, batch_size)
