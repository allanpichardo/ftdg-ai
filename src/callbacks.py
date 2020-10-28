import tensorflow as tf


class TensorBoardImage(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if self.validation_data is None:
            return
        # Load image
        batch = self.validation_data[0]
        X = batch[0]

        Y = self.model.predict_on_batch(X)
        tf.summary.image('Inputs', X)
        tf.summary.image('Outputs', Y)
