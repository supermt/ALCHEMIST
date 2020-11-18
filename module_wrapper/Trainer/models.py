import tensorflow as tf
import datetime
from .MODEL_PARAMETERS import *


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    # log_dir="./logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping], verbose=0)
    return history


class Single_step_dense_net():
    def __init__(self, cell_list):
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=1)
        ])
        self.name = "single step dense net"
        self.name_short = "Singe Dense"

    def __repr__(self):
        return "Single step dense net"


class One_shot_range_multi_feature_dense_net():
    def __init__(self, out_steps, num_features):
        self.net = tf.keras.Sequential([
            # Take the last time-step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(out_steps*num_features,
                                  kernel_initializer=tf.initializers.zeros),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([out_steps, num_features])
        ])

    def __repr__(self):
        return "One-shot Multi step multi feature dense net"


class Multi_linear_range_multi_feature_dense_net():
    def __init__(self, out_steps, num_features):
        self.net = tf.keras.Sequential([
            # Take the last time-step.
            # Shape [batch, time, features] => [batch, 1, features]
            tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
            # Shape => [batch, 1, out_steps*features]
            tf.keras.layers.Dense(out_steps*num_features,
                                  kernel_initializer=tf.initializers.zeros),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([out_steps, num_features])
        ])


class LSTM_range_multi_feature_dense_net():
    def __init__(self, out_steps, num_features):
        self.net = tf.keras.Sequential([
            # Shape [batch, time, features] => [batch, lstm_units]
            # Adding more `lstm_units` just overfits more quickly.
            tf.keras.layers.LSTM(64, return_sequences=False),
            # Shape => [batch, out_steps*features]
            tf.keras.layers.Dense(out_steps*num_features,
                                  kernel_initializer=tf.initializers.zeros),
            # Shape => [batch, out_steps, features]
            tf.keras.layers.Reshape([out_steps, num_features])
        ])
    
    def __repr__(self):
        return "Range forecasting, LSTM"


class Single_shot_multi_step_dense_net():
    def __init__(self, cell_list):
        self.net = tf.keras.Sequential([
            # Shape: (time, features) => (time*features)
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=1),
            # Add back the time dimension.
            # Shape: (outputs) => (1, outputs)
            tf.keras.layers.Reshape([1, -1]),
        ])
        self.name = "multi step dense net"
        self.name_short = "Multi Dense"

    def __repr__(self):
        return "Single shot, multi step dense net"


class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)

        # The prediction for each timestep is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta


class MultiStepLastBaseline(tf.keras.Model):
    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])


class RepeatBaseline(tf.keras.Model):
    def call(self, inputs):
        return inputs


class AutoRegressionLSTM():
    def __init__(self,out_steps,num_features,range_window):
        feedback_model = FeedBack(32,out_steps,num_features)
        # prediction, state = feedback_model.warmup(range_window.example[0])
        self.net = feedback_model

# https://arxiv.org/abs/1308.0850
class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps, num_features):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)

        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        prediction, state = self.warmup(inputs)
        # Insert the first prediction
        predictions.append(prediction)

        # Run the rest of the prediction steps
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state,
                                      training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions
