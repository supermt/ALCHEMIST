import itertools
import pandas as pd
import re
from log_class import log_recorder
import os
import datetime
from MODEL_PARAMETERS import *
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

LOG_DIR = "../StorageMaterial.NVMeSSD/8CPU/128MB/"
report_csv = "report.csv_1202"
LOG_file = "LOG_1202"


# from log_class import log_recorder
COMPACTION_LOG_HEAD = "/compaction/compaction_job.cc:755"
FLUSH_LOG_BEGIN = "flush_started"
FLUSH_LOG_END = "flush_finished"
FLUSH_FILE_CREATEION = "table_file_creation"

def load_log_and_qps(log_file, ground_truth_csv):
    # load the data
    return log_recorder(log_file,ground_truth_csv)

data_set = load_log_and_qps(LOG_DIR+LOG_file, LOG_DIR+report_csv)


ms_to_second=1000000
time_slice = 1000000
switch_ratio = ms_to_second / time_slice
real_time_speed=data_set.qps_df

# bucket = []
feature_columns = ["flushes","l0compactions","other_compactions","read","write"]
# feature_columns = ["l0compactions","other_compactions","read","write"]
example_row = [0,0,0,0,0] # flushes, l0compaction,other_compaction,read,write
bucket = np.zeros([int(real_time_speed.tail(1)["secs_elapsed"] * switch_ratio),len(feature_columns)], dtype = float) 
# for i in range(int(real_time_speed.tail(1)["secs_elapsed"] * switch_ratio)):
#     bucket.append(empty_tuple) # concurrent works,read MB/s, write MB/s,


for index, flush_job in data_set.flush_df.iterrows():
    flush_speed = round(flush_job["flush_size"] / (flush_job["end_time"] - flush_job["start_time"]),2) # bytes/ms , equals to MB/sec
    start_index = int(flush_job["start_time"]/time_slice)
    end_index=int(flush_job["end_time"]/time_slice) + 1
    if start_index >= len(bucket)-10 or end_index >= len(bucket)-5: # the tail part is not accurant
        break
    for element in bucket[start_index:end_index]:
        element[0]+=1
        element[-1]+=flush_speed


# then we use a bucket sort idea to count down the rest things
for index, compaction_job in data_set.compaction_df.iterrows():
    compaction_read_speed = round(compaction_job["input_data_size"] / (compaction_job["compaction_time_micros"]),2) # bytes/ms , equals to MB/sec
    compaction_write_speed = round(compaction_job["total_output_size"] / (compaction_job["compaction_time_micros"]),2) # bytes/ms , equals to MB/sec
    start_index = int(compaction_job["start_time"]/time_slice)
    end_index=int(compaction_job["end_time"]/time_slice) + 1

    if start_index >= len(bucket)-10 or end_index >= len(bucket)-5: # the tail part is not accurant
        break
    for element in bucket[start_index:end_index]:
        element[0]+=0
        if compaction_job["compaction_reason"] == "LevelL0FilesNum":
            element[1]+=1
        else:
            element[2]+=1
        element[-2]+=compaction_read_speed
        element[-1]+=compaction_write_speed

bucket_df = pd.DataFrame(bucket,columns=feature_columns)

bucket_df["qps"] = data_set.qps_df["interval_qps"]
bucket_df.to_csv("trainable_data.csv")

read = bucket_df["read"]
MAX_READ = 2000
bad_read = read >= MAX_READ
read[bad_read] = MAX_READ
plot_features = bucket_df[bucket_df.columns]

_ = plot_features.plot(subplots=True)
plt.savefig("operation_distribution.pdf")
plt.clf()


# separate the train set and eval set
column_indices = {name: i for i, name in enumerate(bucket_df.columns)}

df = bucket_df[200:1100]

n = len(df)
train_df = df[0:int(n*0.5)]
val_df = df[int(n*0.5):int(n*0.7)]

test_df = df[int(n*0.9):]


num_features = df.shape[1]
# normalized, z-score normalized
# train_mean = train_df.mean()
# train_std = train_df.std()

# train_df = (train_df - train_mean) / train_std
# val_df = (val_df - train_mean) / train_std
# test_df = (test_df - train_mean) / train_std
# df_std = (df - train_mean) / train_std
# df_std = df_std.melt(var_name='Column', value_name='Normalized')

def norm(x):
    return (x-df.min()) / (df.max()-df.min())

train_df = norm(train_df)
val_df = norm(val_df)
test_df = norm(test_df)


from WindowGenerator import WindowGenerator

# test of window splitting

w1 = WindowGenerator(input_width=24,label_width=1,shift=24,train_df=train_df, val_df=val_df, test_df=test_df,label_columns=["qps"])

example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
                           np.array(train_df[100:100+w1.total_window_size]),
                           np.array(train_df[200:200+w1.total_window_size])])
example_inputs, example_labels = w1.split_window(example_window)

# test of single step models
from models import *
linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)
])

# creating different windows here
single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['qps'])
conv_window = WindowGenerator(input_width=CONV_WIDTH,label_width=1,shift=1,train_df=train_df, val_df=val_df, test_df=test_df,label_columns=['qps'])

# wide part
wide_window = WindowGenerator(input_width=24, label_width=24, shift=1,train_df=train_df, val_df=val_df, test_df=test_df,label_columns=['qps'])


# performance and running 
history = compile_and_fit(linear, w1)
baseline = Baseline(label_index=column_indices['qps'])

baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}


val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)
# val_performance['Linear'] = linear.evaluate(w1.val)
# performance['Linear'] = linear.evaluate(w1.test, verbose=0)

# plt.bar(x = range(len(train_df.columns)),
#         height=linear.layers[0].kernel[:,0].numpy())
# axis = plt.gca()
# axis.set_xticks(range(len(train_df.columns)))
# _ = axis.set_xticklabels(train_df.columns, rotation=45)
# plt.savefig("linear_weight.pdf")
# plt.clf()


history = compile_and_fit(dense,single_step_window)
val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

# multi_step with conv_window
history = compile_and_fit(multi_step_dense, conv_window)

val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)
# conv model with conv window
history = compile_and_fit(conv_model, conv_window)

val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)

wide_conv_window = WindowGenerator(input_width=INPUT_WIDTH,label_width=LABEL_WIDTH,train_df=train_df, val_df=val_df, test_df=test_df, shift=1, label_columns=['qps'])

wide_window = WindowGenerator(input_width=60, label_width=60, shift=1,train_df=train_df, val_df=val_df, test_df=test_df,label_columns=['qps'])

# LSTMs
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(60, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

history = compile_and_fit(lstm_model, wide_window)

val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=1)

# Res LSTMs

residual_lstm = ResidualWrapper(
    tf.keras.Sequential([
    tf.keras.layers.LSTM(20, return_sequences=True),
    tf.keras.layers.Dense(
        num_features,
        # The predicted deltas should start small
        # So initialize the output layer with zeros
        kernel_initializer=tf.initializers.zeros)
]))
history = compile_and_fit(residual_lstm, wide_window)

val_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.val)
performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.test, verbose=0)



x = np.arange(len(performance))
width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [qps, normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(),
           rotation=45)
_ = plt.legend()

plt.savefig("performance_comparison.pdf")
plt.clf()

multi_window = WindowGenerator(input_width=24,
                               label_width=OUT_STEPS,train_df=train_df, val_df=val_df, test_df=test_df,
                               shift=OUT_STEPS)

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance = {}
multi_performance = {}

multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
multi_performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0)

repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose=0)

multi_linear_model = tf.keras.Sequential([
    # Take the last time-step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])


history = compile_and_fit(multi_linear_model, multi_window)

multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)

multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])
history = compile_and_fit(multi_dense_model, multi_window)

multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test, verbose=0)

multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Shape => [batch, 1,  out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_conv_model, multi_window)
multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)

multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features]
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

history = compile_and_fit(multi_lstm_model, multi_window)
multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)

# https://arxiv.org/abs/1308.0850 autotrgressive
class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps):
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

feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)
prediction, state = feedback_model.warmup(multi_window.example[0])

history = compile_and_fit(feedback_model, multi_window)
multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)

x = np.arange(len(multi_performance))
width = 0.3


metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(),
           rotation=45)
plt.ylabel(f'MAE (average over all times and outputs)')
_ = plt.legend()

plt.savefig("multi_step_performance_comparison.pdf")
plt.clf()