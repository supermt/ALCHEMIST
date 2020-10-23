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
from feature_selection import action_list_feature_vectorize

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

time_slice = 1000000

bucket_df = action_list_feature_vectorize(data_set,time_slice)

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

def norm(x):
    return (x-df.min()) / (df.max()-df.min())

train_df = norm(train_df)
val_df = norm(val_df)
test_df = norm(test_df)

# import tensorflow as tf

# from WindowGenerator import WindowGenerator
# from models import *

# multi_lstm_model = tf.keras.Sequential([
#     # Shape [batch, time, features] => [batch, lstm_units]
#     # Adding more `lstm_units` just overfits more quickly.
#     tf.keras.layers.LSTM(32, return_sequences=False),
#     # Shape => [batch, out_steps*features]
#     tf.keras.layers.Dense(OUT_STEPS*num_features,
#                           kernel_initializer=tf.initializers.zeros),
#     # Shape => [batch, out_steps, features]
#     tf.keras.layers.Reshape([OUT_STEPS, num_features])
# ])

# multi_window = WindowGenerator(input_width=OUT_STEPS,
#                                label_width=OUT_STEPS,train_df=train_df, val_df=val_df, test_df=test_df,
#                                shift=OUT_STEPS)

# last_baseline = MultiStepLastBaseline()
# last_baseline.compile(loss=tf.losses.MeanSquaredError(),
#                       metrics=[tf.metrics.MeanAbsoluteError()])

# multi_val_performance = {}
# multi_performance = {}

# multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
# multi_performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0)

# repeat_baseline = RepeatBaseline()
# repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
#                         metrics=[tf.metrics.MeanAbsoluteError()])

# multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
# multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose=0)

# multi_linear_model = tf.keras.Sequential([
#     # Take the last time-step.
#     # Shape [batch, time, features] => [batch, 1, features]
#     tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
#     # Shape => [batch, 1, out_steps*features]
#     tf.keras.layers.Dense(OUT_STEPS*num_features,
#                           kernel_initializer=tf.initializers.zeros),
#     # Shape => [batch, out_steps, features]
#     tf.keras.layers.Reshape([OUT_STEPS, num_features])
# ])



# history = compile_and_fit(multi_linear_model, multi_window)

# multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
# multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test, verbose=0)

# multi_dense_model = tf.keras.Sequential([
#     # Take the last time step.
#     # Shape [batch, time, features] => [batch, 1, features]
#     tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
#     # Shape => [batch, 1, dense_units]
#     tf.keras.layers.Dense(512, activation='sigmoid'),
#     # Shape => [batch, out_steps*features]
#     tf.keras.layers.Dense(OUT_STEPS*num_features,
#                           kernel_initializer=tf.initializers.zeros),
#     # Shape => [batch, out_steps, features]
#     tf.keras.layers.Reshape([OUT_STEPS, num_features])
# ])
# history = compile_and_fit(multi_dense_model, multi_window)

# multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
# multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test, verbose=0)

# multi_conv_model = tf.keras.Sequential([
#     # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
#     tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
#     # Shape => [batch, 1, conv_units]
#     tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
#     # Shape => [batch, 1,  out_steps*features]
#     tf.keras.layers.Dense(OUT_STEPS*num_features,
#                           kernel_initializer=tf.initializers.zeros),
#     # Shape => [batch, out_steps, features]
#     tf.keras.layers.Reshape([OUT_STEPS, num_features])
# ])

# history = compile_and_fit(multi_conv_model, multi_window)
# multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
# multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test, verbose=0)


# history = compile_and_fit(multi_lstm_model, multi_window)
# multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
# multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=0)

# feedback_model = FeedBack(units=32, out_steps=OUT_STEPS,num_features=num_features)
# prediction, state = feedback_model.warmup(multi_window.example[0])

# history = compile_and_fit(feedback_model, multi_window)
# multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
# multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test, verbose=0)

# x = np.arange(len(multi_performance))
# width = 0.3

# metric_name = 'mean_absolute_error'
# metric_index = multi_dense_model.metrics_names.index('mean_absolute_error')
# val_mae = [v[metric_index] for v in multi_val_performance.values()]
# test_mae = [v[metric_index] for v in multi_performance.values()]

# plt.bar(x - 0.17, val_mae, width, label='Validation')
# plt.bar(x + 0.17, test_mae, width, label='Test')
# plt.xticks(ticks=x, labels=multi_performance.keys(),
#            rotation=45)
# plt.ylabel(f'MAE (average over all times and outputs)')
# _ = plt.legend()

# plt.savefig("multi_step_performance_comparison.pdf")
# plt.clf()