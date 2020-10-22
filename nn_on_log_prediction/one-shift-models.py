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

val_performance = {}
performance = {}

import time

print("Start single shift, single step training")
start = time.time_ns()
history = compile_and_fit(dense,single_step_window)
end = time.time_ns()
print("Single shift, single step training time cost: %.2f s"% (float(end-start)/1000000000))

val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)
# print(performance[;])
print("Single shift, single step mae loss: %.4f" % performance['Dense'][1])

# multi_step with conv_window

print("Start single shift, three step training")
start = time.time_ns()
history = compile_and_fit(multi_step_dense, conv_window)
end = time.time_ns()
print("Single shift, three step training time cost: %.2f s"% (float(end-start)/1000000000))


val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test, verbose=0)
print("Single shift, three step mae loss: %.4f" % performance['Multi step dense'][1])

# conv model with conv window
print("Start single shift, Conv training")
start = time.time_ns()
history = compile_and_fit(conv_model, conv_window)
end = time.time_ns()
print("Single shift, Conv training time cost: %.2f s"% (float(end-start)/1000000000))

val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)
print("Single shift, three step mae loss: %.4f" % performance['Conv'][1])


wide_window = WindowGenerator(input_width=60, label_width=60, shift=1,train_df=train_df, val_df=val_df, test_df=test_df,label_columns=['qps'])

# LSTMs
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(60, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

residual_lstm = ResidualWrapper(
    tf.keras.Sequential([
    tf.keras.layers.LSTM(60, return_sequences=True),
    tf.keras.layers.Dense(
        num_features,
        # The predicted deltas should start small
        # So initialize the output layer with zeros
        kernel_initializer=tf.initializers.zeros)
]))


print("Start single shift, LSTM training")
start = time.time_ns()
history = compile_and_fit(lstm_model, wide_window)
end = time.time_ns()
print("Single shift, LSTM training time cost: %.2f s"% (float(end-start)/1000000000))


val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=1)
print("Single shift, LSTM mae loss: %.4f" % performance['LSTM'][1])

# Res LSTMs
print("Start single shift, LSTM training")
start = time.time_ns()
history = compile_and_fit(residual_lstm, wide_window)
end = time.time_ns()
print("Single shift, LSTM training time cost: %.2f s"% (float(end-start)/1000000000))

val_performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.val)
performance['Residual LSTM'] = residual_lstm.evaluate(wide_window.test, verbose=0)
print("Single shift, LSTM mae loss: %.4f" % performance['Residual LSTM'][1])


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
