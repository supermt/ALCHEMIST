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

bucket_df = pd.DataFrame(bucket,columns=feature_columns).round(1)

bucket_df["qps"] = data_set.qps_df["interval_qps"]
bucket_df.to_csv("trainable_data.csv")
