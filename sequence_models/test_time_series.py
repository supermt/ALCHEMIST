
import itertools
import pandas as pd
import re
from log_class import log_recorder
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

LOG_DIR = "./StorageMaterial.NVMeSSD/8CPU/128MB/"
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


zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True)
csv_path, _ = os.path.splitext(zip_path)


ms_to_second=1000000
time_slice = 1000000
switch_ratio = ms_to_second / time_slice
real_time_speed=data_set.qps_df

bucket = []
bucket = np.empty([3,2], dtype = float) 
# for i in range(int(real_time_speed.tail(1)["secs_elapsed"] * switch_ratio)):
#     bucket.append(empty_tuple) # concurrent works,read MB/s, write MB/s,


for index, flush_job in data_set.flush_df.iterrows():
    flush_speed = round(flush_job["flush_size"] / (flush_job["end_time"] - flush_job["start_time"]),2) # bytes/ms , equals to MB/sec
    start_index = int(flush_job["start_time"]/time_slice)
    end_index=int(flush_job["end_time"]/time_slice) + 1

    if start_index >= len(bucket)-10 or end_index >= len(bucket)-5: # the tail part is not accurant
        break
    if index > 5:
        break
    print(1)



# # then we use a bucket sort idea to count down the rest things
# for index, compaction_job in data_set.compaction_df.iterrows():
#     compaction_read_speed = round(compaction_job["input_data_size"] / (compaction_job["compaction_time_micros"]),2) # bytes/ms , equals to MB/sec
#     compaction_write_speed = round(compaction_job["total_output_size"] / (compaction_job["compaction_time_micros"]),2) # bytes/ms , equals to MB/sec
#     bucket[index].append(flush_speed)
#     start_index = int(compaction_job["start_time"]/time_slice)
#     end_index=int(compaction_job["end_time"]/time_slice) + 1

#     if start_index >= len(bucket)-10 or end_index >= len(bucket)-5: # the tail part is not accurant
#         break
#     for bucket_element in bucket[start_index:end_index]:
#         bucket_element[1]+=1
#         bucket_element[2]+=compaction_read_speed
#         bucket_element[3]+=compaction_write_speed


# # df = pd.DataFrame(,columns=["flushes","compactions","read","write"])


