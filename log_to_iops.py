import itertools
import pandas as pd
import re
from log_class import log_recorder

LOG_DIR = "./test_data/"
report_csv = "report.csv_1200"
LOG_file = "LOG_1200"


# from log_class import log_recorder
COMPACTION_LOG_HEAD = "/compaction/compaction_job.cc:755"
FLUSH_LOG_BEGIN = "flush_started"
FLUSH_LOG_END = "flush_finished"
FLUSH_FILE_CREATEION = "table_file_creation"



def load_log_and_qps(log_file, ground_truth_csv):
    # load the data
    return log_recorder(log_file,ground_truth_csv)

if __name__ == "__main__":
    data_set = load_log_and_qps(LOG_DIR+LOG_file, LOG_DIR+report_csv)
    print(data_set.compaction_df.head())