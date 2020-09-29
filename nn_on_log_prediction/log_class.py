import pandas as pd

import re

import datetime
import time
import json

MS_TO_SEC = 1000000


class log_recorder:
    start_time_micros = 0
    log_lines = []
    flush_df = pd.DataFrame(
        columns=["job", "start_time", "end_time", "flush_size"])
    # compaction_df = pd.DataFrame(
    #     columns=["job", "start_time", "end_time", "input_data_size","compaction_time_cpu_micros","total_output_size"])
    compaction_df = pd.DataFrame()
    qps_df = pd.DataFrame()

    def get_start_time(self, log_line):
        regex = r"\d+\/\d+\/\d+-\d+:\d+:\d+\.\d+"
        matches = re.search(regex, log_line, re.MULTILINE)
        machine_start_time = matches.group(0)
        start_time = datetime.datetime.strptime(
            machine_start_time, "%Y/%m/%d-%H:%M:%S.%f")
        start_time_micros = int(time.mktime(
            start_time.timetuple())) * MS_TO_SEC + start_time.microsecond
        return start_time_micros

    def pair_the_flush_jobs(self):
        flush_start_array = [
            x for x in self.log_lines if x["event"] == "flush_started"]
        # job_id = [x["job"] for x in flush_start_array]
        # print(job_id)
        flush_end_array = [
            x for x in self.log_lines if x["event"] == "flush_finished"]
        # flush_df = pd.DataFrame(["job","start_time","end_time","flush_size"])
        for start_event, index in zip(flush_start_array, range(len(flush_start_array))):
            self.flush_df.loc[index] = [start_event["job"],
                                        start_event['time_micros'] -
                                        self.start_time_micros,
                                        flush_end_array[index]["time_micros"]-self.start_time_micros, start_event["total_data_size"]]
        # print(self.flush_df)

    def get_the_compaction_jobs(self):

        # unlike flush, the compaction processes can be run in parallel,
        # which means one compaction that starts later can be finished eariler
        # so we need to sort it by the time_micros first
        compaction_start_df = pd.DataFrame(
            [x for x in self.log_lines if x["event"] == "compaction_started"]).sort_values("job")
        compaction_end_df = pd.DataFrame(
            [x for x in self.log_lines if x["event"] == "compaction_finished"]).sort_values("job")
        # choose the useful columns only
        compaction_start_df = compaction_start_df[[
            "time_micros", "input_data_size", "job","compaction_reason"]]
        compaction_end_df = compaction_end_df[[
            "time_micros", "compaction_time_micros", "compaction_time_cpu_micros", "total_output_size", ]]
        compaction_start_df["time_micros"] -= self.start_time_micros
        compaction_end_df["time_micros"] -= self.start_time_micros

        # let the time_micros minus the start time,

        compaction_start_df = compaction_start_df.rename(
            columns={"time_micros": "start_time"})

        compaction_end_df = compaction_end_df.rename(
            columns={"time_micros": "end_time"})
        # concat the data frames
        self.compaction_df = pd.concat(
            [compaction_start_df, compaction_end_df], axis=1)
        pass

    def record_real_time_qps(self, record_file):
        self.qps_df = pd.read_csv(record_file)
        pass

    def __init__(self, log_file, record_file=""):
        file_lines = open(log_file, "r").readlines()
        self.start_time_micros = self.get_start_time(file_lines[0])
        for line in file_lines:
            line = re.search('(\{.+\})', line)
            if line:
                log_row = json.loads(line[0])
                self.log_lines.append(log_row)
        self.pair_the_flush_jobs()
        self.get_the_compaction_jobs()
        if record_file != "":
            self.record_real_time_qps(record_file)
