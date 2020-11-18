import pandas as pd
from ..log_load_utils.log_class import log_recorder
from .feature_selection import action_list_feature_vectorize
from .MODEL_PARAMETERS import *
from .models import *
from .WindowGenerator import WindowGenerator

import os
import datetime
import time

import matplotlib as mpldata_cleaning
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class model_wrapper():
    model_list = {}

    @staticmethod
    def load_log_and_qps(log_file, ground_truth_csv):
        # load the data
        return log_recorder(log_file, ground_truth_csv)

    def split_dataset_and_normalize(self):
        n = len(self.df)
        train_df = self.df[0:int(n*0.5)]
        val_df = train_df
        # val_df = self.df[int(n*0.5):int(n*0.7)]
        test_df = self.df[int(n*0.5):]
        self.num_features = self.df.shape[1]
        # the normalize function, use the min-max scale

        def norm(x):
            return (x-self.df.min()) / (self.df.max()-self.df.min())

        self.train_df = norm(train_df).fillna(0)
        self.val_df = norm(val_df).fillna(0)
        self.test_df = norm(test_df).fillna(0)

    def load_bucket_df(self):
        self.bucket_df = action_list_feature_vectorize(
            self.data_set, self.time_slice)
        self.bucket_df["qps"] = self.data_set.qps_df["interval_qps"]
        # bucket_df.to_csv("trainable_data.csv")

    def data_cleaning(self):
        read = self.bucket_df["read"]
        write = self.bucket_df["write"]
        bad_read = read >= self.MAX_READ
        read[bad_read] = self.MAX_READ
        bad_write = write >= self.MAX_WRITE
        write[bad_write] = self.MAX_WRITE

        plot_features = self.bucket_df[self.bucket_df.columns]
        _ = plot_features.plot(subplots=True)
        plt.savefig("../image_results/"+self.config_set +
                    "operation_distribution.pdf")
        plt.clf()

    def assign_the_one_step_window(self, input_width, label_width, shift, label_columns=["qps"]):
        self.one_step_window = WindowGenerator(input_width=input_width, label_width=label_width, shift=shift,
                                               train_df=self.train_df, val_df=self.val_df, test_df=self.test_df, label_columns=label_columns)

    def assign_the_multi_step_window(self, input_width, OUT_STEPS, label_columns=["qps"]):
        self.multiple_steps_window = WindowGenerator(input_width=input_width,
                                                     label_width=OUT_STEPS, train_df=self.train_df, val_df=self.val_df, test_df=self.test_df,
                                                     shift=OUT_STEPS, label_columns=label_columns)

    def set_model_list(self, model_list=[], trainer_window=None):
        if trainer_window not in self.model_list:
            self.model_list[trainer_window] = []
        self.model_list[trainer_window].extend(model_list)

    def compile_and_run_all(self):
        init_map = {"eval_performance": {}, "test_performance": {}}
        for window_type in self.model_list:
            for training_model in self.model_list[window_type]:
                print("start",training_model)
                start = time.time_ns()
                history = compile_and_fit(training_model.net,window_type)
                end = time.time_ns()
                print("end tarining, time cost %.2f sec" % (float(start-end)/ 1000000000))
                init_map["eval_performance"][training_model] = training_model.net.evaluate(window_type.val)
                init_map["eval_performance"][training_model] = training_model.net.evaluate(window_type.test,verbose=0)
                window_type.plot()
                window_type.plot(training_model.net)
                plt.savefig(str(window_type).replace("\n","_")+".pdf")
                plt.clf()

    def __init__(self, LOG_DIR, LOG_file, report_csv, sequence_time_index, device_boundaries):
        self.config_set = LOG_DIR.replace("/", "_")
        self.data_set = model_wrapper.load_log_and_qps(
            LOG_DIR+LOG_file, LOG_DIR+report_csv)
        sequence_time_index = (200, 1000)
        self.time_slice = 1000000
        self.sequence_time_index = sequence_time_index
        self.bucket_df = []
        self.MAX_READ = device_boundaries[0]
        self.MAX_WRITE = device_boundaries[1]

        self.load_bucket_df()
        self.df = self.bucket_df[sequence_time_index[0]:sequence_time_index[1]]

        self.split_dataset_and_normalize()
        self.data_cleaning()
