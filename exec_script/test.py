import load_from_current

from module_wrapper.Trainer.NN_model_compiler import model_wrapper
from module_wrapper.Trainer.WindowGenerator import WindowGenerator
from module_wrapper.Trainer.models import *

LOG_DIR = "../log_traces/Mixgraph/1000_0.000073_450000000/"
report_csv = "report.csv"
LOG_file = "LOG"

trainer = model_wrapper(LOG_DIR, LOG_file, report_csv,
                        (100, 1100), (2000, 2000))

one_step_window_qps_only = WindowGenerator(input_width=1, label_width=1, shift=1,
                                  train_df=trainer.train_df, val_df=trainer.val_df, test_df=trainer.test_df, label_columns=["qps"])

forecasting_period = 60
out_steps = 30
range_window_qps = WindowGenerator(input_width=30, label_width=30, shift=out_steps,
                               train_df=trainer.train_df, val_df=trainer.val_df, test_df=trainer.test_df,label_columns=["qps"])

range_window_IO_status = WindowGenerator(input_width=forecasting_period, label_width=out_steps, shift=out_steps,
                               train_df=trainer.train_df, val_df=trainer.val_df, test_df=trainer.test_df,label_columns=["read","write"])
forecasting_period = 30
out_steps = 5
AR_range_window = WindowGenerator(input_width=forecasting_period, label_width=out_steps, shift=out_steps,
                               train_df=trainer.train_df, val_df=trainer.val_df, test_df=trainer.test_df,label_columns=None)


dense = Single_step_dense_net([64, 64, 1])
IO_status_dense_range_oneshot = One_shot_range_multi_feature_dense_net(out_steps,2)
qps_dense_range_oneshot = One_shot_range_multi_feature_dense_net(out_steps,1)

IO_status_multi_linear_range = Multi_linear_range_multi_feature_dense_net(out_steps,2)
qps_multi_linear_range = Multi_linear_range_multi_feature_dense_net(out_steps,1)

# qps_lstm_range = LSTM_range_multi_feature_dense_net(out_steps,1)
# print("start testing")
trainer.set_model_list([dense], one_step_window_qps_only)
# trainer.set_model_list([IO_status_dense_multi_shot],range_window_IO_status)
# trainer.set_model_list([qps_dense_range_oneshot],range_window_qps)
# trainer.set_model_list([IO_status_multi_linear_range],range_window_IO_status)
# trainer.set_model_list([qps_multi_linear_range],range_window_qps)
# trainer.set_model_list([qps_lstm_range],range_window_qps)


# feedback_model_qps = AutoRegressionLSTM(out_steps,13,AR_range_window)
# trainer.set_model_list([feedback_model_qps],AR_range_window)

print("start compiling and fitting")
trainer.compile_and_run_all()
