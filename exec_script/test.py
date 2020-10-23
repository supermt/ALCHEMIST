import load_from_current

from module_wrapper.Trainer.NN_model_compiler import model_wrapper
from module_wrapper.Trainer.WindowGenerator import WindowGenerator
from module_wrapper.Trainer.models import *

LOG_DIR = "../log_traces/StorageMaterial.NVMeSSD/8CPU/128MB/"
report_csv = "report.csv_1202"
LOG_file = "LOG_1202"

trainer = model_wrapper(LOG_DIR,LOG_file,report_csv,(200,1000),(2000,2000))
trainer.assign_the_one_step_window(1,1,1,label_columns=["qps"])

trainer.assign_the_multi_step_window(24,5,label_columns=None)

dense = Single_shot_single_step_dense_net([64,64,1])
# multi_dense = Multi_step_dense_net([64,64,1])
# cnn = Single_step_CNN([64,64,1])

# compile_and_fit(dense.net,trainer.one_step_window)

# print("start testing")
trainer.set_model_list([dense])
print("start compiling and fitting")
trainer.compile_and_run_all()