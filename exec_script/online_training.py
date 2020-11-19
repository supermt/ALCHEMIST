import load_from_current

from river import compose
from river import time_series
from river import metrics
from river import preprocessing
from river import datasets
# model = compose.Pipeline(
#     preprocessing.MinMaxScaler(),
#     time_series.GroupDetrender(regressor=None,by="secs_elapsed",window_size=10)
# )


# metrics = metrics.Accuracy()

import pandas as pd


import math
import matplotlib.pyplot as plt
from river import compose
from river import datasets
from river import linear_model
from river import metrics
from river import optim
from river import preprocessing
from river import stats
from river import time_series
from river import stream


class timeonly_training_dataset(datasets.base.FileDataset):
    def __init__(self,input_filename):
        super().__init__(filename=input_filename,task=datasets.base.REG,n_features=1,n_samples=1440)

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target='interval_qps',
            converters={'interval_qps': int}
        )

def get_ordinal_date(x):
    return {'ordinal_date': int(x['secs_elapsed'])}


model = compose.Pipeline(
    ('ordinal_date', compose.FuncTransformer(get_ordinal_date)),
    ('scale', preprocessing.MinMaxScaler()),
    ('lin_reg', linear_model.LinearRegression())
)

from river import metrics
import matplotlib.pyplot as plt


# target_data = "../log_traces/Mixgraph/1000_0.0000073_45000/report.csv"
target_data = "../log_traces/StorageMaterial.NVMeSSD/12CPU/64MB/report.csv_1180"
import os
target_data = os.path.abspath(target_data)

def evaluate_model(model): 
    
    metric = metrics.Rolling(metrics.MAE(), 12)
    
    # dates = []
    y_trues = []
    y_preds = []

    for x, y in timeonly_training_dataset(target_data):
        
        # Obtain the prior prediction and update the model in one go
        y_pred = model.predict_one(x)
        model.learn_one(x, y)
        
        # Update the error metric
        metric.update(y, y_pred)
        
        # Store the true value and the prediction
        # dates.append(x['secs_elapsed'])
        y_trues.append(y)
        y_preds.append(y_pred)
        
    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(alpha=0.75)
    ax.plot(y_trues, lw=3, color='#2ecc71', alpha=0.8, label='Ground truth')
    ax.plot(y_preds, lw=3, color='#e74c3c', alpha=0.8, label='Prediction')
    ax.legend()
    ax.set_title(metric)
    plt.show()

model = compose.Pipeline(
    ('ordinal_date', compose.FuncTransformer(get_ordinal_date)),
    ('scale', preprocessing.MinMaxScaler()),
    ('lin_reg', linear_model.LinearRegression(intercept_lr=0,optimizer=optim.SGD(0.03))),
)

model = time_series.Detrender(regressor=model, window_size=10)

evaluate_model(model)