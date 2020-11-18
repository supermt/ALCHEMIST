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

def get_ordinal_date(x):
    return {'secs_elapsed': int(x['secs_elapsed'])}    


def make_model(alpha):
    
    extract_features = compose.TransformerUnion(get_ordinal_date)

    scale = preprocessing.StandardScaler()

    learn = linear_model.LinearRegression(
        intercept_lr=0,
        optimizer=optim.SGD(3),
        loss=optim.losses.Quantile(alpha=alpha)
    )

    model = extract_features | scale | learn
    model = time_series.Detrender(regressor=model, window_size=12)

    return model

metric = metrics.MAE()

models = {
    'lower': make_model(alpha=0.05),
    'center': make_model(alpha=0.5),
    'upper': make_model(alpha=0.95)
}

dates = []
y_trues = []
y_preds = {
    'lower': [],
    'center': [],
    'upper': []
}


class online_training_dataset(datasets.base.FileDataset):
    def __init__(self,input_filename):
        super().__init__(filename=input_filename,task=datasets.base.REG,n_features=1,n_samples=1440)

    def __iter__(self):
        return stream.iter_csv(
            self.path,
            target='interval_qps',
            converters={'interval_qps': int}
        )

# for x,y in online_training_dataset():
#     print(x,y)

target_data = "../log_traces/Mixgraph/1000_0.0000073_45000/report.csv"
import os
target_data = os.path.abspath(target_data)
for x, y in online_training_dataset(target_data):
    y_trues.append(y)
    dates.append(int(x['secs_elapsed']))
    
    for name, model in models.items():
        y_preds[name].append(model.predict_one(x))
        model.learn_one(x, y)

    # Update the error metric
    metric.update(y, y_preds['center'][-1])

# Plot the results
fig, ax = plt.subplots(figsize=(10, 6))
ax.grid(alpha=0.75)
ax.plot(dates, y_trues, lw=3, color='#2ecc71', alpha=0.8, label='Truth')
ax.plot(dates, y_preds['center'], lw=3, color='#e74c3c', alpha=0.8, label='Prediction')
ax.fill_between(dates, y_preds['lower'], y_preds['upper'], color='#e74c3c', alpha=0.3, label='Prediction interval')
ax.legend()
ax.set_title(metric);
plt.savefig("online_predicting")
plt.clf()