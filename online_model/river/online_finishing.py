from river import datasets

dataset = datasets.Phishing()

from river import compose 
from river import linear_model
from river import metrics
from river import preprocessing

model = compose.Pipeline(preprocessing.StandardScaler(),linear_model.LogisticRegression())
metric = metrics.Accuracy()

for x,y in dataset:
  y_pred = model.predict_one(x)
  metric = metric.update(y,y_pred)
  model = model.learn_one(x,y)

print(metric)
