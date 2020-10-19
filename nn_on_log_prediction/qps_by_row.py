
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

dataset = pd.read_csv("trainable_data.csv")

train_dataset= dataset[0:800]
test_dataset = dataset[0:800]

print("training --------------------- one row to one qps")


train_stats = train_dataset.describe().pop("qps").transpose()

# train_stats = train_dataset.describe().transpose()
train_labels = train_dataset.pop("qps")
test_labels = test_dataset.pop("qps")

def norm(x):
    return x
#   return (x - train_stats['mean']) / train_stats['std']
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def one_row_one_qps_model():
  model = keras.Sequential([
    layers.Dense(5, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(5, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = one_row_one_qps_model()

EPOCHS = 1000
def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [qps]')
  plt.plot(hist['epoch'], hist['mae'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mae'],
           label = 'Val Error')
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$qps^2$]')
  plt.plot(hist['epoch'], hist['mse'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mse'],
           label = 'Val Error')
  plt.legend()
  plt.show()

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(train_dataset, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=0, callbacks=[early_stop])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
plot_history(history)
# print(hist)

test_predictions = model.predict(normed_test_data).flatten()
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [qps]')
plt.ylabel('Predictions [qps]')
plt.axis('equal')
plt.axis('square')
plt.show()