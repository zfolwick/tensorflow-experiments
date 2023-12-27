import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# get the data
mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape)

# normalize so vals are 0 < data < 1

x_train, x_test = x_train / 255, x_test / 255

def display(quantity):
    for i in range(quantity):
        plt.subplot(2, 3, i+1)
        plt.imshow(x_train[i], cmap='gray')
    plt.show()
    
# display(6)

def create_model():
    model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    # 
    keras.layers.Dense(128, activation='relu'),
    # 10 different classes
    keras.layers.Dense(10),
])
    return model

model = create_model()

# print(model.summary())

# loss and optimizer
# y = 0, y = [1, 0,0,0,0,0,0,0,0,0,0]
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# adam is a popular optimizer.  why?
optim = keras.optimizers.Adam(lr=0.001)
metrics=["accuracy"]

model.compile(loss=loss, optimizer=optim, metrics=metrics)

#training
batch_size=64 # how many pics at once?
epochs=5   # how many times to run through a batch size?
# output:
# 938/938 - 3s - loss: 0.2998 - accuracy: 0.9158 - 3s/epoch - 3ms/step
# Epoch 2/5
# 938/938 - 2s - loss: 0.1327 - accuracy: 0.9618 - 2s/epoch - 2ms/step
# Epoch 3/5
# 938/938 - 2s - loss: 0.0931 - accuracy: 0.9725 - 2s/epoch - 2ms/step
# Epoch 4/5
# 938/938 - 2s - loss: 0.0717 - accuracy: 0.9792 - 2s/epoch - 2ms/step
# Epoch 5/5
# 938/938 - 2s - loss: 0.0556 - accuracy: 0.9840 - 2s/epoch - 2ms/step
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

# evaluate
model.evaluate(x_test, y_test, batch_size=batch_size, verbose=2)

# predictions
#method 1
probability_model = keras.models.Sequential([
    model,
    keras.layers.Softmax()
])

predictions = probability_model(x_test)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)

# method 2: model + softmax
predictions = model(x_test)
predictions = tf.nn.softmax(predictions)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)

# method 3: 
predictions = model.predict(x_test, batch_size=batch_size)
predictions = tf.nn.softmax(predictions)
pred0 = predictions[0]
print(pred0)
label0 = np.argmax(pred0)
print(label0)

pred05s = predictions[0:5]
print(pred05s.shape)
label05s = np.argmax(pred05s, axis=1)
print(label05s)