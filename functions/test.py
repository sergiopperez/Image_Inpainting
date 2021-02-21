#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 11:36:54 2021

@author: sergioperez
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import plots
import finite_volumes as fv



import mnist
test_images = mnist.test_images()
test_images = test_images.reshape((-1,784))
test_images = (test_images / 255) *2-1
example = test_images[0,:]

#plots.plot_image(example)


intensity = 0.3

damage = np.random.choice(np.arange(example.size), replace=False, size=int(example.size * intensity))
example[damage] = 0
#plots.plot_image(example)

restored_example = fv.temporal_loop(example, damage)

plots.plot_image(restored_example)

# %%

import mnist
import neural_network as nn

train_images = mnist.train_images()
train_labels = mnist.train_labels()

# Normalize the test image as CH Equation work for (-1,1)
train_images = (train_images / 255) *2-1

# Flatten the test images
train_images = train_images.reshape((-1,784))

model, history = nn.training(train_images, train_labels)

# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical

#Loss plot

acc_plot = history.plot(y="loss", title = "Loss vs. Epochs",legend=False)
acc_plot.set(xlabel="Epochs", ylabel="Loss")

#Accuracy plot

acc_plot = history.plot(y="accuracy", title="Accuracy vs. Epochs", legend=False)
acc_plot.set(xlabel="Epochs", ylabel="Accuracy")



test_images = mnist.test_images()
test_labels = mnist.test_labels()

# Normalize the test image as CH Equation work for (-1,1)
test_images = (test_images / 255) *2-1

# Flatten the test images
test_images = test_images.reshape((-1,784))
test_loss, test_accuracy = model.evaluate(test_images, to_categorical(test_labels), verbose=2)



# %%

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import plots
import finite_volumes as fv



import mnist

n_images = 5
indices_images = range(n_images)

test_images = mnist.test_images()
test_images = test_images.reshape((-1,784))
test_images = (test_images / 255) *2-1
example = test_images[indices_images,:]


intensity = 0.3

damage = np.zeros((len(indices_images), int(example.shape[1] * intensity)), dtype=int)

for i in indices_images:

    
    damage[i, :] = np.random.choice(np.arange(example.shape[1]), replace=False, size=int(example.shape[1] * intensity))
#    print(damage[i, :])
    example[i, damage[i, :]] = 0
    
restored_example = np.zeros(example.shape)

for i in range(n_images):
    restored_example[i,:] = fv.temporal_loop(example[i, :], damage[i, :])
    

# %%
  
predictions_damaged = np.argmax(model.predict(example), axis=1)  
predictions_restored = np.argmax(model.predict(restored_example), axis=1)    
    
print(predictions_damaged)
print(predictions_restored)
    