"""Neural-network architecture for MNIST classification

The functions in this module allow to train a neural network based on dense layers

Author: Sergio P. Perez
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax
from tensorflow.keras.utils import to_categorical

#####################################################################################
#
# FUNCTION: temporal_loop
#
#####################################################################################

def training(train_images, train_labels):
    """Training and definition of the neural network

    Args:
        train_images: training set of images
        train_labels: labels of the training images
    Returns: 
        model: tensorflow trained model object
        pd.DataFrame(history.history): training history
    """

    # Build the model
    model = Sequential([
      Dense(64, activation='relu', input_shape=(784,)), 
      Dense(64, activation='relu'),  
      Dense(10, activation='softmax'), 
    ])
    
    # Compile the model
    model.compile(
      optimizer='adam',    
      loss='categorical_crossentropy', 
      metrics=['accuracy'], 
    )
    
    # Train the model
    history = model.fit( 
      train_images,
      to_categorical(train_labels),
      epochs=8, 
      batch_size=32,  
    )
    
    # Evaluate the model
#    model.evaluate(
#     test_images,
#     to_categorical(test_labels)
#    )
    
    return model, pd.DataFrame(history.history)



