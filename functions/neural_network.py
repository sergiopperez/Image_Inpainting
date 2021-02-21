import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax
from tensorflow.keras.utils import to_categorical

def training(train_images, train_labels):

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
      epochs=5, 
      batch_size=32,  
    )
    
    # Evaluate the model
#    model.evaluate(
#     test_images,
#     to_categorical(test_labels)
#    )
    
    return model, pd.DataFrame(history.history)


# Save the model
# model.save_weights('model.h5')
