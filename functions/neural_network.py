"""Neural-network architecture for MNIST classification

The functions in this module allow to train a neural network based on dense layers

Author: Sergio P. Perez
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Softmax, Layer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model



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
    
    return model, pd.DataFrame(history.history)



#####################################################################################
#
# CLASS: DenseLayer
#
#####################################################################################


class DenseLayer(Layer):

    def __init__(self, units, input_dim):
        super(DenseLayer, self).__init__()
        self.w = self.add_weight(shape=(input_dim, units),
                             initializer='random_normal')
        self.b = self.add_weight(shape=(units,),
                             initializer='zeros')
    def call(self, inputs):
        return tf.matmul(inputs, self.w)+self.b
    

class MyModel(Model):

    def __init__(self, units_1, input_dim_1, units_2, units_3):
        super(MyModel, self).__init__()
        self.layer_1 = DenseLayer(units_1, input_dim_1)
        self.layer_2 = DenseLayer(units_2, units_1)
        self.layer_3 = DenseLayer(units_3, units_2)
        self.softmax = Softmax()

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = tf.nn.relu(x)
        x = self.layer_2(x)
        x = tf.nn.relu(x)
        x = self.layer_3(x)
        return self.softmax(x)

@tf.function
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)

def loss(model, x, y):
    y_ = model(x)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    return loss_object(y_true=y, y_pred=y_)



def customized_training(train_images, train_labels):
    
    model = MyModel(64, 784, 64, 10)
    model(tf.ones((1, 784)))
    model.summary()
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
    train_dataset = train_dataset.batch(32)
    
    # Keep results for plotting
    train_loss_results = []
    train_accuracy_results = []
    
    num_epochs = 5
    
    # Measure the training time
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
     
        # Training loop
        for x, y in train_dataset:
            # Optimize the model
            loss_value, grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
            # Compute current loss
            epoch_loss_avg(loss_value)  
            # Compare predicted label to actual label
            epoch_accuracy(to_categorical(y), model(x))
    
      # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())
    
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                      epoch_loss_avg.result(),
                                                                      epoch_accuracy.result()))
        
    print("Duration :{:.3f}".format(time.time() - start_time))
    
    
    
    return model


def customized_validation(test_images, test_labels, model):
    
    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    test_dataset = test_dataset.batch(32)
    
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
    
    # Test loop
    for x, y in test_dataset:
        # Optimize the model
        loss_value = loss(model, x, y)
        
        # Compute current loss
        epoch_loss_avg(loss_value)  
        # Compare predicted label to actual label
        epoch_accuracy(to_categorical(y), model(x))
    
    # End epoch
    print("Test loss: {:.3f}".format(epoch_loss_avg.result().numpy()))
    print("Test accuracy: {:.3%}".format(epoch_accuracy.result().numpy()))
    

