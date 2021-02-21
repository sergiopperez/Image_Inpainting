import matplotlib.pyplot as plt
import numpy as np

def plot_image(phi):

    if phi.shape == (28,28):

        plt.imshow(np.reshape(phi,(28,28)), cmap='Greys', vmin=-1, vmax=1, extent=[0,100,0,1], aspect=100)
        plt.axis('off')

    elif phi.shape == (784,):

        plt.imshow(np.reshape(phi,(28,28)), cmap='Greys', vmin=-1, vmax=1, extent=[0,100,0,1], aspect=100)
        plt.axis('off')
