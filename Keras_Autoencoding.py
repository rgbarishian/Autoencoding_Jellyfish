#https://www.analyticsvidhya.com/blog/2018/06/unsupervised-deep-learning-computer-vision/
#https://blog.keras.io/building-autoencoders-in-keras.html
import os
import keras
import numpy as np
import pandas as pd
import keras.backend as K

from time import time
from sklearn.cluster import KMeans
from keras import callbacks
from keras.models import Model
from keras.optimizers import SGD
from keras.layers import Dense, Input
from keras.initializers import VarianceScaling
from keras.engine.topology import Layer, InputSpec
from keras.preprocessing.image import ImageDataGenerator

from scipy.misc import imread
from sklearn.metrics import accuracy_score, normalized_mutual_info_score

train_path = '/home/ryan/Documents/Galaxy_Zoo/Jell_Test/Grayscale/Train_Independent'
valid_path = '/home/ryan/Documents/Galaxy_Zoo/Jell_Test/Grayscale/Valid'
test_path = '/home/ryan/Documents/Galaxy_Zoo/Jell_Test/Test'

train_batches = ImageDataGenerator(rescale=1/255).flow_from_directory(train_path, target_size=(424,424), classes=['Not_Jellies','Poss_Jellies'], batch_size=35)
valid_batches = ImageDataGenerator(rescale=1/255).flow_from_directory(valid_path, target_size=(424,424), classes=['Not_Jellies','Poss_Jellies'], batch_size=11)
test_batches = ImageDataGenerator(rescale=1/255).flow_from_directory(test_path, target_size=(424,424), batch_size=15)

#preprocessing of images, flattening
#Can't use imagedatagenerator:
#TypeError: unsupported operand type(s) for /: 'DirectoryIterator' and 'int'

# train_x = train_batches/255.#255 brightness depth
# val_x = valid_batches/255.
# train_x = train_x.reshape(-1, 179776)#24^2
# val_x = val_x.reshape(-1, 179776)

print(len(train_batches))
print('1\n\n')
print(train_batches[1])
print('\n2\n\n')
print(train_batches[7])

# for i in train_batches:
#     print (i)
# this is our input placeholder
input_img = Input(shape=(179776,))

# "encoded" is the encoded representation of the input
encoded = Dense(2000, activation='relu')(input_img)
encoded = Dense(500, activation='relu')(encoded)
encoded = Dense(500, activation='relu')(encoded)
encoded = Dense(10, activation='sigmoid')(encoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(500, activation='relu')(encoded)
decoded = Dense(500, activation='relu')(decoded)
decoded = Dense(2000, activation='relu')(decoded)
decoded = Dense(784)(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

print(autoencoder.summary())

#  this model maps an input to its encoded representation
# encoder = Model(input_img, encoded)

# estop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

# train_history = autoencoder.fit(train_batches, epochs=50, batch_size=30, validation_data=valid_batches, callbacks=[estop])