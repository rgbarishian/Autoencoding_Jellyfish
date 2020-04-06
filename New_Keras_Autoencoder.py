# https://blog.keras.io/building-autoencoders-in-keras.html
 #https://github.com/keras-team/keras/issues/3923

 #Convolutional
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.optimizers import Adam, rmsprop, Adadelta, adamax

def fixed_generator(generator):
    for batch in generator:
        yield (batch, batch)

# dimensions of our images.
img_width, img_height = 424, 424

train_data_dir = '/home/ryan/Documents/Galaxy_Zoo/Jell_Test/Grayscale/OnlyPositive_Autoencoder/Train'
validation_data_dir = '/home/ryan/Documents/Galaxy_Zoo/Jell_Test/Grayscale/OnlyPositive_Autoencoder/Validation'
nb_validation_samples = 44
batch_size = 35

input_img = Input(shape=(img_width, img_height,3))

x = Convolution2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)
myoptimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
autoencoder.compile(optimizer=myoptimizer, loss='binary_crossentropy')

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None)

print(type(train_generator))

validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None)

autoencoder.fit_generator(
        fixed_generator(train_generator),
        steps_per_epoch=3,
        epochs=20,
        validation_data=fixed_generator(validation_generator),
        validation_steps=3
        )
##save weights and start conv network with these weights
#test

import matplotlib.pyplot as plt
import numpy as np
# import cv2

# filename = '/home/ryan/Documents/Galaxy_Zoo/Jell_Test/Grayscale/Test/956159.jpg'
# im = cv2.resize(cv2.imread(filename), (256, 256)).astype(np.float32)
# im = im * 1./255
# im = im.transpose((2,0,1))
# #order of np.zeros is not correct
# datas = np.zeros((256, 256, 3, 1))
# datas[0, :, :, :] = im;

# decoded_imgs = autoencoder.predict(datas)

# finim = (decoded_imgs[0]*255).astype(int)
# finim = finim.transpose(1, 2, 0)
# plt.imshow(finim)
# plt.show()

# img = next(validation_generator)[:1] # Get one image
# dec = autoencoder.predict(img) # Decoded image
# img = img[0]
# dec = dec[0]
# img = (img.transpose((1, 2, 0))*255).astype('uint8')
# dec = (dec.transpose((1, 2, 0))*255).astype('uint8')

# plt.imshow(np.hstack((img, dec)))
# plt.title('Original and reconstructed images')
# plt.show()