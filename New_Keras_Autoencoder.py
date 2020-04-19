# https://blog.keras.io/building-autoencoders-in-keras.html
 #https://github.com/keras-team/keras/issues/3923

 #Convolutional
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model, model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.optimizers import Adam, rmsprop, Adadelta, adamax

def fixed_generator(generator):
    for batch in generator:
        yield (batch, batch)

# dimensions of our images.
img_width, img_height = 424, 424

#train paths
train_data_dir = '/home/ryan/Documents/DATA/Autoencoder/Plain/Training'
validation_data_dir = '/home/ryan/Documents/DATA/Autoencoder/Plain/Validation'
#train_data_dir = '/home/ryan/Documents/DATA/Autoencoder/Sliced/Training'
#validation_data_dir = '/home/ryan/Documents/DATA/Autoencoder/Sliced/Validation'

#batch data
nb_validation_samples = 58
batch_size = 35

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.3,
        rotation_range=360,
        width_shift_range=.3,
        height_shift_range=.3,
        fill_mode='wrap')
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=True,
        seed=5)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=360)
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=True,
        seed=12)

#Network
input_img = Input(shape=(img_width, img_height,3))

x = Convolution2D(16, (3, 3), activation='relu', padding='same', name='Conv1')(input_img)
x = MaxPooling2D((2, 2), padding='same', name = 'Pool1')(x)
x = Dense(64, activation='relu', name = 'Dense1')(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same', name='Conv2')(x)
x = MaxPooling2D((2, 2), padding='same', name='Pool2')(x)
x = Dense(32, activation='relu', name='Dense2')(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same', name='Conv3')(x)
encoded = MaxPooling2D((2, 2), padding='same', name='Pool3')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional
#in future try conv2dtranspose for decoding layer
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, (3, 3), activation='relu', padding='same')(x)
x = Dense(32, activation='relu')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(16, (3, 3), activation='relu', padding='same')(x)
x = Dense(64, activation='relu')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, (3, 3), activation='sigmoid', padding='same')(x)

#compile and run
autoencoder = Model(input_img, decoded)
myoptimizer = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
autoencoder.compile(optimizer=myoptimizer, loss='binary_crossentropy')

autoencoder.fit_generator(
        fixed_generator(train_generator),
        steps_per_epoch=5,
        epochs=20,
        validation_data=fixed_generator(validation_generator),
        validation_steps=1
        )
#summaries
print(autoencoder.summary())
##save weights and and model start conv network with these weights
encoder = Model(input_img, encoded)
encoder.save_weights('Encoded.h5')

decoder = Model(input_img, decoded)
decoder.save_weights('Decoded.h5')
