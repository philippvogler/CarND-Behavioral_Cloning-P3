
'LIBRARIES'
import numpy as np
import pandas as pd
import cv2

from keras.models import Sequential
from keras.layers import Convolution2D, ELU, Flatten, Dropout, Dense, BatchNormalization
from keras.preprocessing.image import img_to_array, load_img
from keras.utils.visualize_util import plot

from sklearn.model_selection import train_test_split
import scipy.misc
import pylab

image_shape = (80, 80, 3) # original shape (160, 320, 3)

'MODEL'
def nvidia_model():
# Keras Implementation of NVIDIA Model

    # drop out for regulazation
    drop_out_rate = 0.4

    # initial distribution of weights
    initial_distribution = 'normal'

    # building a sequential Keras model
    model = Sequential()

    # debugging
    # print ("DOR: " + str(drop_out_rate))
    # print ("Distribution: " + str(initial_distribution))
    # print ("Image shape: " + str(image_shape))

    # Normalizing
    model.add(BatchNormalization(epsilon=0.001, mode=1, axis=-1, weights=None, beta_init='zero', gamma_init='one', gamma_regularizer=None, beta_regularizer=None, input_shape = image_shape))

    # Convolutional feature maps:
    # 24@
    model.add(Convolution2D(24,5,5, subsample = (2,2), border_mode ='valid', init=initial_distribution))
    model.add(ELU())

    # 36@
    model.add(Convolution2D(36,5,5, subsample = (2,2), border_mode ='valid', init=initial_distribution))
    model.add(ELU())
    model.add(Dropout(drop_out_rate))

    # 48@
    model.add(Convolution2D(48,5,5, subsample = (2,2), border_mode ='valid', init=initial_distribution))
    model.add(ELU())
    model.add(Dropout(drop_out_rate))

    # 64@
    model.add(Convolution2D(64,3,3, subsample = (2,2), border_mode ='valid', init=initial_distribution))
    model.add(ELU())
    model.add(Dropout(drop_out_rate))

    #64@
    model.add(Convolution2D(64,3,3, subsample = (2,2), border_mode ='valid', init=initial_distribution))
    model.add(ELU())
    model.add(Dropout(drop_out_rate))

    model.add(Flatten())

    # Fully-connected layers
    model.add(Dense(1164, init=initial_distribution))
    model.add(Dropout(drop_out_rate))
    model.add(ELU())

    model.add(Dense(100, init=initial_distribution))
    model.add(Dropout(drop_out_rate))
    model.add(ELU())

    model.add(Dense(50, init=initial_distribution))
    model.add(Dropout(drop_out_rate))
    model.add(ELU())

    model.add(Dense(10, init=initial_distribution))
    model.add(ELU())

    model.add(Dense(1, init=initial_distribution))

    # debugging
    # model.summary()
    # plot(model, to_file='model.png', show_shapes=True)


    return model


'PREPROCESSING'

def sample_extractor (row, camera):
# taks the path to the image from the specified camera cell and loads the corresponding image from the data file

    # image
    img = load_img("data/" + row[camera].strip())
    image = img_to_array(img)

    # steering
    steering_angle = row['steering']

    return image, steering_angle, camera


def angle_argumentor (steering_angle, camera):
# arguments the images and steering angles form the right and left camera

    if (camera == 'left'):
        steering_angle = steering_angle + .25
    if (camera == 'right'):
        steering_angle = steering_angle - .25

    return steering_angle


def image_flipper (image, steering_angle):
# flipps the image randomly horizontally

    if np.random.rand() > 0.5:
        image = cv2.flip(image, 1)
        steering_angle = - steering_angle

    return image, steering_angle


def image_preprocessor (image):
# cropps the image to the area of interest and resizes the image to a square
# also used in the drive.py file

    # cropping
    image = image[55:135, :, :]

    # resizing
    image = cv2.resize(image, (80, 80))

    return image


probabilities = [0.33, 0.34, 0.33]

# debugging
# print ("Probabilities: " + str(propabilities))

def sample_preprocessor(sample):
# sample preprocessing piepline

    # picking a camera angel
    camera = np.random.choice(['right', 'center', 'left'], p = probabilities)

    # getting the sample
    image, steering_angle, camera = sample_extractor(sample, camera)

    # preprocessing the image
    image = image_preprocessor (image)

    # argumenting the left and right camera angles
    steering_angle = angle_argumentor (steering_angle, camera)

    # flipping images and angle randomly
    image, steering_angle = image_flipper (image, steering_angle)

    # saves proccesed image to disc
    #scipy.misc.imsave('outfile.jpg', image)

    return image, steering_angle


'GENERATOR'
def batch_generator (data, batch_size):
# creats a batch of argumented images and corresponding steering angles from the data given

    data_set_iterator = 0
    batch_count = data.shape[0]//batch_size

    while(True):
        # creats empty batches
        image_batch = np.empty(shape = (batch_size, 80, 80, 3))
        steering_batch = np.empty(shape = (batch_size,))

        # selects next slice of the data set for batching
        begin = data_set_iterator * batch_size
        end = begin + (batch_size - 1)

        # fills the empty batches with preprocessed data
        batch_index = 0
        for idx, sample in data.iloc[begin:end].iterrows():
            image_batch[batch_index], steering_batch[batch_index] = sample_preprocessor (sample)
            batch_index += 1

        data_set_iterator += 1

        # starts the next cycle, when there are not enough samples for an additional batch
        if data_set_iterator == batch_count - 1:
            data_set_iterator = 0

        yield image_batch, steering_batch


'MAIN'
if __name__ == "__main__":

    'DATA IMPORT'
    # loads colums center, left, right, and steering from Udacity dataset
    with open('data/driving_log.csv') as file:
        data_frame = pd.read_csv(file, usecols=[0, 1, 2, 3])

    # # plots histogram of steering angles
    # plot = data_frame['steering'].plot.hist()
    # pylab.savefig('plot.png')

    # creats randomly a 75% training data set and a 25% validation data set for testing
    split = 0.75
    training_data, validation_data = train_test_split(data_frame
                    ,train_size = split
                    ,random_state = 1
                    )

    # releases the original data frame from memory
    data_frame = None

    # fetches the model
    model = nvidia_model()

    'COMPILATION'
    optimization_algorithem = "adam"
    loss_function = "mse"
    #metric = ["accuracy"]

    model.compile(optimizer=optimization_algorithem
                , loss=loss_function
                #, metrics = metric
                )


    'DATA GENERATION'
    batch_size = 32

    # debugging
    # print ("Batch: " + str(batch_size))

    training_data_generator = batch_generator(training_data, batch_size)
    validation_data_generator = batch_generator(validation_data, batch_size)


    'TRAINING'
    number_epochs = 8

    # debugging
    # print("Epochs: " + str(number_epochs))

    model.fit_generator(training_data_generator,
                        samples_per_epoch =  (training_data.shape[0]//batch_size) * batch_size,
                        nb_epoch = number_epochs,
                        verbose = 1,
                        validation_data = validation_data_generator,
                        nb_val_samples =  (validation_data.shape[0]//batch_size) * batch_size
                        )

    'SAVING'
    model.save_weights('model.h5')
    with open('model.json', 'w') as outfile:
        outfile.write(model.to_json())

    print("Finished!")

'SOURCES'
'''
Subodh Malgonde
https://github.com/subodh-malgonde/behavioral-cloning/blob/5ed2fefa4fea52583150b3a70e20fe09a6c11150/model.py

Nvidia Corp.
http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf

Vivek Yadava
https://chatbotslife.com/learning-human-driving-behavior-using-nvidias-neural-network-model-and-image-augmentation-80399360efee#.qc7svcryi

https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.3k8qwljcl

Valipour Mojtaba
https://medium.com/@ValipourMojtaba/my-approach-for-project-3-2545578a9319#.byc7ui6hx
'''
