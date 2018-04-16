# -*- coding: utf-8 -*-
# Copyright (C) 2018, Josef Brechler. 
# Follows Kaggle Competitions license, see https://www.kaggle.com/terms
# Some codes used in class plants_data_preprocessing were adopted
#    from https://www.kaggle.com/gaborvecsei/plants-t-sne submission
#

import os

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dense, Activation, Dropout, Flatten, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import History
from keras.callbacks import CSVLogger

from plants_classes import *  # probably not nice - use namespace here!
from plants_functions import *  # probably not nice - use namespace here!
from helper_functions import *  # probably not nice - use namespace here!

import constants

# version identifier
ver_id = 'v20180416a'

conv_depths=[32,32,64,64]
dropout_probs=[0.5,0.5,0.5,0.5]
batch_size = 64 # in each iteration, we consider 32 training examples at once
num_epochs = 500 # we iterate 200 times over the entire training set
hidden_size = 512 # the FC layer will have 512 neurons
strides=2

data_preprocessing = plants_data_preprocessing(train_data_fld = constants.TRAIN_DATA_FOLDER,
                                               prepr_img_fname = constants.PREPROCESSED_IMAGES_FNAME,
                                               prepr_lab_fname = constants.PREPROCESSED_LABELS_FNAME,
                                               prepr_data_fld = constants.PREPROCESSED_DATA_FOLDER,
                                               lab_enum_fname = constants.LABELS_ENUMARATORS_FNAME,
                                               force_preprocess=False)

# preprocess or load data
data_preprocessing.preprocess()
data_preprocessing.preprocess_for_cnn()

id_to_label_dict=data_preprocessing.labels_enumerators['id_to_label_dict']

# split to train / test data
X_train, X_test, Y_train, Y_test = train_test_split(data_preprocessing.X_for_cnn ,
                                                    data_preprocessing.Y_for_cnn,
                                                    test_size=0.2,
                                                    random_state=42)


num_train, height, width, depth = X_train.shape # there are 50000 training examples in CIFAR-10
num_test = X_test.shape[0] # there are 10000 test examples in CIFAR-10
num_classes = Y_train.shape[1]
#num_classes = np.unique(y_train).shape[0] # there are 10 image classes

# save model input data
model_input_data = (X_train, X_test, Y_train, Y_test)
fpath_input_data = os.path.join(constants.MODEL_INPUT_DATA_FOLDER, 'model_input_data_%s.h5' % (ver_id))
pickle_wrapper('w', fpath_input_data, x=model_input_data, rewrite=False)

################

input_size = [height, width, 1]
#input_size = [32, 32, 3]

model = Sequential()

model.add(Conv2D(filters=conv_depths[0], kernel_size=(3,3), 
                 strides=strides, input_shape=input_size,
                 data_format="channels_last"))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Dropout(dropout_probs[0]))

model.add(Conv2D(32, (3,3)))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Dropout(dropout_probs[1]))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Dropout(dropout_probs[2]))

model.add(Conv2D(64, (3,3)))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Dropout(dropout_probs[3]))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation(activation='relu'))

model.add(Dense(num_classes))
model.add(Activation(activation='softmax'))

# define and compile model
model.compile(loss='categorical_crossentropy', # using the cross-entropy loss function
              optimizer='adam', # using the Adam optimiser
              metrics=['accuracy']) # reporting the accuracy

# define callback loggers
history = History()
csv_logger = CSVLogger('../saved_objects/cnn_history/cnn_hist_%s.csv' % (ver_id), append=True, separator=';')


################


#train_datagen = ImageDataGenerator(
#    featurewise_center=True,
#    featurewise_std_normalization=True,
#    rotation_range=20,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    horizontal_flip=True)
#test_datagen = ImageDataGenerator(
#    featurewise_center=True,
#    featurewise_std_normalization=True,
#    rotation_range=20,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    horizontal_flip=True)
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

###############

train_datagen.fit(X_train)
test_datagen.fit(X_test)

# train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

# fits the model on batches with real-time data augmentation:
model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, 
                    epochs=num_epochs,
                    verbose=1,
                    validation_data=test_datagen.flow(X_test, Y_test),
                    callbacks=[history, csv_logger])

########################

# save model and callback history
model.save('../saved_objects/trained_cnn/cnn_%s.h5' % (ver_id))
pickle_wrapper('w', '../saved_objects/cnn_history/cnn_hist_%s.obj' % (ver_id), x=history.history, rewrite=True)

plt.plot(history.history['val_acc'])
