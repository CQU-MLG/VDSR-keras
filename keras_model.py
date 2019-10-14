from __future__ import print_function
import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import Conv2D, MaxPooling2D, Input, ZeroPadding2D, merge, add
import tensorflow as tf
from keras.models import load_model
from keras import optimizers
from keras import losses
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger

import os, glob, sys, threading
import scipy.io
from scipy import ndimage, misc
import numpy as np
import re
import math
from prepare_data import(
read_npy,
down_up_samples,
wavelet_transform,
read_bmp,
array_to_patch
)

IMG_SIZE = (41, 41, 1)
BATCH_SIZE = 64
EPOCHS = 200
TRAIN_SCALES = [2, 3, 4]
VALID_SCALES = [4]


def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def PSNR(y_true, y_pred):
    max_pixel = 1.0
    return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

# make data
label = read_bmp()
input = down_up_samples(label)
# label = np.array(label)
# input = np.array(input)
train_input, train_label = array_to_patch(strid=10, inputs=input, label=label, image_size=41)

input_img = Input(shape=IMG_SIZE)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(input_img)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)

model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(model)
model = Activation('relu')(model)
model = Conv2D(1, (3, 3), padding='same', kernel_initializer='he_normal')(model)
res_img = model

output_img = add([res_img, input_img])

model = Model(input_img, output_img)

# model.load_weights('./checkpoints/weights-improvement-20-26.93.hdf5')

adam = Adam(lr=0.001, decay=1e-5, clipvalue=0.1, epsilon=1e-8)
sgd = SGD(lr=1e-2, momentum=0.9, decay=1e-4, nesterov=False)
model.compile(adam, loss='mse', metrics=[PSNR, "accuracy"])

model.summary()

filepath = "./checkpoints2/vdsr-{epoch:02d}-{PSNR:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor=PSNR, verbose=1, mode='max')
callbacks_list = [checkpoint]

train_log = CSVLogger(filename="train.log")
model.fit(x=train_input, y=train_label, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=callbacks_list, shuffle=True)

print("Done training!!!")

print("Saving the final model ...")

# model.save('vdsr_model_80epoch.h5')  # creates a HDF5 file
# del model  # deletes the existing model