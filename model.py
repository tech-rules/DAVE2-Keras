import numpy as np
import pandas as pd
import cv2
import matplotlib.image as mpimg
import json

from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D, Dense, Dropout
from keras.layers import Activation, Flatten, Lambda, Input, ELU
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator

# Read and pre-process the logfile generated in training mode
csv_file = 'data/driving_log.csv' # Training logfile
df = pd.read_csv(csv_file, index_col = False)

# Removing data with throttle below 0.2
ind = df['throttle'] > 0.2
df = df[ind].reset_index()

# Remove 'throttle', 'brake', 'speed' columns
df = df.drop(['throttle', 'brake', 'speed'], 1)

# Seprate data frames for center, right, left
df_c = df.drop(['right', 'left'], 1)
df_r = df.drop(['center', 'left'], 1)
df_l = df.drop(['center', 'right'], 1)

# Reduce steering angle of 0s from the center data frame
# Remove all zeros and then add back 50% of it
ind = df_c['steering'] != 0
df_c_zeros = df_c[~ind].reset_index(drop=True)
df_c_zeros = df_c_zeros.sample(frac=0.2)
df_c = df_c[ind].reset_index(drop=True)

# Add/Remove fixed steering offset from the left and right data frames
CAMERA_OFFSET = 0.20
df_r['steering'] = df_r['steering'].apply(lambda x: x - CAMERA_OFFSET)
df_l['steering'] = df_l['steering'].apply(lambda x: x + CAMERA_OFFSET)

# Rename columns to match and then concatenate all three dataframes
df_c.columns = ['index', 'image_path', 'steering']
df_c_zeros.columns = ['index', 'image_path', 'steering']
df_r.columns = ['index', 'image_path', 'steering']
df_l.columns = ['index', 'image_path', 'steering']
df = pd.concat([df_c, df_c_zeros, df_r, df_l], axis=0, ignore_index=True)
df = df.drop('index', 1)

# Split dataset into training and validation
df_train = df.sample(frac=0.95)
df_val = df.loc[~df.index.isin(df_train.index)]

# Functions to read and preprocess images
def readProcess(image_file):
    """Function to read an image file and crop and resize it for input layer
    
    Args: 
      image_file (str): Image filename (expected in 'data/' subdirectory)

    Returns:
      numpy array of size 66x200x3, for the image that was read from disk
    """
    # Read file from disk
    image = mpimg.imread('data/' + image_file.strip())
    # Remove the top 20 and bottom 20 pixels of 160x320x3 images
    image = image[20:140, :, :]
    # Resize the image to match input layer of the model
    resize = (200, 66)
    image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
    return image

def randBright(image, br=0.25):
    """Function to randomly change the brightness of an image
    
    Args: 
      image (numpy array): RGB array of input image
      br (float): V-channel will be scaled by a random between br to 1+br

    Returns:
      numpy array of brighness adjusted RGB image of same size as input
    """
    rand_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    rand_bright = br + np.random.uniform()
    rand_image[:,:,2] = rand_image[:,:,2]*rand_bright
    rand_image = cv2.cvtColor(rand_image, cv2.COLOR_HSV2RGB)
    return rand_image

# Convert training dataframe into images and labels arrays
X_train = []
for im in df_train.image_path:
    X_train.append(readProcess(im))
X_train = np.asarray(X_train)
y_train = np.array(df_train.steering, dtype=np.float32)

# Convert validation dataframe into images and labels arrays
X_val = []
for im in df_val.image_path:
    X_val.append(readProcess(im))
X_val = np.asarray(X_val)
y_val = np.array(df_val.steering, dtype=np.float32)

# Training data generator with random shear and random brightness
datagen = ImageDataGenerator(shear_range=0.1, preprocessing_function=randBright)

# Start of MODEL Definition
input_shape = (66, 200, 3)
model = Sequential()
# Input normalization layer
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=input_shape, name='lambda_norm'))

# 5x5 Convolutional layers with stride of 2x2
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", name='conv1'))
model.add(ELU(name='elu1'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid", name='conv2'))
model.add(ELU(name='elu2'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid", name='conv3'))
model.add(ELU(name='elu3'))

# 3x3 Convolutional layers with stride of 1x1
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", name='conv4'))
model.add(ELU(name='elu4'))
model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid", name='conv5'))
model.add(ELU(name='elu5'))

# Flatten before passing to the fully connected layers
model.add(Flatten())
# Three fully connected layers
model.add(Dense(100, name='fc1'))
model.add(Dropout(.5, name='do1'))
model.add(ELU(name='elu6'))
model.add(Dense(50, name='fc2'))
model.add(Dropout(.5, name='do2'))
model.add(ELU(name='elu7'))
model.add(Dense(10, name='fc3'))
model.add(Dropout(.5, name='do3'))
model.add(ELU(name='elu8'))

# Output layer with tanh activation 
model.add(Dense(1, activation='tanh', name='output'))

adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(optimizer="adam", loss="mse")

# Train and save the model
BATCH_SIZE = 100
NB_EPOCH = 9
NB_SAMPLES = 2*len(X_train)
model.fit_generator(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
                    samples_per_epoch=NB_SAMPLES, nb_epoch=NB_EPOCH,
                    validation_data=(X_val, y_val))

model.save_weights('model.h5')
with open('model.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
