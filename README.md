## Goal: 
The goal of this project was to implement an end-to-end neural network, for behavioral cloning of a simulated car driver. The input to the network are timestamped camera images (left, right , and center mounted).  The output of the network is a single floating point number, representing the steering angle of the car (for simplicity, other car controls e.g. throttle and brake were assumed constants and steering angle was kept in the range of -1 to 1). A Unity engine based driving simulator was used for training and testing the network. This simulator was provided by Udacity as part of the Self Driving Car nanodegree program. In case you want to try it out: [Link to download the linux driving simulator.](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)

The simulator has two  different tracks. Track 1 was used for training and validation. The purpose of Track 2 is to make sure that the solution is generalized enough, and not overfitted to Track 1.

<img src="images/simulator.png?raw=true" width="720">


## Architecture Choices: 
Two of the previous work on similar end-to-end deep-learning self driving car are:

1. [Nvidia's DAVE-2 system](https://arxiv.org/pdf/1604.07316v1.pdf)
2. [Comma.ai's steering angle model](https://github.com/commaai/research)

Another choice considered was to use transfer learning (fine-tuning of final few layers) with a pre-trained image classification CNN (e.g. VGG-16 or Inception-v3). The purpose of these CNNs are quite different (Imagenet classification challenge). These CNNs tend to have many more layers and an order of magnitude larger number of parameters than the choices (1) or (2) above which are designed to run real-time on resource constrained systems i.e. DrivePX and smart phones.

## Final Architecture:
  The final model was based upon Nvidia's DAVE-2, with a few modifications borrowed from Comma.ai's steering model. Here is a diagram of Nvidia's DAVE-2 CNN from (1) above:
  
  <img src="images/dave2.png?raw=true" width="360">

The number and types of convolutional and fully connected layers were borrowed exactly as above (including the kernel sizes, strides and the in-line normalization layer). One drawback of this architecture is that it does not use any dropout layers. Dropout layers have proven very effective in reducing overfitting problem of deep neural networks. In my model, I have added dropout layers after each fully-connected layer of the above architecture. Another modification was to use ELU as the activation function of hidden layers, instead of Relu, based upon Comma.ai's use of ELU. Finally, a tanh activation was used at the output neuron to keep the steering angle prediction within the range of -1 to +1.
  
  Adam optimizer was used with a learning rate of 1e-4, based upon my previous experience of traffic sign classifier. MSE was used for the single float32 output. Below is a code-snippet of the implementation of the final model:
  
```python
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

# Flatten before passing to Fully Connected layers
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

```
Summary of the model as reported by Keras' `model.summary()`:
```
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
lambda_norm (Lambda)             (None, 66, 200, 3)    0           lambda_input_3[0][0]             
____________________________________________________________________________________________________
conv1 (Convolution2D)            (None, 31, 98, 24)    1824        lambda_norm[0][0]                
____________________________________________________________________________________________________
elu1 (ELU)                       (None, 31, 98, 24)    0           conv1[0][0]                      
____________________________________________________________________________________________________
conv2 (Convolution2D)            (None, 14, 47, 36)    21636       elu1[0][0]                       
____________________________________________________________________________________________________
elu2 (ELU)                       (None, 14, 47, 36)    0           conv2[0][0]                      
____________________________________________________________________________________________________
conv3 (Convolution2D)            (None, 5, 22, 48)     43248       elu2[0][0]                       
____________________________________________________________________________________________________
elu3 (ELU)                       (None, 5, 22, 48)     0           conv3[0][0]                      
____________________________________________________________________________________________________
conv4 (Convolution2D)            (None, 3, 20, 64)     27712       elu3[0][0]                       
____________________________________________________________________________________________________
elu4 (ELU)                       (None, 3, 20, 64)     0           conv4[0][0]                      
____________________________________________________________________________________________________
conv5 (Convolution2D)            (None, 1, 18, 64)     36928       elu4[0][0]                       
____________________________________________________________________________________________________
elu5 (ELU)                       (None, 1, 18, 64)     0           conv5[0][0]                      
____________________________________________________________________________________________________
flatten_3 (Flatten)              (None, 1152)          0           elu5[0][0]                       
____________________________________________________________________________________________________
fc1 (Dense)                      (None, 100)           115300      flatten_3[0][0]                  
____________________________________________________________________________________________________
do1 (Dropout)                    (None, 100)           0           fc1[0][0]                        
____________________________________________________________________________________________________
elu6 (ELU)                       (None, 100)           0           do1[0][0]                        
____________________________________________________________________________________________________
fc2 (Dense)                      (None, 50)            5050        elu6[0][0]                       
____________________________________________________________________________________________________
do2 (Dropout)                    (None, 50)            0           fc2[0][0]                        
____________________________________________________________________________________________________
elu7 (ELU)                       (None, 50)            0           do2[0][0]                        
____________________________________________________________________________________________________
fc3 (Dense)                      (None, 10)            510         elu7[0][0]                       
____________________________________________________________________________________________________
do3 (Dropout)                    (None, 10)            0           fc3[0][0]                        
____________________________________________________________________________________________________
elu8 (ELU)                       (None, 10)            0           do3[0][0]                        
____________________________________________________________________________________________________
output (Dense)                   (None, 1)             11          elu8[0][0]                       
====================================================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
____________________________________________________________________________________________________
```
[Link to data flow graph of the model](images/model.png)

## Training Data Preparation:
  Training data captured while driving (by an expert driver;) on Track-1 was provided by Udacity. The data contained timestamped jpg images (160x320x3) from left, right, and center cameras alongwith a CSV logfile. The rows of the CSV logfile tied together the name and path of the image files, sterring angle, throttle, speed and brake values at every timestamp. Here is the histogram of the steering angles in this data.

![](images/steering_hist.png?raw=true)  
  
  As you can see from the above histogram, the steering angles in this data are very well balanced between right and left turns. However, there is a very high percentage of zero steering angle. Having a disproportionally high number of data corresponding to one class is not a good input to a neural network, as it can bias the neural network towards outputting that value. Also, in real life, when we teach someone new about driving, we teach them to do small and frequent steering corrections. These facts lead me to reduce the number of zero-steering data by 80%. i.e. only 20% of the zero-steering data was retained for actual training.
  
   Nvidia's DAVE-2 paper (Architecture choice #1 above) describes how to use right and left camera images to our advantage. The testing phase uses only center camera images i.e. input to the model is a single image at a time. Instead of throwing away the left and right images, we can use them as additional data for our training, after applying a fixed camera offset to the steering values. This gives us the first step towards data augmentation. Additionally, since the right and left camera see a more "extreme" turning conditions than just the center images, they are more helpful in "teaching" the neural network how to recover from extreme conditions, and help us towards a generalized solution during training. Below are some sample images from the three cameras:
   
![](images/orig_1.png?raw=true)   
![](images/orig_2.png?raw=true)   
![](images/orig_3.png?raw=true)   
   
 As the first steps in preparing the images for feeding into the model, I remove the top and bottom 20-pixels from each image and resize the resulting image to 66x200x3 (from the original 160x320x3). The cropping is done in order to remove the top pixels above horizon, and bottom pixels containing car's bumper. The resizing is performed to match the size of the input layer of our model. For the visualization, the above images, after cropping and resizing appear as:
 
![](images/crop_1.png?raw=true)   
![](images/crop_2.png?raw=true)   
![](images/crop_3.png?raw=true) 
  
  A big concern in this project was to avoid overfitting and be able to generalize for different tracks. Track-1 did not have much variation in lighting conditions and the track was very flat (no uphill, downhill, banking on curves etc.). In order to include such variation in training data, I have used data augmentation using Keras [ImageDataGenerator()](https://keras.io/preprocessing/image/).
  
  For efficient implementation of data augmentation, Keras has a model training method called fit_generator() that I used. It can take a python generator, such as ImageDataGenerator(), generating bacthes of augmented training data on-the-fly. The generator is run in parallel to the model and you can do real-time data augmentation on images on CPU in parallel to training your model on GPU.
  
  Keras' ImageDataGenerator() provides many built-in image transformation options, but it did not have a brightness adjustment option. Fortunately it provides hook for your own custom image transformation preprocessor. I added a random brightness change function as the preprocessor to the ImageDataGenerator(). In addition to the random brightness change, I used random shear (similar to openCV warpAffine transform) option that was built-in, for data augmentataion. Example images after applying these two augmentation techniques:
  
![](images/augm_1.png?raw=true)   
![](images/augm_2.png?raw=true)   
![](images/augm_3.png?raw=true) 
  
## Training and Validation:  
  The original data (after preprocessing, but before augmentation) was split between training dataset and validation dataset. This project was peculiar in the sense that validation loss was not a very good indicator of performance on the tracks. The losses were computed based upon still images from the dataset. In practice, there are many "right" answers for each image, and there is a time-series dependance where future steering values can correct for any "apparently" wrong decision on the current image. So, I kept the validation fraction to only 5% of the dataset, just to give me a sense of number of epochs to run. Any further increase in the validation fraction eats into the avaialble training data and started to hurt the performance on the tracks. In a sense, running on Track-1 became the main validation criterion and running on Track-2 became the test criterion. Number of epochs were kept at 9, because the validation loss started to increase substantially after that. Batch size was chosen to be 100, since it gave a smoother driving when compared to lower batch sizes of 32 or 64. Number of augmented samples in each epoch were chosen to be 2x training data size. Here is the output from Keras fit_generator while training:
    
```
Epoch 1/9
35466/35466 [==============================] - 53s - loss: 0.0367 - val_loss: 0.0274
Epoch 2/9
35466/35466 [==============================] - 53s - loss: 0.0301 - val_loss: 0.0276
Epoch 3/9
35466/35466 [==============================] - 54s - loss: 0.0249 - val_loss: 0.0186
Epoch 4/9
35466/35466 [==============================] - 53s - loss: 0.0233 - val_loss: 0.0205
Epoch 5/9
35466/35466 [==============================] - 54s - loss: 0.0220 - val_loss: 0.0250
Epoch 6/9
35466/35466 [==============================] - 53s - loss: 0.0209 - val_loss: 0.0222
Epoch 7/9
35466/35466 [==============================] - 53s - loss: 0.0202 - val_loss: 0.0262
Epoch 8/9
35466/35466 [==============================] - 53s - loss: 0.0197 - val_loss: 0.0261
Epoch 9/9
35466/35466 [==============================] - 53s - loss: 0.0195 - val_loss: 0.0239
```

## Test Results:

Track 1 video link (can it be embedded in readme.md ?)
Track 2 video link (can it be embedded in readme.md ?)

## System and SW used: 
* Ubuntu 16.04, Intel Core i7-6800K, 32GB System RAM
* Nvidia GTX1080 GPU with 8GB Graphics RAM
* CUDA 8.0, CuDNN 5.1
* Framework: Keras with Tensorflow backend
* Misc. Python libraries (numpy, pandas, openCV etc.)

## Final Thoughts:

