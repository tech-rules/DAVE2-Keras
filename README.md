## Goal: 
The goal of this project was to implement an end-to-end neural network, for behavioral cloning of a simulated car driver. The input to the network are timestamped camera images (left, right , and center mounted).  The output of the network is a single floating point number, representing the steering angle of the car (other car controls e.g. throttle and brake were assumed constants). A Unity engine based driving simulator was used for training and testing the network. This simulator was provided by Udacity as part of the Self Driving Car nanodegree program. In case you want to try it out: [Link to download Linux Driving Simulator](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/5831f0f7_simulator-linux/simulator-linux.zip)

The simulator has two  different tracks. Track 1 was used for training and validation. The purpose of Track 2 is to make sure that the solution is generalized enough, and not overfitted to Track 1.

![](images/simulator.png?raw=true)

## Architecture Choices: 
Two of the previous work on similar end-to-end deep-learning self driving car are:

1. [Nvidia's DAVE-2 system](https://arxiv.org/pdf/1604.07316v1.pdf)
2. [Comma.ai's steering angle model](https://github.com/commaai/research)

Another choice considered was to use transfer learning (fine-tuning of final few layers) with a pre-trained image classification CNN (e.g. VGG-16 or Inception-v3). The purpose of these CNNs are quite different (Imagenet classification challenge), and their parameter space is much larger than the choices (1) or (2) above.

## System and SW used for this project: 
* Ubuntu 16.04, Intel Core i7-6800K, 32GB System RAM
* Nvidia GTX1080 GPU with 8GB Graphics RAM
* CUDA 8.0, CuDNN 5.1
* Framework: Keras with Tensorflow backend
* Misc. Python libraries (numpy, pandas, openCV etc.)

## Final Architecture:
  The final model was based upon DAVE-2 with a few modifications borrowed from Comma.ai's steering model. Here is the diagram of Nvidia's DAVE-2 CNN:
![](images/dave2.png?raw=true)  

  Input normalization layer as in Nvidia. Its benefits. 
  Added Dropout layers to reduce overfitting. Why. (nvidia may have a lot more data trained for days)
  Used ELU instead of RELU. Why ? deemed better for regression problems (relus better for classification) inspired by comma.
  Optimizer choice. Learning Rate choice.
  
  Below is the snippet of implementation in Keras, of the final model:
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
Summary of the model as reported by Keras' model.summary():
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
[A picture of the model](images/model.png)

## Training Data Preparation:
  Udacity provided training data. Explain what was in the data. csv file, images. center, left, right image. histogram of steering angle (insert picture) showed that center had too many zeros. removed 75% zeros, in order to teach the NN more frequent and small sterring adjustments, similar to what we teach a new human driver. right and left camera images were used as a means to teach recovery and generate aditional data (similar to nvidia paper). CSV and pandas processing. (insert new histogram)
  
  Decision to augment data for reducing overfitting and be able handle different kind of tracks. 
  Benefits of using generator for data augmentation. 
  Keras image preprocessing generator fits the bill. in-built and flexibility to extend via pre-processing function.
  show function call to image generator and fit_generator.
  Added a random brightness function on top of random shear and zoom functions available in keras generator
  
## Training/Validation/Testing:  
  Batch size experimentation. (like a hyperparameter)
  Number of Epoch experimentation
  Udacity provided track 1 data as training data. Self generated data on track 2 as validation data.
  Actual running on tracks as test.
  Analysis of training trial and error process.
    
```
Epoch 1/10
19802/19802 [==============================] - 30s - loss: 0.0516 - val_loss: 0.0233
Epoch 2/10
19802/19802 [==============================] - 29s - loss: 0.0421 - val_loss: 0.0691
Epoch 3/10
19802/19802 [==============================] - 29s - loss: 0.0372 - val_loss: 0.0324
Epoch 4/10
19802/19802 [==============================] - 29s - loss: 0.0346 - val_loss: 0.0584
Epoch 5/10
19802/19802 [==============================] - 29s - loss: 0.0322 - val_loss: 0.0477
Epoch 6/10
19802/19802 [==============================] - 29s - loss: 0.0316 - val_loss: 0.0708
Epoch 7/10
19802/19802 [==============================] - 29s - loss: 0.0284 - val_loss: 0.0961
Epoch 8/10
19802/19802 [==============================] - 29s - loss: 0.0278 - val_loss: 0.0679
Epoch 9/10
19802/19802 [==============================] - 29s - loss: 0.0273 - val_loss: 0.0705
Epoch 10/10
19802/19802 [==============================] - 29s - loss: 0.0273 - val_loss: 0.0616
```

## Test Results:
Track 1 video link (can it be embedded in readme.md ?)
Analysis of Track 1 video

Track 2 video link (can it be embedded in readme.md ?)
Analysis of Track 2 video
