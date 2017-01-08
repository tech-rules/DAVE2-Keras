# DAVE2-Keras
## Goal: 
problem statement from udacity. Track 1 and Track 2 generalization (sharper turns, uphill/downhill, more varying lighting conditions). Link to simulator (unity based).

## Architecture Choices: 
Regression problem. Previous published work of real world in Nvidia paper (link) and commai.ai research (link). Transfer learning + Fine-tuning using VGG-16 or Inception-v3 (much larger netowrks, meant for classification, less flexibility).

## System and SW details: 
NVidia GTX1080 (8GB VRAM). Intel Core i7 (32GB DRAM). Ubuntu 16.04. CUDA version. CuDNN version. Keras with Tensorflow backend. Misc. Python libraries (OpenCV, pandas, numpy, etc.).

## Final Architecture:
  Decided to use Nvidia architecture as a base, and customizing for problem at hand and picking some of the good ideas from Comma. Here is a diagram of Nvidia's DAVE-2 architecture from the above paper.
  Input normalization layer as in Nvidia. Its benefits. 
  Added Dropout layers to reduce overfitting. Why. (nvidia may have a lot more data trained for days)
  Used ELU instead of RELU. Why ? deemed better for regression problems (relus better for classification) inspired by comma.
  Optimizer choice. Learning Rate choice.
  Description of my layers. number of parameters. keras model.info().
  My diagram (tensorboard or draw.io)

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
Epoch 1/8
18404/18404 [==============================] - 28s - loss: 0.0550 - val_loss: 0.0230
Epoch 2/8
18404/18404 [==============================] - 27s - loss: 0.0446 - val_loss: 0.0381
Epoch 3/8
18404/18404 [==============================] - 27s - loss: 0.0397 - val_loss: 0.0830
Epoch 4/8
18404/18404 [==============================] - 27s - loss: 0.0394 - val_loss: 0.0812
Epoch 5/8
18404/18404 [==============================] - 27s - loss: 0.0364 - val_loss: 0.0446
Epoch 6/8
18404/18404 [==============================] - 27s - loss: 0.0332 - val_loss: 0.0460
Epoch 7/8
18404/18404 [==============================] - 27s - loss: 0.0321 - val_loss: 0.0412
Epoch 8/8
18404/18404 [==============================] - 27s - loss: 0.0312 - val_loss: 0.0628
```

## Test Results:
Track 1 video link (can it be embedded in readme.md ?)
Analysis of Track 1 video

Track 2 video link (can it be embedded in readme.md ?)
Analysis of Track 2 video
