# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

#### Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* model.h5 containing a trained convolution neural network 
* drive.py for driving the car in autonomous mode. It takes model.h5 as an input argument.
* writeup_report.md summarizing the results

Helper files:
* model_lib.py contains different model architectures. Each function is a separate architecture.
* preprocess_data.py contains the data generator and augmentation function.
* visualize_layers.py contains functions to visualize the filters and feature maps.
* resnet_v1.py contains the a reduced model resent architecture which cannot be found in the keras model library.

#### Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

You can find the different model architectures used for this project in model_lib.py. I found the Nvidia end to end deep learning network to give the best results. It has about 350,000 trainable parameters.


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x160x3 RGB image   							| 
| Crop         		| Trim parts of irrelevant data from the images   							| 
| Gray         		| Convert images to grayscale   							| 
| Normalize         		| Normalize image between -0.5 to 0.5   							| 
| Convolution 5x5, 24 layers     	| 2x2 stride, same padding|
| RELU					|			Activation									|
| Convolution 5x5, 36 layers     	| 2x2 stride, same padding|
| RELU					|			Activation									|
| Convolution 5x5, 48 layers     	| 2x2 stride, same padding|
| Batch Normalization					|												|
| RELU					|			Activation									|
| Convolution 3x3, 64 layers     	| 1x1 stride, same padding|
| RELU					|			Activation									|
| Convolution 3x3, 64 layers     	| 1x1 stride, same padding|
| RELU					|			Activation									|
| Fully Connected					|			Output = 100									|
| Dropout					|												|
| Fully Connected					|			Output = 50									|
| Dropout					|												|
| Fully Connected					|			Output = 10									|
| Fully Connected					|			Output = 1									|

#### 2. Attempts to reduce overfitting in the model

The model contains 2 dropout layers in order to reduce overfitting. The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

```
BATCH_SIZE= 64
num_epochs = 50
```
Along with these I used two callback functions to regulate the learning rate. See model.py::43

```
# Prepare callbacks for model saving and for learning rate adjustment.
lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [lr_reducer, lr_scheduler]
```

#### 4. Appropriate training data

I used the training data provided with this project. To augment the data, I flipped the images and added fake steering to the left and right camera images.

```
                   image = cv2.imread(name)
                    images.append(image)
                    measurement = float(batch_sample[3])
                    if i == 1: # left camera, steer right
                        measurement += correct_factor
                    elif i == 2: # right camera, steer left
                        measurement -= correct_factor
                    measurements.append(measurement)
            
            # augment data  by flipping images
            augmented_images, augmented_measurements = [], []
            for image, measurement in zip(images, measurements):
                augmented_images.append(image)
                augmented_measurements.append(measurement)
                augmented_images.append(cv2.flip(image, 1))
                augmented_measurements.append(measurement*-1.0)
```

### 4. Final training and validatation loss

![image](https://user-images.githubusercontent.com/22652444/118587954-7d823100-b76b-11eb-9c8e-c481f792721b.png)

### 5.Experimentation with other model architectures

Simple model:
     
![image](https://user-images.githubusercontent.com/22652444/118588490-79a2de80-b76c-11eb-9f1b-410cc75be1a1.png)
         
LeNet:      
![image](https://user-images.githubusercontent.com/22652444/118588518-87586400-b76c-11eb-905e-510f124f63a6.png)

### 6. Thoughts for future work

The Nvidia end to end deep learning paper suggested to smartly augment data to add more images of turning rather than following straight lines. This would help the model maintain its lane. I did not employ that strategy in this project because although it seems likes a smart strategy, it would be too time consuming. Because of memory constraints, I could not train the model over a Resnet 50 architecture with a reasonable batch size. In the future, this would be a good starting point to understand if more training data is needed.


