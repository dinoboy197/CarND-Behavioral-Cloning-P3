## Behavioral Cloning Project

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a [convolution neural network](https://github.com/dinoboy197/CarND-Behavioral-Cloning-P3/blob/master/model.py#L56-L73) based a successful model used by NVidia which inputs 160x320 RGB images from multiple camera angles at the front of the vehicle and outputs a single steering wheel angle instruction.

Specifically, the model includes:
* four convolutional layers
  * three 5x5 filters with 24, 36, and 48 depth
  * one 3x3 filter with 64 depth
* a maximum pooling layer with 2x2 pooling
* three fully-connected layers with 100, 50, and 10 outputs
* a final steering angle output layer

The input images are cropped to remove the top 50 and bottom 20 pixels to reduce noise in the image un-correlated with steering commands. Each pixel color value in the image is then normalized to [-0.5,0.5]. The model includes RELU layers to introduce nonlinearity after each of the convolutional and fully-connected layers.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers after each fully-connected layer in order to reduce overfitting. Each dropout layer is set to drop out 50% of inbound data during training.

The model was [trained and validated](https://github.com/dinoboy197/CarND-Behavioral-Cloning-P3/blob/master/model.py#L78-L80) on [different data sets](https://github.com/dinoboy197/CarND-Behavioral-Cloning-P3/blob/master/model.py#L22) to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an [Adam optimizer](https://github.com/dinoboy197/CarND-Behavioral-Cloning-P3/blob/master/model.py#L76), so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of multiple lanes of center lane driving in addition to segments of recovering from the left and right sides of the road during tight curves.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a well-known and high-performance network, and tune it for this steering angle prediction task.

My first step was to use a convolution neural network model similar to [the publically published NVidia architecture used for their self-driving car efforts](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). Given that I'm attempting to solve the exact same problem (steering angle command prediction) and their network is state of the art, that seemed like a good place to start. I removed one convolutional and one fully connected layer from the NVidia architecturet to reduce memory processing costs during training.

In between the convolutional layers of my model, I introduced Relu activations to introduce non-linearity, max pooling to reduce overfitting and computatational complexity, and 50% dropout during training (also to reduce overfitting).

The final step was to run the simulator to see how well the vehicle was driving around track one. The original data provided to me trained a model which only can off the track in one spot. After I augmented the data with left and right recovery, as well as images flipped on the vertical, the vehicle trained with this data did not run off the track; in only one spot did the simulated vehicle get close to running of the track on a curve.

#### 2. Final Model Architecture

The [model architecture](https://github.com/dinoboy197/CarND-Behavioral-Cloning-P3/blob/master/model.py#L56-L73) consists of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB color image   					|
| Cropping         		| 50 pixel top, 20 pixel bottom crop   			|
| Normalization        	| [0,255] -> [-0.5,0.5]                			|
| Convolution 5x5     	| 1x1 stride, valid padding, output depth 24 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride                   				    |
| Convolution 5x5     	| 1x1 stride, valid padding, output depth 36 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride                   				    |
| Convolution 5x5     	| 1x1 stride, valid padding, output depth 48 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride                   				    |
| Convolution 3x3     	| 1x1 stride, valid padding, output depth 64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride                   				    |
| Flattening	      	| 2d image -> 1d pixel values  				    |
| Fully connected		| 100 output neurons                        	|
| RELU					|												|
| Dropout				| 50% keep fraction								|
| Fully connected		| 50 output neurons                         	|
| RELU					|												|
| Dropout				| 50% keep fraction								|
| Fully connected		| 10 output neurons                          	|
| Output        		| Output - 1 steering angle command 			|

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

_Training Set_

I began with a sample of good driving behavior provided to me for training, which provided center, left, and right images taken from different points on the front end of the vehicle. This data includes multiple laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to correct major driving errors when the vehicle is about to run off the road. These images show what a recovery looks like starting from the left side:

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles during training to further generalize the model. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 8253 data image frames, each including center, left, and right images for a total of 24759.

_Training Process_

During training, the entire image data set is shuffled, then 80% of the images used for training and 20% used for testing.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I used an [early stopping condition based on knee-finding using the validation loss, with a patience of 2 epochs](https://github.com/dinoboy197/CarND-Behavioral-Cloning-P3/blob/master/model.py#L80). I used an [Adam optimizer](https://github.com/dinoboy197/CarND-Behavioral-Cloning-P3/blob/master/model.py#L76) so that manually training the learning rate wasn't necessary.
