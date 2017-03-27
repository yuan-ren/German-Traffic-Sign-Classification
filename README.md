# German Traffic Sign Classification

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/class_size.png "Visualization"
[image2]: ./examples/grayscale.png "Grayscaling"
[image3]: ./examples/augmentation.png "Changing perspective"
[image4]: ./examples/test1.jpg "Traffic Sign 1"
[image5]: ./examples/test2.jpg "Traffic Sign 2"
[image6]: ./examples/test3.jpg "Traffic Sign 3"
[image7]: ./examples/test4.jpg "Traffic Sign 4"
[image8]: ./examples/test5.jpg "Traffic Sign 5"
[image9]: ./examples/test_images_cropped.png "Processed test images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing the number of images in each class.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

I decided to convert the images to grayscale because, by looking at examples of these traffic signs, I found that RGB colors do NOT help classification. After converting to grayscale, human being can still classify these traffic signs.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

I also normalized the image data to the range of [0, 1].

#### 2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Since the validation data set is provided, I did NOT splitting the data into training and validation sets. 

The 5th and 6th code cells of the IPython notebook contains the code for augmenting the data set. I decided to generate additional data because the image distribution is not uniform across different classes. For example, traffic sign 02 (speed limit 50km/h) has 2250 images, but traffic sign 00 (speed limit 20km/h) only has 210 images. In the 5th code cell, I made all classes having the same number (10000) of images. This was done by randomly picking images from the class and put them into a large array. Note that many images were used multiple times in order to augment data set. Then in the 6th code cell, I made all images unique by randomly changing perspective using OpenCV. Traffic signs are still classified to the same class after the transformation. This also help generalize the data set.

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is 1) each class in augmented data set has 10000 images 2) more images are generated by changing the perpective of original images.

My final training set had 430000 number of images. My validation set and test set had 4410 and 12630 number of images.

#### 3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 9th cell of the ipython notebook. 

My final model is similar to LeNet-5 model. The difference is that my model is wider with 108 channels, thus more degrees of freedom. Reason for this is that standard LeNet-5 model gives a validation accuracy about 0.89, meaning underfitting of training set. By making the model wider, it increases the number of parameters for training.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x108 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  valid padding, outputs 14x14x108 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x108 		|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  valid padding, outputs 5x5x108 				|
| Flatten  | outputs 2700  |
| Fully connected		| outputs 120        									|
| Fully connected		| outputs 84        									|
| Fully connected		| outputs 43        									|
| Softmax				|       									|
 


#### 4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 10th, 11th and 12th cells of the ipython notebook. 

To train the model, I used the AdamOptimizer provided by tensorflow. Here are the parameters that I used:
Batch size: 128
Epoch: 10
Learning rate: 0.001

#### 5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 11th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 0.99585
* validation set accuracy of 0.96735 
* test set accuracy of 0.96112

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen? 
I initially chose the standard LeNet-5 model, which gave a validation accuracy about 0.89

* What were some problems with the initial architecture?
The validation accuracy is low, meaning that the data set is under fitting.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I increased the width of the LeNet-5 model. Layer 1 was increased from 6 to 108 channels. Layer 2 was increased from 16 to 108 channels. This way more parameters were added into the LeNet-5 model, so it would be more powerful in fitting traffic sign images.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
I didn't use dropout or other regularization method in this submit. My final validation and test accuracies are not very high, around 0.96. So I think overfitting is the major problem of this model. I should probably add more covolution layers to make the model deeper. Then I may need to consider adding dropout layer into the model.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I cropped the image to 32x32, then I normalized the images and converted them into grayscale. After processing, test images look like this:

![alt text][image9]

The first two images are easy. The third one is a little difficult as the sign is off-center. The fourth and fifth images are not very clear so they might be dificult to classify.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead      		| Turn right ahead   									| 
| Road work     			| Road work 										|
| Road work					| Road work											|
| No passing	      		| No passing					 				|
| 30km/h			| 30km/h      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 17th cell of the Ipython notebook.

For the first image, the model is sure that this is a sign of turn right ahead (probability of 1.0), and the image does contain a sign of turn right ahead. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead | 100.00% |
| Keep left | 0.00% |
| Speed limit (60km/h) | 0.00% |
| End of speed limit (80km/h) | 0.00% |
| Vehicles over 3.5 metric tons prohibited | 0.00% |

For the second image, the model is sure that this is a sign of road work (probability of 1.0), and the image does contain a sign of road work. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work  | 100.00% |
| Slippery road | 0.00% |
| Road narrows on the right | 0.00% |
| Bumpy road | 0.00% |
| Traffic signals | 0.00% |

For the third image, the model is sure that this is a sign of road work (probability of 1.0), and the image does contain a sign of road work. The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Road work  | 100.00% |
| Speed limit (20km/h)  | 0.00% |
| Traffic signals  |  0.00% |
| Speed limit (30km/h)  |  0.00% |
| Road narrows on the right  |  0.00% |

For the fourth image, the model is only slightly sure that this is a sign of no passing (probability of 0.5). The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| No passing |  50.51% |
| End of no passing |  49.32% |
| Yield |  0.17% |
| Speed limit (120km/h) |  0.00% |
| Vehicles over 3.5 metric tons prohibited |  0.00% |

For the fifth image, the model is pretty sure that this is a sign of 30km/h (probability of 0.96). The top five soft max probabilities were 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (30km/h) |  95.92% |
| Speed limit (50km/h) |  4.03% |
| Speed limit (20km/h) |  0.05% |
| Speed limit (70km/h) |  0.00% |
| Speed limit (120km/h) |  0.00% |
