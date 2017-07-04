# Writeup Report Traffic Sign Classifier Project

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Visualize the networks state with test images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./example_images/writeup1.png "Random image"
[image2]: ./example_images/writeup2.png "Labeled image"
[image3]: ./example_images/writeup3.png "Data Distribution"
[image4]: ./example_images/writeup4.png "Reference image"
[image4a]: ./example_images/writeup4a.png "Grayscale"
[image5]: ./example_images/writeup5.png "Histogram equalized"
[image6]: ./example_images/writeup6.png "Normalized"
[image7]: ./example_images/writeup7.png "Noise"
[image8]: ./example_images/writeup8.png "Rotated"
[image9]: ./example_images/writeup9.png "Training Curve"
[image10]: ./example_images/writeup10.png "New images"
[image11]: ./example_images/writeup11.png "New images with predictions"
[image12]: ./example_images/writeup12.png "Top 5 Softmax"
[image13]: ./example_images/trained_l1.png "Trained Layer1 visualization"
[image14]: ./example_images/trained_l2.png "Trained Layer2 visualization"
[image15]: ./example_images/untrained_l1.png "Untrained Layer1 visualization"
[image16]: ./example_images/untrained_l2.png "Untrained Layer2 visualization"


## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

You're reading it! and here is a link to my [project code](https://github.com/Alexander-Frank/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

*1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.*

I used vanilla python to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

*2. Include an exploratory visualization of the dataset.*

I started by displaying a random image from the dataset:

![alt text][image1]

I decided it is beneficial to create a way of accessing the real signname. I'll be using the dictionary I used in this section multiple times during the entire notebook. Here is the same picture with its label displayed:

![alt text][image2]

I used the pandas library to show an exploratory visualization of the dataset and 
calculate summary statistics of the traffic signs data set.

Here is the exploratory visualization of the data set. It is a bar chart showing how the train, test and validation datasets are distributed by signtypes:

![alt text][image3]

We can see that the distribution is not uniform. The difference of images per class in the training set is important. This can lead to classification issues since cetrain classes are underrepresentated.

To have a better look at the  data I used the pandas library to calculate summary statistics of the traffic
signs data set:

|   	|y_train 	    |y_test 	    |y_valid    |
|:-----:|:-------------:|:-------------:|:---------:|
|count 	|43.000000 	    |43.000000 	    |43.000000  |
|mean 	|809.279070 	|293.720930 	|102.558140 |
|std 	|626.750855 	|233.442389 	|69.662213  |
|min 	|180.000000 	|60.000000 	    |30.000000  |
|25% 	|285.000000 	|90.000000 	    |60.000000  |
|50% 	|540.000000 	|180.000000 	|60.000000  |
|75% 	|1275.000000 	|450.000000 	|150.000000 |
|max 	|2010.000000 	|750.000000 	|240.000000 |

Note the min and max for y_train.

### Design and Test a Model Architecture

*1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)*

To display the preprocessing steps, I'll be using the following reference image:

![alt text][image4]

As a first step, I decided to convert the images to grayscale because it reduces the size of the training images from (32,32,3) to (32,32,1), thus decreasing the amount of time it takes to train the model. Most importantly though, it takes away distoritons caused by color. For example, color saturation at night or in low light might differ substantially from direct sunlight. Grayscale can help to reduce errors. I used a simple average for grayscaling the image.

Here is the image after grayscaling:

![alt text][image4a]

Next, I histogram equalized the image from grayscale. I used the built in cv2.equalizeHist() function. I did this to account for inequal images. As visible in the image, it corrects the brightness visibly:

![alt text][image5]

As a last step, I normalized the image data. As you can see this doesn't change the image visually. However, it does add to the training process of the net, since values are normalized and not on a scale of 0 - 255.

![alt text][image6]

I've experimented with adding additional data. I used two techniques for augemntation: 
- Adding Gaussian Noise
- Rotating the image (-max_ange = 8 degress, max_angle = 8 degrees)

![alt text][image7] ![alt text][image8]

However, I decided not to use the additional augmented data. I left the code cells in the notebook for reference. The model accuracy didn't improve when using the augmented data as additional input to the model.

---

*2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.*

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscaled image   					| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x6x16 				    |
| Fully connected		| input 400, output 120							|
| RELU					|												|
| Dropout				| 0.5 during training							|
| Fully connected		| input 120, output 84							|
| RELU					|												|
| Dropout				| 0.5 during training							|
| Softmax				| output = 43 									|
 
---

*3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.*

To train the model, I experimented with increasing and reducing the number of epochs, batchsize, leraning rate, adding in new layers and changing the hyperparameter for random initialization of weights and biases.

I adapted most of the settings and general architecture from the LeNet-5 architecture. The biggest change I made was adding dropout to the two fully connected layers and and changing the batchsize to 64. The dropout helped generalizing the model, the reduced batchsize yielded better accuracy results.

---

*4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.*

My final model results were:
* training set accuracy of 100%
* validation set accuracy of 97.4%
* test set accuracy of 94.7%

I used a combination of well known structure and iterative approach. My architecture is based on the LeNet-5 architecture. This architecture has proven to work well on image data of similar size. Traffic signs have similar features to handwritten numbers, which can be learned by the model.

To generalize, I added a dropout of 0.5 to both fully connected layers.

Below you can see the training process:

![alt text][image9]

Towards the end of the 100 epochs (which may have been a bit overkill) we see a training accuracy of 100% and a steady validation accuracy of 95% - 100%.

To evaluate tha data, a function was created. It can be found in cell 25.

The first approach I used was the standard LeNet-5 architecture with all provided training data without using grayscale. I adapted the architecture to accomodate the image shape of (32,32,3) for color images. After that I stared with grayscaling, added histogram equalization and normalizdation later on. I added additional pooling layers and adapted the sizes, but saw a drop in accuracy. 

I haven't experienced overfitting. After experimenting with the architecture I turned towards the number of epochs and the batch size. Increasing the number of epochs generllay led to better results. Decreasing the batch size was my final adjustment, which helped to increase the model accuracy. I did this after reading acrticles about optimal batchsizes. Gernerally, lower batchsizes lead to more iterations and better tuning of the model parameters. 

The final accuracy show evidence that the adaptation of the LeNet-5 architecture is a good approach to robust traffic sign classification.


### Test the Model on New Images

*1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.*

I've decided to depart from the official task given of "find 5 on the web". I happen to live in Germany and decided to go out and take some pictures myself. 

All street signs match the training data.
However, since the training data does not include all german street signs,
I only took pictures of those included in the training data.

All images were taken with my OnePlus3 phone, with dimensions of 3488 x 3488.

Locations: East Berlin and Nuremberg Airport  
Conditions: Sunny weather  
Date and time: June 2017, afternoon  

I downscaled the images to 32 x 32.

Here are all streetsigns with their corresponding labels (manually added):

![alt text][image10]

All images pose their own challenges. The picture hasn't allways been taken standing straight before the sign, therefore angle and rotation differ. Some pictures show glare from sunlight. All pictures have different backgrounds.

---

*2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).*

Here are the results of the prediction:

![alt text][image11]

The new dataset has an accuracy of 81%. This is below the test dataset accuracy of 94% but generally favors the model performance. Imges with glare from sunshine showed problems, a solution could be to eradicate bright spots from the images.

---

*3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)*

The code for making predictions on my final model is located in the 26th cell of the Ipython notebook.

For the first image, the model is 100% sure that it's a yield sign. The other Probablities are negligible.

Here is a graphic depicting top 5 softmax for each image, prediction and correct value combined in barcharts to easily see how certain the model is:

![alt text][image12]

Overall, certainties are often close to 100%. On images which were classified wrong, the correct result was within the top 5 for most cases. However, some images were classified not correctly with high certainty. This has to be resolved especially for high importnace signs such as "STOP". Options vary from using 2 models and averaging predictions or using the shape to pre-classify the image. A stop sign will have a hexagon shahpe, warning signs will be a triangle, etc. Then build individual models for each sign shape.

### Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
*1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?*

During my development process I trainded multiple models, showing similar results on the visualization. Especially the first layer visualization makes it clear that the model is focusing on lines (sample image choosen: 30th image from my own daat) of the triangle and the content within the triangle.

Here is the output for both, layer 1 and 2 of my trained model:

![alt text][image13] ![alt text][image14]

I compared the results agains an untrained net. It's again displaying l1 and l2 stimulated by the same image. Since the values are randomly initialized, one can clearly see the progress made through training the net. 

![alt text][image15] ![alt text][image16]
