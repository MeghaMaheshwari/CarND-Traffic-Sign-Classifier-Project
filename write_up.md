## <b>Udacity Self-Driving Nanodegree 
<hr/>


## Traffic Sign Recognition
======================================================================

### Build a Traffic Sign Recognition Project

> **The Steps in the Project are as following :**

> - Load the training, test and validation data from the picked file provided in the project.
> - Visualize and explore the dataset.
> - Preprocess the data set - shuffle the data set, convert images to grayscale and normalize.
> - Train the training data set on a model architecture - LeNet architecture has been used here.
> - Validate the model architecture on the validation set. 
> - Test the output of the model architecture on the test set.
> - Use the model to make predictions on new images.
> - Analyze the softmax probabilities of the new images.
> - Summarize the results with a written report.


#### <b>Data Set Summary & Exploration </b>

The data set for this project is available as pickled files. There are three sets of pickled files provided- one for each of training, testing and validation. These picked files are loaded and information about the dataset is extracted from them. This is done is <b> cell 2 and 3 </b> of the Python notebook.
>*  The length of the training data set = 34799
>* The length of the testing example = 12630
>* The shape of each image in the data set = 32x32x3 
>* The number of unique classes in the dataset = 43

#### <b>Data Set Visualisation</b>
Some of the test images as available in the data set can be seen below : </br>
</br>
</br>
<img src="testimages/test_10700.jpg" width="150" style="margin-right: 50px;"> <img src="testimages/test_11654.jpg" width="150" style="margin-right: 50px;">  <img src="testimages/test_11682.jpg" width="150" style="margin-right: 50px;">  <img src="testimages/test_11694.jpg" width="150"> 

In addition to this the count of each sign is calculated using the unique function available in Python and numpy and plotted using matlab functions.
The code for this is available in cell 4 of the IPython notebook.

![Bar Chart](testimages/barchart.png "Bar Chart Image")
 
#### <b> Preprocessing the dataset </b>

##### <b>Shuffling the images</b> - As a first step, the dataset is shuffled to ensure that there is good mix of different kinds of images in each batch run. This was done because without shuffling when the images from the training set was printed, it was seen that bunch of similar images occurred together in the dataset.

##### <b> Grayscaling the images</b> -  Each image in the dataset is a colored image. Thus images were converted to grayscale to ensure that the network has to operate on one channel instead of three channels. The formula for luminosity was used to grayscale the images as this provided a better estimate than simply averaging the RGB values.
More information on grayscaling can be obtained from the link :
<a href="https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/">https://www.johndcook.com/blog/2009/08/24/algorithms-convert-color-grayscale/</a> </br></br>
An example of an image before and after grayscaling is seen below: </br>
</br>

<img src="gray/gray.png" style="margin-right: 50px;"> <img src="gray/test_16.jpg" width="280">

##### <b>Normalization</b> - This was done as suggested in one of the lectures in tensor flow to maintain numerical stability. Initially I did not choose to normalize but using normalization improved the validation accuracy. 

The above pre processing steps were done for each of training, testing and validation dataset. The code for this is available in cells 5, 6, 7 and 8 of the IPython
notebook.

#### <b> Model Architecture </b>

My network is a convolutional neural network, as these tend to do very well with images. I mostly used the same architecture as the LeNet neural network did, with 2 convolutional layers and 3 fully connected layers. The convolutional neural network is defined in the 10th cell of the IPython notebook.

Grayscaling changed the input data to 32x32 from 32x32x3 and hence I had to reshape the input array so that it could be 32x32x1.
I initially used an epoch of value 10 with a batch size of 128. However, experimenting with different batch size between 100 to 250 I found a reasonable accuracy with a batch size of 160.

Changing the epoch from 10 to 20 did not provide any improvement in accuracy. However, using a significantly larger epoch like 40 provided a better convergence.

For the model hyperparameters, a mean of 0 and standard deviation/sigma of 0.1 was taken. These hyperparameters attempt to keep a mean of 0 and equal variance.
I utilized the AdamOptimizer from within TensorFLow to optimize, which seemed to do better than a regular Gradient Descent Optimizer.

I experimented also by adding a dropout in the convolutional network but this reduced my validation accuracy. I would assume that there was no issues with
overfitting of data and hence a dropout actually led to reduced data being trained and led to a decrease in the accuracy. Hence I decided not to use
dropout.

<table style="width:100%; align:center;">
  <tr>
    <th>Layer</th>
    <th>Description</th>    
  </tr>
  <tr>
    <td>Input</td>
    <td>32x32x3</td>    
  </tr>
  <tr>
    <td>Convolution</td>
    <td>1x1 stride,padding="VALID",output = 28x28x6</td>   
  </tr>
  <tr>
    <td>RELU</td>
    <td>output = 28x28x6</td>   
  </tr>
  <tr>
    <td>Max Pooling</td>
    <td>2x2 stride, output = 14x14x6</td>   
  </tr>
  <tr>
    <td>Convolution</td>
    <td>1x1 stride,padding="VALID",output = 10x10x16</td>   
  </tr>
  <tr>
    <td>RELU</td>
    <td>output = 10x10x16</td>   
  </tr>
  <tr>
    <td>Max Pooling</td>
    <td>2x2 stride, output = 5x5x16</td>   
  </tr>
  <tr>
    <td>Flatten</td>
    <td>output = 400</td>   
  </tr>
  <tr>
    <td>FullyConnected</td>
    <td>input = 400, output = 120</td>   
  </tr>
  <tr>
    <td>RELU</td>
    <td>output = 120</td>   
  </tr>
  <tr>
    <td>FullyConnected</td>
    <td>input = 120, output = 84</td>   
  </tr>
  <tr>
    <td>RELU</td>
    <td>output = 84</td>   
  </tr>
  <tr>
    <td>FullyConnected</td>
    <td>input = 84, output(Logits) = 43</td>   
  </tr>
</table>			

> **Note:** I did not split the training data into training and validation sets as the validation data that was provided gave reasonable accuracy results.

##### <b> My final model results were : </b>
Validation Accuracy = 94.6% </br>
Testing accuracy = 91.5%

#### <b> Testing model on new images </b>

Five German traffic signs were downloaded from the internet and ran using our network. The five images are as follows:

<img src="german_traffic_sign/NoEntry_Sign.jpg" width="150" style="margin-right: 50px;"> <img src="german_traffic_sign/Pedestrain_Sign.jpg" width="150" style="margin-right: 50px;">  <img src="german_traffic_sign/SpeedLimit_50_Sign.jpg" width="150" style="margin-right: 50px;">  <img src="german_traffic_sign/Stop_Sign.jpg" width="150" style="margin-right: 50px;"> <img src="german_traffic_sign/Yeild_Sign.jpg" width="150">

The five images I have added are: </br>
1: No Entry sign <br/>
2: Pedestrian sign <br/>
3: 50 Speed Limit <br/>
4: Stop Sign <br/>
5: Yield sign <br/>

The accuracy result is 100%. I read the signnames.csv file and compared the predicted labels that were obtained by training these values against the values
that were specified in signnames.csv for these images which can be seen in <b>cell 25</b> of the IPython notebook and the values match completely.

<table style="width:100%; align:center;">
  <tr>
    <th>Image</th>
    <th>Predicted Output</th>		
  </tr>
  <tr>
    <td>No Entry Sign</td>
    <td>No Entry Sign</td>    
  </tr>
  <tr>
    <td>Pedestrain Sign</td>
    <td>Pedestrain Sign</td>   
  </tr>
  <tr>
    <td>50 Limit Sign</td>
    <td>50 Limit Sign</td>   
  </tr>
  <tr>
    <td>Stop Sign</td>
    <td>Stop Sign</td>   
  </tr>
  <tr>
    <td>Yeild Sign</td>
    <td>Yeild Sign</td>   
  </tr>
 </table> 
 
One of the reasons for such a good accuracy result is that, I printed th training images and saw that a large number of images closely match to the traffic sign
images that I chose from the web. 

In order to confirm my resuts, I also replaced one of the traffic sign images with a random image and could see the accuracy going down.

From the soft max probability matrix output, it is also seen that each of the correct labels were predicted with the highest probability.

