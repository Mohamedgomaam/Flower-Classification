# Flower-Classification

Our Task:
We are classifying 104 types of flowers based on their images drawn from five different public datasets using CNN (Convolution Neural Network).

Step1
Firstly, we downloaded the given dataset from Microsoft teams. We will show a sample of the dataset .  

![image](https://github.com/user-attachments/assets/48a1cfbb-123c-4441-a3df-6b5f4ec87c66)

Now we will display the number of sample in each class.

![image](https://github.com/user-attachments/assets/3cb2f502-05cc-43eb-b0c0-2c9f5b0ef9ae)

(Hint!!  Before we split the data we had to change the name of each folder for example folder 0 will be pink primrose as given in the data definition, the reason for that change because when we read the data in python it rank the folder 0 and 1 and then 10 and latter when we index them for example folder 10 will take the index 2 and it will be misleading later when we calculate the accuracy or doing any evaluation so for insurance we changed each folder to its real name, and also sort the list called class_names because if its not sorted it will also give an error and misleading for classification . Now we are ready to work on the data).
We split the data into 90 percent for training and the rest for validation.
Before we put the data into out model , we must first scale the pixels of each image so it makes our training much faster. We also reshaped our image from 192*192*3 to 224*224*3 for later purpose in the transfer learning.

Step2
Secondly we created our model , before we settled on our model we tried many models one of them consist of 3 convolution layers and two fully connected layers after we try it we found that it has a big problem which is overfitting because of high complexity so after many trials we settled on the following model. We will illustrate some functions we used to be clear further when we illustrate our model.

Batch Normalization
it normalizes the contributions to a layer for every mini-batch ,and it  allows us to increase the speed of training (computational time) by setting higher learning .
Max Pooling
It is a max pooling operation which calculates the largest or maximum value in every patch and the feature map so why we used it because it returns  the most prominent features of the feature map, and the returned image is sharper than the original image.

![image](https://github.com/user-attachments/assets/fe9c0308-d916-4fb6-adb1-cd61fce2d481)

The model
our model consists of 8 convolution (3 of them is 1*1 kernel) layers our first layer consists of 16 kernel of 7*7 dimension we chose 7*7 kernel because we can extract many features as we can for the first low level features then we used the batch normalization. after that we used maxpooling(2*2 pool size with stride =2) .
We will repeat this again but when we go deeper we increase the number of kernels .
The second layer consists of 64 kernels of 3*3 dimension we also chose 3*3 dimension because of extremely longer training time consumed and expensiveness, we no longer use such large kernel sizes in the following layers so 3*3 is the optimal kernel for the small size kernels then we used batch normalization and maxpooling the same as we mentioned in the last model.
In the upcoming layers as we see in figure 1 it has the same structure but it has three 1*1 kernels .1*1 kernels is very important as it control the number of feature maps so it enables dimensionality reduction by reducing the number of filters whilst retaining important, feature-related information by this it makes the model computational time much faster . 




Our model architecture:
Figure1

![image](https://github.com/user-attachments/assets/22797079-d6b9-4b8b-9754-2366e63a6bbf)


Now will show our model accuracy:
The first 10 epochs

![image](https://github.com/user-attachments/assets/0d8faa49-988f-463f-a7cd-bed8fc1d0f3d)


Another 10 epochs 

![image](https://github.com/user-attachments/assets/b3e22d20-475d-4b05-9996-78743b62e3d6)

We used two famous architecture which are google net and resnet for transfer learning and ensemble .
Transfer learning
Transfer learning generally refers to a process where a model trained on one problem is used in some way on a second related problem. For example, we can use the googlenet architecture in our dataset because it is pretrained on a similar but bigger data which is the image net .we untrained the layers of the googlenet because as we mentioned it is pretrained model on a similar data after that we put our layers which is our fully connected layers similarly we will do the same thing with the resnet model
Googlenet
GoogLeNet is a convolutional neural network that is 22 layers deep. It has been trained on over a million images and can classify images into 1000 object categories.

![image](https://github.com/user-attachments/assets/abaf47dd-a0b8-470d-81a3-44eef83f7891)

Inception module
 design a good local network topology (network within a network) and then stack these modules on top of each other, it is an image model block that aims to approximate an optimal local sparse structure in a CNN. it allows for us to use multiple types of filter size, instead of being restricted to a single filter size, in a single image block, which we then concatenate and pass onto the next layer.

Our accuracy using google net:
The first 5 epochs:  

![image](https://github.com/user-attachments/assets/5c408e31-e5e9-4ec7-848c-9b8a8f63cede)

The second 5 epochs:

![image](https://github.com/user-attachments/assets/05fd6fd3-ec64-4154-b995-191ddb5b156b)

Resnet
A Residual Neural Network  is a deep learning model in which the weight layers learn residual functions with reference to the layer inputs. We used resnet50 which is consists of 50 layers.  

![image](https://github.com/user-attachments/assets/f84b8408-e37d-4c57-a037-9c11b5d8496d)

Residual block:A residual block is a stack of layers set in such a way that the output of a layer is taken and added to another layer deeper in the block.


Our accuracy using Resnet:
The first 10 epochs:

![image](https://github.com/user-attachments/assets/f2617d5f-5eff-443d-9812-2ddb853f7466)

Our second 10 epochs

![image](https://github.com/user-attachments/assets/a59243a2-37fb-4de6-aa4c-bf8c413ad77b)


Ensamble
Ensamble learning combines several individual models to obtain better generalization performance. For example we will use our model and 2 famous architecture we will take the average of the three output and make the final prediction.

Ensamble testing

![image](https://github.com/user-attachments/assets/c1623809-9470-42a9-aab9-a443ada2d82a)

![image](https://github.com/user-attachments/assets/a2f0e90e-a1bc-4e32-b0c9-7b947e977a38)
 
Step3
Now will compute our model performance on the test data by 3 things the accuracy , Macro F-Score and the confusion matrix.
Macro F-Score
When we receive an image our model gives 104 predict f score takes the 104 predict and then sort them descending and take the top five probabilities and if the true label in this five it counted as a correct prediction.
Our model testing:

![image](https://github.com/user-attachments/assets/9c45c42b-dd3e-45b7-a406-2cc67841aea8)

![image](https://github.com/user-attachments/assets/3015e199-1e28-4cbd-ac13-f02e8a0817af)


google net testing:

![image](https://github.com/user-attachments/assets/bc58a242-8ccd-4ad6-a65a-929ec33b0dce)

![image](https://github.com/user-attachments/assets/86354f8c-c274-4d42-8fbd-f5bff369d167)
 
Resnet testing:

![image](https://github.com/user-attachments/assets/47bd1bd9-43b2-4b7d-8c01-160d4f0633ea)

![image](https://github.com/user-attachments/assets/306f4229-4287-4108-a035-aef62763b678)
 
Ensambled model:

![image](https://github.com/user-attachments/assets/d91f535a-bbcc-4cfd-94d3-82b86630d0ee)

![image](https://github.com/user-attachments/assets/04b2bd25-fd04-4963-b783-fc7570d430c3)


Confusion Matrix:
A confusion matrix is a table that is used to define the performance of a classification algorithm. A confusion matrix visualizes and summarizes the performance of a classification algorithm ,so It shows how many true predict and false predictions our models made so we can identify the complex classes.
So now we will print the confusion for each model
1-our architecture :

![image](https://github.com/user-attachments/assets/3618e8c1-615e-4c7f-b667-fa4a77db6a67)

we can see that  the confusion classes here is Black-eyed Susan ,Wild geranium  ,Wild rose ,Wild flower and there is more in the picture

2-google_model :

![image](https://github.com/user-attachments/assets/e8dfd21b-2f31-4002-9e5d-b700f17fef4b)

we can see that  the confusion classes here is Black-eyed Susan ,Wild geranium ,Wild pansy ,Wild rose ,Wild flower

3-Resnet model:

![image](https://github.com/user-attachments/assets/16277775-aeca-4ba2-b825-2cf832d13c83)

we can see that  the confusion classes here is rose ,pink primrose ,iris ,common dandelion , Wild geranium ,Wild pansy ,Wild rose ,Wild flower


3-Ensamble model:

![image](https://github.com/user-attachments/assets/460fcb01-64ee-4dc1-bfed-28d0592d549d)

first we can see that it gives less error and less confusion classes which is Wild rose ,Wild geranium

From the 4 confusion matrix we can conclude that the most confusion classes is Wild geranium ,Wild pansy ,Wild rose ,Wild flower

Now with the plots of the performance of each model:
1-our-model:

![image](https://github.com/user-attachments/assets/cefd7e37-f5b5-4714-b332-b3f1f32df204)

![image](https://github.com/user-attachments/assets/225d414c-d87b-41c4-924f-f61eca034104)

 
We can conclude here the model is doing good in the accuracy and  loss in train and validation
2-googlenet model:

![image](https://github.com/user-attachments/assets/c1bf142d-fe18-42b1-8286-9b86edc6149e)

![image](https://github.com/user-attachments/assets/4c8a0e37-84a9-4fc5-b1be-66d023e90805)
 
We can conclude here that if we train the model for more epochs it might fall in overfitting the train data because as we can see the loss increase at the end and the accuracy decrease as well so it might not be more general model if we train it more (we actually try to train it more but it fall in overfitting which gives us a bad performance)
3-resnet model:

![image](https://github.com/user-attachments/assets/95ed39ca-d328-43d9-aff2-12465fca9a49)

![image](https://github.com/user-attachments/assets/b19c95fc-415e-43e2-8bed-d7a25f4b4714)
 
We can conclude here if we train the model for more epochs it will fall in overfitting problem that’s why we trained for 10 epochs (we actually train it for more epochs but it give a bad performance )
Now with some classification predict 
We will classify using resnet_model using images from wild geranium
So given this image  

![image](https://github.com/user-attachments/assets/fcc46be7-9c2d-4edf-a505-4035f7a5f4da)

It give

![image](https://github.com/user-attachments/assets/cd9ca83a-8c6d-4155-8014-cfbba23baf71)

And when we give it the image  

![image](https://github.com/user-attachments/assets/00d92229-d701-4bc3-909c-a76658d9e375)

It give

![image](https://github.com/user-attachments/assets/38da135a-c339-45ea-b9b5-06ce7f98c8ff)

 
conclusion

Google net was much better than resnet as it gets the highest accuracy in less number of epochs in training as well in testing it gets better accuracy than resnet which means that googlenet was better generalization also, on the other hand, our model was the best in training according to the accuracy and also the computational time but it failed to be generalized as in testing googlenet was better ,so when we combined them all in Ensemble model we get the best model from the three.

