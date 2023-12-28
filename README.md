# Model-Architecture-Design-for-Face-Recognition


Classical network architectures, such as AlexNet, LeNet and VGG-16 have played a vital role in shaping the foundation of deep learning and establishing benchmarks for image  recognition tasks. These architectures have paved the way for subsequent advancements in the field, showcasing 
the potential of deep neural networks in achieving remarkable accuracy. However, with the rapid growth of data 
availability and computing power, there is an increasing need to explore novel approaches and fine-tune existing 
models to achieve even higher performance
This Repo showcases the implementation of Above mentioned network on specific dataset to create another network by tuning the best model for our given dataset.

## Dataset

You can use any dataset for these networks. You just need to place the samples in a specific manner to implement supervised learning and since we are using images as samples so by Following the steps:

- You need to create directories of each individual object.
- make sure that all individual have same number of samples
- make sure to place all these directories in main `Dataset` directory place in main workspace.

## Implementation

All these models are implemented on framework of [Keras/Tensorflow](https://keras.io/). You need to have knowloedge of Keras to implement and modify model parameters for finetuning.

### ALEXNET

AlexNet demonstrated remarkable performance in image classification tasks, achieving a significant 
reduction in error rates compared to previous methods.

<p align = "center">
<img src = "https://github.com/AsdiIqbal/Model-Architecture-Design-for-Face-Recognition/blob/main/resources/as.PNG">
</p>

The architecture of AlexNet consists of 
multiple convolutional layers, followed by max-pooling layers and fully connected layers. It has a 
total of eight layers, with the first five layers being convolutional and the remaining three being fully 
connected.The net count of the parameters isas shown below:

```
================================================================
Total params: 67116162 (256.03 MB)
Trainable params: 67116162 (256.03 MB)
Non-trainable params: 0 (0.00 Byte)
________________________________________________________________
```

This net shows remarkable results on this dataset and show the testing accuracy of 97%

```
4/4 [==============================] - 0s 19ms/step - loss: 0.0998 - accuracy: 0.9700
LOSS : 0.09978172928094864
ACCURACY : 0.9700000286102295
```

### LENET

LeNet-5 is a pioneering convolutional neural network architecture developed by Yann LeCun, Leon 
Bottou, Yoshua Bengio, and Patrick Haffner in the 1990s. It was primarily designed for handwritten 
digit recognition and played a significant role in popularizing convolutional neural networks (CNNs) 
for image classification tasks. 

<p align = "center">
<img src = "https://github.com/AsdiIqbal/Model-Architecture-Design-for-Face-Recognition/blob/main/resources/lenet.PNG">
</p>

The architecture of LeNet-5 consists of seven layers,including three convolutional layers followed by two fully connected layers and two subsampling (pooling) layersThe net count of the parameters isas shown below:

```
================================================================
Total params: 61,706
Trainable params: 61,706
Non-trainable params: 0
________________________________________________________________
```

This net shows yet another good result of 92% of Test Accuracy.

```
4/4 [==============================] - 0s 20ms/step - loss: 0.1896 - accuracy: 0.9200
LOSS : 0.18956588208675385
ACCURACY : 0.9200000166893005
```


### VGG-16

The VGG-16 (Visual Geometry Group 16) is a deep convolutional neural network architecture that 
was developed by the Visual Geometry Group at the University of Oxford. It was designed to address 
the challenge of image classification and object recognition. 

<p align = "center">
<img src = "https://github.com/AsdiIqbal/Model-Architecture-Design-for-Face-Recognition/blob/main/resources/Capture.PNG">
</p>

The VGG-16 architecture is 
characterized by its depth and simplicity. It consists of 16 layers, including 13 convolutional layers 
and 3 fully connected layers. The net count of the parameters isas shown below:

```
================================================================
Total params: 33638218 (128.32 MB)
Trainable params: 33638218 (128.32 MB)
Non-trainable params: 0 (0.00 Byte)
________________________________________________________________
```

VGG shows a very poor result on this dataset and could hold up to 9% of test accuracy


```
4/4 [==============================] - 0s 88ms/step - loss: 2.3395 - accuracy: 0.0900
LOSS : 2.339451789855957
ACCURACY : 0.09000000357627869
```

## Architecture Design

Now that we have AlexNet as our base to start building our own CNN Architecture. In contrast with 
AlexNet, we would use (48,48,3) sized data set to get maximum efficiency from AlexNet by hyper tuning
its parameters like filters, kernel size, pool size, strides and padding.
Hyper-parameter tuning of CNNs are a tad bit difficult than tuning of dense networks due to the above 
conventions. This is because the Hp uses random search for the best possible model which in-turn may 
lead to disobeying few conventions, to prevent this from happening we need to design CNN architectures 
and then fine-tune hyper-parameters in Hp to get our best model.
tuning dense networks, wherein we give a set a value for all the hyperparameters and let the module 
decide which is best won’t work for tuning CNN. This is because at each layer the input dimensions 
decrease due to operations like convolution and max-pooling hence if we give a range of values for 
hyperparameters like stride, filter-size etc. there always exists a chance that Hps chooses a module which 
ends up in a negative dimension exception and stops before completion. Now to build such NNs we create 
architecture sheet an then choosing different models in tuning to get the best model out of them.
The same is done in this tuning as 5 different models were left on the choice of tuner to bring the best out 
of them.

```
 Layer (type)                Output Shape              Param #   
================================================================
 Conv_1 (Conv2D)             (None, 23, 23, 70)        1960      
                                                                 
 MaxPool2D_1 (MaxPooling2D)  (None, 23, 23, 70)        0         
                                                                 
 Conv_2 (Conv2D)             (None, 19, 19, 100)       175100    
                                                                 
 MaxPool2D_2 (MaxPooling2D)  (None, 17, 17, 100)       0         
                                                                 
 Conv_3 (Conv2D)             (None, 15, 15, 150)       135150    
                                                                 
 Conv_4 (Conv2D)             (None, 13, 13, 150)       202650    
                                                                 
 Conv_5 (Conv2D)             (None, 13, 13, 100)       135100    
                                                                 
 MaxPool2D_3 (MaxPooling2D)  (None, 12, 12, 100)       0         
                                                                 
 flatten (Flatten)           (None, 14400)             0         
                                                                 
 FC_Layer1 (Dense)           (None, 4096)              58986496  
                                                                 
 dropout (Dropout)           (None, 4096)              0         
                                                                 
 FC_Layer2 (Dense)           (None, 4096)              16781312  
                                                                 
 dropout_1 (Dropout)         (None, 4096)              0         
                                                                 
 FC_Layer3 (Dense)           (None, 4096)              16781312  
                                                                 
 dropout_2 (Dropout)         (None, 4096)              0         
                                                                 
 Out_Layer (Dense)           (None, 1000)              4097000   
                                                                 
 Out_Layer_2 (Dense)         (None, 10)                10010     
                                                                 
================================================================
Total params: 97,306,090
Trainable params: 97,306,090
Non-trainable params: 0
________________________________________________________________
```

After getting this best model so far, we get its hyperparameters train this model on our dataset as follows:

```
best_hps=Tuner.get_best_hyperparameters(1)[0]
new_Model=Tuner.hypermodel.build(best_hps)
history=new_Model.fit(MI_X_Train,MI_Y_Train,epochs=10,validation_split=0.2)
```

Surprisingly, we get pretty good result from this model.

```
4/4 [==============================] - 0s 80ms/step - loss: 0.1434 - accuracy: 0.9500
LOSS : 0.14342497289180756
ACCURACY : 0.949999988079071
```

## NOTE: 
- This model is still in testing phase. It is not inference-ready.
- Used Dataset is not public. You need to use your own dataset.


## References
-[A guide to build NN architecture](https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7?gi=75beb0daf8b5)
-[Different Type of Neural Network](https://www.analyticsvidhya.com/blog/2020/02/cnn-vs-rnn-vs-mlp-analyzing-3-types-of-neural-networks-in-deep-learning/)

