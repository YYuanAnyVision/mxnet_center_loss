# mxnet_center_loss

This is a simple implementation of the *center loss* introduced by this [paper](http://ydwen.github.io/papers/WenECCV16.pdf) : *《A Discriminative Feature Learning Approach for Deep Face Recognition》*,*Yandong Wen, Kaipeng Zhang, Zhifeng Li, and Yu Qiao, Shenzhen* check their [site](http://ydwen.github.io/) 

[中文](https://pangyupo.github.io/2016/10/16/mxnet-center-loss/)

## Prerequisities

install mxnet

for visualization, you may have to install **seaborn** and **matplotlib**

> sudo pip install seaborn matplotlib

## code 

* **center_loss.py** implementation of the operator and custom metric of the loss
* **data.py** custom MNIST iterator, output 2 labels( one for softmax and one for center loss
* **train_model.py** copied from [mxnet example](https://github.com/dmlc/mxnet/tree/master/example/image-classification) with some modification
* **train.py** script to train the model
* **vis.py** script to visualise the result

## running the tests

### 1 set path of mxnet

  change **mxnet_root** to your mxnet root folder in **data.py**

### 2 train

* with cpu

  > python train.py --batch-size=128

* with gpu

  > python train.py --gpus=0

  or multi device( not a good idea for MNIST example here )

  > python train.py --gpus=0,1 --batch-size=256
  
then you can see the output by typing

`tail -f log.txt`

### 3 visualize the result

run

> python vis.py

  You will see something like right picture... Now compare it with the 'softmax only' experiment in left, all the samples are well clustered, therefor we can expect better generalization performance. But the difference is not fatal here(center loss does help with convergence, see the last figure), since the number of classes is actually the same during train and test stages. For other application such as face recognition, the potential number of classes is unknown, then a good embedding is essential. 

![center_loss](http://7xsc78.com1.z0.glb.clouddn.com/centerloss_example.jpg)



training log:

![train_log](http://7xsc78.com1.z0.glb.clouddn.com/training_log.png)