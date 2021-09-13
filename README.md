# Fully-Connected Model from Scratch

## Goal
Classify input data by doing backpropagation and renewing model parameters. Models are built from scratch, **without** using deep learning framework such as Tensorflow and PyTorch.

## Testing Cases
There are 2 cases. Each case contains 2 classes
### 1. Linear
![Linear](img/linear.png) 
### 2. XOR
![XOR](img/xor.png)

## Model
### Neural Network
2 hidden layers & 1 output layer.  
![NN Structure](img/nn_structure.svg)

### Weight Initialization
Drawn from  
![Weight Initialization](img/weight_init.png)

### Activation Function
Sigmoid  
![Sigmoid](img/sigmoid.png)  
Derivative of sigmoid
![Derivative of Sigmoid](img/sigmoid_der.png)

### Loss Function
MSE Loss  
![MSE Loss](img/mse.png)

### Backpropagation
![bp1](img/bp1.png)
![bp2](img/bp2.png)

## Testing Results
### Linear Case
Settings:
- First/Sec hidden layer neurons: 3 & 3
- Learning rate = 1
- Batch size: 1/10 dataset size

Results:
- Testing loss: 0.021464
- Testing accuracy: 97%

![](img/res_linear1.png)
![](img/res_linear2.png)
![](img/res_linear3.png)
Weights before training:  
![](img/res_linear4.png)
Weights after training:  
![](img/res_linear5.png)

### XOR Case
![](img/res_xor1.png)
![](img/res_xor2.png)
![](img/res_xor3.png)
Weights before training:  
![](img/res_xor4.png)
Weights after training:  
![](img/res_xor5.png)

## Other Experiments

In this section, linear dataset are used for the following experiments. Basic setttings:

### Different Learning Rates

Settings:

- First & Second hidden layer neurons: 3 & 3
- Second hidden layer neurons: 3
- Batch size ratio: 1 （full-batch)

Model learns faster with higher learning rate. However, it's prone to be unstable with high learning rates. The blue curve drops in early stage but it swung for a period of time.

![](img/o_e_1.png)

### Different Batch Size

Settings:

- First & Second hidden layer neurons: 3 & 3
- Learning rate: 0.1

P.S. "batch size ratio" times total number of data is batch size.

Training with a lower batch size performs better after the same epochs since model with a lower batch size updates more in a single epoch. In the graph, models with lower batch size ratio drop earlier but swing a lot. 

![](img/o_e_2.png)

### Number of Neurons in Hidden Layers

Settings:

- First & Second hidden layer neurons: 3 & 3
- Second hidden layer neurons: 3
- Batch size ratio: 1 （full-batch)

Loss of 5x3 model drops fastest and loss of 2x3 drops slowest. However, 3x3 drops faster than 4x3. So, it's difficult to distinguish whether neuron number is the key factor.

![](img/o_e_3.png)

From the next two graphs, we may consider more complicated models fit this linear dataset more.

![](img/o_e_4.png)

![](img/o_e_5.png)

