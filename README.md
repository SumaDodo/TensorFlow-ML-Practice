Tensorflow Pratice programs

**Resource Links:**  
*Machine Learning:*  
  1. [An Introduction to Machine Learning Theory and Its Applications: A Visual Tutorial with Examples](https://www.toptal.com/machine-learning/machine-learning-theory-an-introductory-primer)
```
- Explanation about measurement of wrongness.
- Cost Function
- Regression vs Classification
```
  2. [A Gentle Guide to Machine Learning](https://monkeylearn.com/blog/gentle-guide-to-machine-learning/)  
```
  - Principle of Occam’s razor followed by algorithms in Machine Learning
```
  3. [A visual introduction to machine learning](http://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
  ```
  - Wonderful inspiration and introduction to R2D3
  - A nice explanation for classification problem
  ```

*Deep Learning and Neural Network*

  1. [Introduction to Neural Networks](http://www.cs.stir.ac.uk/~lss/NNIntro/InvSlides.html)
  2. [Image Recognition with Deep Learning](https://medium.com/@ageitgey/machine-learning-is-fun-part-3-deep-learning-and-convolutional-neural-networks-f40359318721)
  ```
  - Image Recognition 
  - Convolutional Neural Network
  ```
**Getting to know TensorFlow:**  
The computations in tensorflow are considered as a flow of data through the graph with nodes being computation units 
and edges being a flow of tensors i.e. multi-dimensional arrays. 

Before the start of the execution, tensorflow builds the computation graph. When the nodes are defined, the graph is not executed. 
The execution happens after the complete graph is assembled. It is then deployed and executed in a "session". Session is the run-time environment which binds the hardware it is going to run in.

This makes an easy visualization of the problems. 

TensorFlow is a low-level computation library which uses simple operators like add and matrix multiplication to implement an algorithm. Because of its low-level computation,  TensorFlow is comparable to Numpy.

**TensorFlow and Automatic Differentiation**

Automatic Differentiation plays a vital role in applications that are dependent on neural networks. As the name suggests, automatic differentiation is all both computing the derivatives automatically. The program is broken down into small pieces and TensorFlow calculates the derivatives from the computation graph by using the chain rule.

This is helpful, especially in backpropogation. We don't want to have to hand-code new variation of backpropagation everytime we are experimenting with a new arrangement of neural networks. Every node in Tensorflow has an attached gradient operation which calculates derivatives of input with respect to output. Thus the gradients of parameters are calculated automatically during backpropagation.
Automatic differentiation is one helpful tool that reduces the tension about implementation errors, especially during backpropagation. 
Automatic Differentiation in itself is a very important topic in Numerical Optimization and it is discussed in detail [here](https://github.com/SumaDodo/Numerical-Optimization/tree/master/Automatic_differentiation).

**Understanding Tensorflow with Computational Graph:**

There are five important components in a tensorflow computational graph:  
  1. **Placeholders:** Variables that are used in place of inputs to fee to the graph.  
  2. **Variables:** Model variable that are going to be optimized to make the model perform better.  
  3. **Model:** Mathematical function that calculates output based on placeholder and model variables.  
  4. **Loss Measure:** Guide for optimization of model variables.
  5. **Optimization:** Update model for tuning model variables.  
  
We can understand the above listed concepts by implementing a linear classifier to classify handwritten digits from MNIST dataset:

  *Step 1: Defining Placeholders*  
  Our placeholders are the input values of images that are stored as vectors and their labels.
  ```
  #Step 1:
#Defining the Placeholders
with tf.name_scope('Input'):
    #Attributes: datatype, shape and name
    x = tf.placeholder(tf.float32,[None, 784],name = "x")
    y_true = tf.placeholder(tf.float32,[None,10],name = "labels")
  ```
   *Step 2: Variables*  
   Variables are stateful nodes that are used to store model parameters. In our program we define two variables:
   ```
   Weights and bias
   ```
   weights variable is a 2 dimensional tensor of size input vector size by output vector size (784×10). 
   We initialize the tensor to have random numbers from Gaussian distribution.
   
   Bias is a 1 dimensional vector of size output vector, which is 10. We initialize it to zeros.
   ```
   #Step 2: 
#Defining the Variables
with tf.name_scope('Weights'):
    weights = tf.Variable(tf.random_uniform([784,10],-1,1),name="weights")
with tf.name_scope('Biases'):
    biases = tf.Variable(tf.zeros([10]),name="biases")
   ```
    
   *Step 3: Model*  
   Model is a mathematical function that maps inputs to outputs.
   For the classifier here we use simple matrix multiplication.
   ```
   logits = tf.matmul(X,weights) + biases
  ```
  The output here is stored in the varibale logits and to convert the output to probability distribution, we apply softmax.
  ```
  y_pred = tf.nn.softmax(logits)
  ```
  And from here we pick the class with the highest probability.
  ```
  y_pred_cls = tf.argmax(y_pred,dimension=1)
  ```
  Thus, putting together all of this we have:
  ```
  #Step 3:
# Definining the model
# Mathematical function that calculates output based on placeholder and variables
with tf.name_scope('LinearModel'):
    logits = tf.matmul(x,weights) + biases
    y_pred = tf.nn.softmax(logits)
  ```
  *Step 4: Loss Measure*  
  Cross entropy meansure is used as loss measure. Our goal here is to minimize the cross entropy loss as much as possible. 
  ```
  #Step 4:
# Definining cost measure
with tf.name_scope('CrossEntropy'):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y_true)
    loss = tf.reduce_mean(cross_entropy)
  ```
  *Step 5: Optimization*  
  Here we are using Gradient Descent as our optimization method.
  ```
  train_step = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(loss)
  ```
  The minimize(loss) does two things:  
  1. It computes the gradient  
  2. It applies the gradient update to all the variables  
  ```
  #Step 5:
# create optimizer
with tf.name_scope('GDOptimizer'):
    train_step = tf.train.GradientDescentOptimizer(learning_rate = 0.1).minimize(loss)
  ```
  Putting together all of these we can check the complete flow through tensorboard:
  ![Graph](https://github.com/SumaDodo/TensorFlow/blob/master/graph_large_attrs_key%3D_too_large_attrs%26limit_attr_size%3D1024%26run%3D%20(1).png)
