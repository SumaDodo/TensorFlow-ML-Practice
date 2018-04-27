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

