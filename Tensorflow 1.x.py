#!/usr/bin/env python
# coding: utf-8

# In[1]:


#%load_ext cudf.pandas
import pandas as pd
import tensorflow as tf


# In[2]:


get_ipython().system('pip show tensorflow')


# ### Constant

# In[3]:


hello_constant= tf.constant("Hello World")
hello_constant


# In[4]:


with tf.compat.v1.Session() as sess:
    output= sess.run(hello_constant)
    print(output)


# ### Tensor

# In tensorflow, data is not stored as integer, float, or strings. These values are encapsulated in an object called as tensor. In case of hello_constant= tf.constant("Hello World"), hello_constant is a 0-dimensional string tensor, but tensor comes in variety of sizes.

# In[5]:


a= tf.constant(45)  #0-dimension int32 tensor

b= tf.constant([1,2])  #1-dimension int32 tensor 

c= tf.constant([[1,2],[4,5]])  #2-dimension int32 tensor


# tf.constant() is one of the many Tensorflow operations we will use. The tensor returned by the tf.constant() is called as a constant tensor, because the value of the tensor never changes. 

# In[6]:


a


# ### Session 

# Session is a type of instance which tries to allocate CPU, memory and RAM to our execution. In Tensorflow 1.x there is a provision to allocate a location before exectuting something. Basically there was no dyanmic location of the resources. That's why we create the a Session.

# Tensorflow's api is build around the idea of a computational graph, a way of visualizing a mathematical process. Let's take a tensorflow code and turn into a graph.

# In the last session the tensor was passed into the session and it returned the result. What if we want to use a non-constant? this is where tf.placeholder() and feed_dict come into place. In this we will go over the basics of feeding data into Tensorflow.

# ### tf.placeholder()

# tf.placeholder() returns a tensor that gets its value from the data passed to the tf.session.run() function, allowing us to set the input right before the session runs.

# In[7]:


x=tf.placeholder(tf.string)

with tf.compat.v1.Session() as sess:
    output= sess.run(x, feed_dict={x:"Hello World"})
    print(output)


# In[8]:


x= tf.placeholder(tf.string)
y= tf.placeholder(tf.int32)
z= tf.placeholder(tf.float32)

with tf.compat.v1.Session() as sess:
    output_x= sess.run(x, feed_dict={x:"Test String", y:123, z:45.67})
    output_y= sess.run(y, feed_dict={x:"Test String", y:123, z:45.67})
    output_z= sess.run(z, feed_dict={x:"Test String", y:123, z:45.67})
    print(output_x)
    print(output_y)
    print(output_z)


# ### Tensorflow Maths

# In[9]:


x= tf.add(5,2)

with tf.compat.v1.Session() as sess:
    output= sess.run(x)
    print(output)


# In[10]:


x= tf.subtract(10,2)


# In[11]:


y=tf.multiply(2,4)


# ### Converting Types

# In[12]:


#tf.subtract(tf.constant(2.0),tf.constant(1))  #This will show error, we need to do our typecasting by ourself.

x=tf.subtract(tf.cast(tf.constant(4.0),tf.int32),tf.constant(1))  #we need to do a type casting


# In[13]:


with tf.compat.v1.Session() as sess:
    output= sess.run(x)
    print(output)


# In[14]:


x= tf.constant(10)
y= tf.constant(2)
z=tf.subtract(tf.divide(x,y),2)

with tf.compat.v1.Session() as sess:
    output= sess.run(z)
    print(output)


# ## Tensorflow Linear Function

# The most common operation in neural network is calculating the linear combination of inputs, weights, and biases. As a reminder, we can write the output of the linear as: 
# ## y = xW+b
# Here W is a matrix of weights connecting two layers. The output y , input x and biases b are all vectors.

# #### Weights and Bias in Tensorflow

# The goal of training a neural network is to modify the weights and biases to predict the best labels. In order to use weights and bias, we need a Tensor that can be modifed. This leaves out tf.placeholder() and tf.constant(), since those tesor cant be modified. This is where tf.Variable() comes in place.

# ### Variable

# The tf.Variable() creates a tensor that can be changed or modified in a run time, much like a python variable.

# In[15]:


x= tf.Variable(27)


# The above tensor stores it's state in the session, so we must initialize the state of the tensor manually. we will use the tf.global_variables_initializer() function to the intialize the state of the Variables tensors

# In[16]:


init= tf.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)


# The tf.global_variables_initializer() call return an operation that will initialize all Tensorflow variables from the graph.

# We call the operation using a session to initlialize all the variables as shown above. Using the tf.Variable class allows using to the change the weights and bias, but initial values should be choosen.
# 
# Initializing the weights with the random numbers from a normal distribution is good practice. Randomizing the weights helps the model from being stuck in the same place every time we train.
# 
# Similarly, choosing weights from a normal distribution prevents any one weight from overwhelming other weights. We will use the tf.truncated_normal() function to generate the random numbers from a normal distribution.

# ### tf.truncated_normal()

# In[17]:


n_feature= 10
n_labels= 5
weights= tf.Variable(tf.truncated_normal((n_feature, n_labels)))  # It will create 10*5 matrix. 10 rows, 5 column.


#  The function returns a tensor with random values from a normal distribution whose magnitude is no more than 2 standard deviation form the mean.

# Since the weights are already helping prevent the model from getting stuck , we dont need to randomize the bias. Lets use a simple solution of setting bias to 0.

# ### tf.zeros()

# In[18]:


n_labels=5
bias= tf.Variable(tf.zeros(n_labels))
bias


# ## Softmax

# softmax helps to find out the linear probability of occurance of any number.
# The formula of finding the softmax is:
# ### e^y/(∑e^y)

# It is a perfect function to use as the output activation for a network predicting multiple classes.

# if there are 3 numbers, 2.0,1.0 and 0.2 then the softmax of 2.0 is:
# 
# e^2.0/(e^2.0+e^1.0+e^0.2)
# 
# that would be: 0.65.
# Which means that, the probablity of occurance of 2.0 is 0.65.

# In[27]:


x= tf.nn.softmax([2.0,1.0,0.2]) #the function takes in logits and returns softmax activation.

with tf.compat.v1.Session() as sess:
    output= sess.run(x)
    print(output)


# In[26]:


def softmax():
    output= None
    logit_data= [2.0,1.0,0.2]
    logits= tf.placeholder(tf.float32)  #using placeholder
    
    softmax= tf.nn.softmax(logit_data)
    
    with tf.compat.v1.Session() as sess:
        output= sess.run(softmax, feed_dict={logits:logit_data})
    
    return output

print(softmax())


# In[ ]:


"""
#To create a cross entropy function in Tensorflow, we need two more function:
-tf.reduced_sum()
-tf.log()
"""


# ### Reduce Sum

# In[29]:


x= tf.reduce_sum([1,2,3,4,5]) #15


# ### Natural Log

# In[31]:


l= tf.log(100.0)  #4.60517, log with base e.
l


# In[32]:


softmax_data=[0.7,0.2,0.1]
one_hot_data=[1.0,0.0,0.0]

softmax= tf.placeholder(tf.float32)
one_hot=tf.placeholder(tf.float32)

#cross entropy from session
cross_entropy= -tf.reduce_sum(tf.multiply(one_hot_data,tf.log(softmax_data)))

with tf.compat.v1.Session() as sess:
    output= sess.run(cross_entropy,feed_dict={one_hot:one_hot_data, softmax:softmax_data})
    print(output)


# In[ ]:




