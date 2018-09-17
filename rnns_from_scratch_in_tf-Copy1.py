
# coding: utf-8

# ## Building Seq2Seq RNN in tf from scratch

# tutorial taken from: https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767  
# working with __tf.\_\_version\_\___=1.2.1; __np.\_\_version\_\___=1.14.2

# In[22]:

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd

from IPython.display import Image


# In[6]:

tf.__version__


# ### Params

# In[34]:

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15
state_size = 4
num_classes = 2
echo_step = 3 # shift between x and y
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length; print("num_batches:", num_batches)


# - __num_epochs__: num_epochs
# - __total_series_length__: full len of the original sequence
# - __truncated_backprop_length__: number of rnn previous steps to roll-back in BPTT (?)
# - __state_size__: inner state vector size
# - __num_classes__: binary i/o sequences
# - __echo_step__: the shift between x and y sequences
# - __batch_size__: batch_size
# - __num_batches__: num_batches

# ### Generate Data

# we generate data as:
# - __x__: a random binary (0/1) sequence of length __total_series_length__
# - __y__: we shift x by __echo_step__ indices (shift with rotation: i.e., last elements that are ommited will be added at the beginning of they sequence by the np.roll function)

# In[16]:

# # for clear understanding of generateData, run this cell:
# x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))

# print(pd.Series(x).value_counts())

# y = np.roll(x, echo_step)
# y[0:echo_step] = 0

# print(x[:10])
# print(y[1:11])
# print(y[:10])


# In[3]:

def generateData():
    x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0

    x = x.reshape((batch_size, -1))  # The first index changing slowest, subseries as rows
    y = y.reshape((batch_size, -1))

    return (x, y)


# In[19]:

x, y = generateData()
x.shape, y.shape


# In[32]:

# Image(filename="rnn_to_batches.png")


# ### Build Graph

# #### Placeholders

# In[45]:

with tf.name_scope(name="input") as scope:
    batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length], name="batchX_placeholder")
    batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length], name="batchY_placeholder")

    init_state = tf.placeholder(tf.float32, [batch_size, state_size], name="init_state")


# In[46]:

print("batchX_placeholder:", batchX_placeholder)
print("batchY_placeholder:", batchY_placeholder)
print("init_state:", init_state)


# #### Variables

# In[48]:

with tf.name_scope(name="weights") as scope:
    W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32, name="W")
    b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32, name="b")

    W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32, name="W2")
    b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32, name="b2")


# In[49]:

print("W:", W)
print("b:", b)
print("W2:", W2)
print("b2:", b2)


# In[51]:

# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[74]:

class A():
    def __init__(self, a1="a1", a2="a2"):
        assert(type(a1)==str and type(a2)==str)
        self.a1 = a1
        self.a2 = a2
    def print_me(self):
        print(self.a1, self.a2)
        
class B():
    def __init__(self, b1=1, b2=2):
        assert(type(b1)==int and type(b2)==int)
        self.b1 = b1
        self.b2 = b2
    def print_me(self):
        print(self.b1, self.b2)


# In[75]:

class C():
    def __init__(self, subclass_type="a", params_dict={"a1":"a1", "a2":"a2"}):
        if subclass_type=="a":
            self.subclass = A(**params_dict)
        else:
            self.subclass = B(**params_dict)
            
    def print_me(self):
        self.subclass.print_me()


# In[76]:

params_dict={"a1":"a1", "a2":"a2"}
a = A(**params_dict)
a.print_me()


# In[77]:

c = C()
c.print_me()


# In[106]:

class D():
    def __init__(self, subclass_type, **kargs):
        for key in kargs.keys():
            setattr(self, key, kargs[key])
        
        if subclass_type=="a":
            self.subclass = A(kargs["a1"], kargs["a2"])
        elif subclass_type=="b":
            self.subclass = B(kargs["b1"], kargs["b2"])
    def print_me(self):
        self.subclass.print_me()


# In[107]:

# d = D(subclass_type="a", a1="a1", a2="a2")

d = D(subclass_type="b", b1 = 2, b2 = 3, b3 = 4)

d.print_me()


# In[108]:

d.b3


# In[109]:

d.b3


# In[105]:

D.b1


# In[128]:

from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.base import BaseEstimator, ClassifierMixin


# In[146]:

svc = SVC()
svc.get_params()


# In[129]:

X = np.random.randn(100, 5)
y = np.random.randint(3, size=(100,))
X.shape, y.shape


# In[143]:

class SVCC(BaseEstimator, ClassifierMixin):
    def __init__(self, **kargs):
        for key in kargs.keys():
            setattr(self, key, kargs[key])
    def fit(self, X, y):
        self.svc = SVC(C=self.C)
        self.svc.fit(X, y)
        return self
    def predict(self, X):
        return self.svc.predict(X)
    def score(self, X, y):
        return self.svc.score(X, y)


# In[142]:

svc = SVCC(C=1)
svc.set_params(C=1)


# In[ ]:




# In[132]:

rs = RandomizedSearchCV(estimator=svc, cv=3, param_distributions={"C":[1,10,100]}, n_iter=3)


# In[133]:

rs.fit(X,y)


# In[ ]:




# In[ ]:



