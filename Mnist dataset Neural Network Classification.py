#!/usr/bin/env python
# coding: utf-8

# pip install keras

# In[1]:


import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 


# In[2]:


pip install tensorflow


# In[3]:


mnist = keras.datasets.mnist
(X_train_full, y_train_full), (X_test,y_test) = mnist.load_data()


# In[4]:


X_train_full.shape


# In[5]:


X_test.shape


# In[6]:


X_train_full[0]


# In[7]:


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(10,10))
a=0

for i in range(3):
    for j in range(3):
        axes[i,j].imshow(X_train_full[a], cmap=plt.get_cmap('gray'))
        a=a+1

plt.show()


# In[8]:


X_valid, X_train = X_train_full[:5000] / 255, X_train_full[5000:]/255
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test/255


# In[9]:


X_valid[0]


# In[10]:


y_test


# In[11]:


class_names = ["0","1","2","3","4","5","6","7","8","9"]


# In[12]:


class_names[y_train[8]]


# In[13]:


plt.imshow(X_train[8], cmap=plt.get_cmap('gray'))


# In[14]:


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation = "relu"))
model.add(keras.layers.Dense(100, activation = "relu"))
model.add(keras.layers.Dense(10, activation = "softmax"))

#Sigmoid: probabilities produced by a Sigmoid are independent.
#Softmax: the outputs are interrelated. The sum of all outputs are 1.
        


# In[15]:


model.summary()


# In[16]:


model.layers


# In[17]:


# https://keras.io/api/losses/
# https://keras.io/api/optimizers/#available-optimizers
# https://keras.io/api/metrics/

model.compile(loss="sparse_categorical_crossentropy", optimizer="sgd", metrics=["accuracy"])


# In[18]:


X_train.shape


# In[19]:


# batch = The batch size is a hyperparameter that defines the number of samples to work through before updating the internal model parameters.
# epochs = how many times to do a pass over all the dataset

# https://keras.io/api/models/model_training_apis/#fit-method
history = model.fit(X_train, y_train, epochs=30, validation_data = (X_valid, y_valid), batch_size=32) # also possible to use 


# In[20]:


import pandas as pd

pd.DataFrame(history.history).plot(figsize=(15,8))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()


# In[21]:


model.evaluate(X_test, y_test)


# In[22]:


model.predict(X_test)


# In[23]:


y_prob = model.predict(X_test)
y_classes = y_prob.argmax(axis=-1)
y_classes


# In[24]:


confusion_matrix = tf.math.confusion_matrix(y_test, y_classes)


# In[25]:


import seaborn as sb    

# ax = plt.figure(figsize=(8, 6))
fig = sb.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Greens')  #annot=True to annotate cells, ftm='g' to disable scientific notation

# labels, title and ticks
fig.set_xlabel('Predicted labels')
fig.set_ylabel('True labels')
fig.set_title('Confusion Matrix')
fig.xaxis.set_ticklabels(class_names) 
fig.yaxis.set_ticklabels(class_names)
fig.figure.set_size_inches(10, 10)


plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




