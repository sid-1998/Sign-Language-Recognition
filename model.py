
# coding: utf-8

# In[1]:


import numpy as np
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from keras.models import Sequential, save_model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os
import cv2


# In[2]:


path = './preprocessed_new/'


# In[3]:


gestures = os.listdir(path)


# In[4]:


dict_lables = {
    'A': 1,
    'B': 2,
    'C':3,
    'D':4,
    'E':5,
    'F':6,
    'G':7,
    'H':8,
    'I':9,
    'K':10,
    'L':11,
    'M':12,
    'N':13,
    'O':14,
    'P':15,
    'Q':16,
    'R':17,
    'S':18,
    'T':19,
    'U':20,
    'V':21,
    'W':22,
    'X':23,
    'Y':24,
    
}


# In[5]:


x, y = [], []
for ix in gestures:
    images = os.listdir(path + ix)
    for cx in images:
        img_path = path + ix + '/' + cx
        img = cv2.imread(img_path, 0)
        img = img.reshape((50,50,1))
        img = img/255.0
        x.append(img)
        y.append(dict_lables[ix])


# In[6]:


X = np.array(x)
Y = np.array(y)
Y = np_utils.to_categorical(Y)


# In[7]:


categories = Y.shape[1]


# In[8]:


X, Y = shuffle(X, Y, random_state=0)


# In[9]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)


# In[10]:


print(X_train.shape, X_test.shape)
print(Y_train.shape, Y_test.shape)


# In[11]:


model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(50,50,1), activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(16, kernel_size=(3,3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(categories, activation='softmax'))

model.summary()


# In[12]:


checkpoint = ModelCheckpoint('model-{val_acc:.4f}.h5', monitor='val_acc', verbose=0, save_best_only=True, mode='auto')


# In[13]:


model.compile(optimizer='Adam', metrics=['accuracy'], loss='categorical_crossentropy')


# In[14]:


history = model.fit(X_train, Y_train, batch_size=64, epochs=10, validation_data=[X_test, Y_test], callbacks=[checkpoint])

