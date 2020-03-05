# =============================================================================
# Import Libraries
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns
sns.set(style="whitegrid")
import os
import glob as gb
import cv2
import tensorflow as tf
import keras

# =============================================================================
# Read Data
# =============================================================================
trainpath = 'lego_train/'
testpath = 'lego_test/'

# =============================================================================
# number of images on lego_train file
# =============================================================================

for folder in  os.listdir(trainpath + 'lego_train') : 
    files = gb.glob(pathname= str( trainpath +'lego_train//' + folder + '/*.jpg'))
    print(f'For training data , found {len(files)} in folder {folder}')
# =============================================================================
#     # number of images on lego_test file
# =============================================================================
    
for folder in  os.listdir(testpath +'lego_test') : 
    files = gb.glob(pathname= str( testpath +'lego_test//' + folder + '/*.jpg'))
    print(f'For testing data , found {len(files)} in folder {folder}')
    
    
# =============================================================================
# Check images for lego_train
# =============================================================================
size = []
for folder in  os.listdir(trainpath +'lego_train') : 
    files = gb.glob(pathname= str( trainpath +'lego_train//' + folder + '/*.jpg'))
    for file in files: 
        image = plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()


# =============================================================================
# Check images for lego_test
# =============================================================================

size = []
for folder in  os.listdir(testpath +'lego_test') : 
    files = gb.glob(pathname= str( testpath +'lego_test//' + folder + '/*.jpg'))
    for file in files: 
        image = plt.imread(file)
        size.append(image.shape)
pd.Series(size).value_counts()

# =============================================================================
# Creat X and y
# =============================================================================
code = {'2x1':0 ,'2x2':1,'2x4':2}

def getcode(n) : 
    for x , y in code.items() : 
        if n == y : 
            return x
s=100
X_train = []
y_train = []
for folder in  os.listdir(trainpath +'lego_train') : 
    files = gb.glob(pathname= str( trainpath +'lego_train//' + folder + '/*.jpg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (s,s))
        X_train.append(list(image_array))
        y_train.append(code[folder])
        
print(f'we have {len(X_train)} items in X_train')

# =============================================================================
# plot X and y (train)
# =============================================================================
plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_train),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_train[i])   
    plt.axis('off')
    plt.title(getcode(y_train[i]))
    
    
# =============================================================================
# X and y test
# =============================================================================

s=100
X_test = []
y_test = []
for folder in  os.listdir(testpath +'lego_test') : 
    files = gb.glob(pathname= str( testpath +'lego_test//' + folder + '/*.jpg'))
    for file in files: 
        image = cv2.imread(file)
        image_array = cv2.resize(image , (s,s))
        X_test.append(list(image_array))
        y_test.append(code[folder])
        
print(f'we have {len(X_test)} items in X_test')    
# =============================================================================
# plot X and y (test)
# =============================================================================

plt.figure(figsize=(20,20))
for n , i in enumerate(list(np.random.randint(0,len(X_test),36))) : 
    plt.subplot(6,6,n+1)
    plt.imshow(X_test[i])   
    plt.axis('off')
    plt.title(getcode(y_test[i]))
    
    
# =============================================================================
# Convert to array
# =============================================================================
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

print(f'X_train shape  is {X_train.shape}')
print(f'X_test shape  is {X_test.shape}')
print(f'y_train shape  is {y_train.shape}')
print(f'y_test shape  is {y_test.shape}')



# =============================================================================
# Save Data
# =============================================================================
import pickle

pickle_out = open("X_train.pickle","wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("y_train.pickle","wb")
pickle.dump(y_train, pickle_out)

pickle_out = open("X_test.pickle","wb")
pickle.dump(X_train, pickle_out)
pickle_out.close()

pickle_out = open("y_test.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()