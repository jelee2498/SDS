#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from typing import Union, Tuple
import itertools
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
from sklearn.metrics import confusion_matrix


# ### Assignment 0: Path Configuration
# Configure ```data_path``` properly to load the dataset.
# 

# In[ ]:


#------------------------------------------WRITE YOUR CODE------------------------------------------#
data_path = Path.cwd()
#------------------------------------------END OF YOUR CODE-----------------------------------------#
if not data_path.exists():
    raise FileNotFoundError(f'Invalid data path {data_path}')


# ### Extract the dataset

# In[ ]:


if not Path('/dataset').exists():
    import zipfile
    with zipfile.ZipFile(data_path / 'dataset.zip', 'r') as f:
        f.extractall('/dataset')
    print('Successfully extracted dataset.zip')

data_path = Path('/dataset')


# ### Split the dataset into training and test set

# In[ ]:


training_metadata = pd.read_csv(data_path / 'train_metadata.csv', index_col=0, encoding='utf-8', engine='python', dtype={'path': str, 'directory': int, 'sex': int, 'width': int, 'height': int})
test_metadata = pd.read_csv(data_path / 'test_metadata.csv', index_col=0, encoding='utf-8', engine='python', dtype={'path': str, 'directory': int, 'sex': int, 'width': int, 'height': int})


# ### Preprocess the input images

# ### Assignment 1: Tune HOG descriptor parameters
# You should find optimal parameters to maximize the performance of the classifier.

# In[ ]:


#--------------------------------------WRITE YOUR CODE--------------------------------------#
hog = cv2.HOGDescriptor((64,64), (16,16), (16,16), (8,8), 8)
# (winSize=(64, 64), blockSize=(16, 16), blockStride=(16, 16), cellSize=(8, 8), nBins=8)
# hog.winSize = (64, 64)
# hog.blockSize = (16, 16)
# hog.blockStride = (16, 16)
# hog.cellSize = (8, 8)
# hog.nBins = 8
#--------------------------------------END OF YOUR CODE-------------------------------------#


# ### Assignment 2: Load an image
# You should read the image as grayscale and resize it to $64\times64$. Use cv2.imread() and cv2.resize().

# In[ ]:


def load_image(name: Union[Path, str]) -> np.ndarray:
    filename = str(data_path / name)
    #--------------------------------------WRITE YOUR CODE--------------------------------------#
    img = 
    #--------------------------------------END OF YOUR CODE-------------------------------------#
    return img


# ### Assignment 3: Extract features from the image
# Use the HOG extractor ```hog``` to extract features from the given image. You need to flatten the features so that it returns one-dimensional array with the shape of (512,).

# In[ ]:


def extract_features(img: np.ndarray, hog: cv2.HOGDescriptor) -> np.ndarray:
    #--------------------------------------WRITE YOUR CODE--------------------------------------#
    features = 
    #--------------------------------------END OF YOUR CODE-------------------------------------#
    return features


# In[ ]:


def preprocess_images(metadata: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    n_total = metadata.shape[0]
    inputs = np.empty((n_total, 512), dtype=float)
    outputs = np.empty((n_total,), dtype=int)
    with tqdm(desc='Processing', total=n_total) as pbar:
        for i, (name, row) in enumerate(metadata.iterrows()):
            inputs[i, :] = extract_features(load_image(name), hog)
            outputs[i] = row['sex']
            pbar.update(1)
    return inputs, outputs


# In[ ]:


train_x, train_y = preprocess_images(training_metadata)


# In[ ]:


test_x, test_y = preprocess_images(test_metadata)


# ### Assignment 4: Train the classifier
# You should train the ```classifier``` (e.g. SVM, decision tree, ...). To implement the classifier, use SVC(), RandomForestClassifier(), and so on.

# In[ ]:


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#------------------------------------------WRITE YOUR CODE------------------------------------------#
classifier = 
#------------------------------------------END OF YOUR CODE-----------------------------------------#
classifier.fit(train_x, train_y)


# ### Result: Visualization of Confusion Matrix

# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    accuracy = np.sum(cm * np.eye(cm.shape[0])) / np.sum(cm)
    print(f'Accuracy: {accuracy}')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


# Training Result

# In[ ]:


pred_y = classifier.predict(train_x)
plot_confusion_matrix(confusion_matrix(train_y, pred_y), classes=['Female', 'Male'])


# Test Result

# In[ ]:


pred_y = classifier.predict(test_x)
plot_confusion_matrix(confusion_matrix(test_y, pred_y), classes=['Female', 'Male'])

