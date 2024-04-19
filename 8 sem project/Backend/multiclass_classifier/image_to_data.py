

import numpy as np
import cv2
import glob


X_data = []  # Empty list to be append for input data
y_labels = [] # Empty list to be append for labelling the input data

files = glob.glob ("D:/ml-test/nature/*.jpg") # image you want to classify
files2 = glob.glob ("D:/ml-test/desert/*.jpg") # rest of images

for myFile in files: # This is the image you want to classify set
    
    image = cv2.imread (myFile)
    X_data.append (image)  #Getting all the CKT images
    y_labels.append(1) #  1 for CKT sets

for file2 in files2: # This is the all the other images that are not your image.
    
    image2 = cv2.imread(file2)
    X_data.append(image2) #Getting all non CKT images
    y_labels.append(0)  # 0 for non CKT sets
    
# Now X_Data is a list of image data from both sets

X_sets =  np.array(X_data)  #Changing the List into a NumPy array

m  = X_sets.shape[0] # Number of samples
 
features_array = X_sets.reshape(X_sets.shape[0], -1).T 
 # The rows represent the features, column represent the amount of samples
 

y_labels = np.array(y_labels) #making it into a NumPy Array

y =  y_labels.reshape(1,-1) # Remember that your output must be size (1,m)
x  = features_array/255.0 #Normalise


    
# Randomly shuffling the datasets (Just in case it was set up in an orderly fashioned):
# which in our case it is set up into two organised datasets so shuffling is a MUST
 
permutation = list(np.random.permutation(m))
shuffled_X = x[:, permutation]
shuffled_Y = y[:, permutation] 

# Now split X and Y to training and test sets:

# ~Percent % for training, ~Percent % for testing . (I am thinking it is too small to include dev sets)

percent = 0.85  # Tune this to how much percent of training and test samples do you want.

train_x =  shuffled_X[:,0:int(percent*m)]
train_y =  shuffled_Y[:,0:int(percent*m)]

test_x  = shuffled_X[:,int(percent*m):]
test_y  = shuffled_Y[:,int(percent*m):]

print('Dataset is ready')






 