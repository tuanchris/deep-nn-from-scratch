import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import h5py
from lr_utils import load_dataset
from logistics_regression import model

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

m_train = train_set_y.shape[1]
m_test = test_set_y.shape[1]
num_px = train_set_x_orig.shape[1]

# Reshape the training and test examples

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

# Standardize dataset
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.

d = model(train_set_x, train_set_y, test_set_x, test_set_y,
    num_iterations = 2000, learning_rate = 0.005, print_cost = True)

'''
    Comment: Training accuracy is close to 100%. This is a good sanity check: our
    model is working and has high enough capacity to fit the training data.
    Test accuracy is 70%. It is actually not bad for this simple model, given the
    small dataset we used and that logistic regression is a linear classifier.

    We can alos see that the model is clearly overfitting the training data.
    In later section, I will implement several methods to reduce overfitting, 
    for example using regularization.
'''
