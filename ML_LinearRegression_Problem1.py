'''
Problem Description:

    Implementation Linear Regression with Gradient Descent algorithm from scratch using numpy on the Boston Housing Dataset.
    
    To load the Boston Housing Dataset, follow the following lines.
        from sklearn.datasets import load_boston
        boston_dataset = load_boston()
        boston = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)

    The above code segment will give you a Pandas dataframe of the Boston Housing Dataset. 
    Visualization of the data as needed. 
    
    The next step is to preprocess the data. Use mean scaling normalisation of the features.
    
    The formula is as follows    
        𝑥 = (𝑚𝑒𝑎𝑛 𝑜𝑓 𝑥 −𝑥) / (max 𝑥 − min 𝑥)

    The next step is to split the dataset into two parts.
    One will be used for training and the other will be used as testing. 
    Splitting the dataset as 80% for training and 20% for testing.

    Define a function Linear Regression which takes in the training set and returns the values of the weights.
    Finally, predict the target variable using the weights obtained from the Linear Regression function. 
    Compute the sum of squared error on the test dataset and display the R2 score.

    Note: You can also plot the loss after each iteration of the gradient descent algorithm using matplotlib

'''

############################### Importing required Libraries ##################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

################################## Loading boston dataset ######################################

from sklearn.datasets import load_boston
boston_dataset = load_boston()
boston = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)

############## Add one dummy feature vector of ones into X for incorporating bias ################

xo = np.array(boston)                # feature matrix from boston data set
n,m = xo.shape                       # shape of feature matrix == (no. of datasets X no. of features)
x0 = np.ones((n,1))                  # generating one column vector i.e our dummy feature vector 
                                     # with all its element as 1 to add in the orignal feature vector
x = np.hstack((x0, xo))              # adding dummy feature vector into the orignal one at 0
y = np.array(boston_dataset.target)  # target matrix i.e our target is to pridict it i.e it's matrix for house pricing
y = y.reshape((len(y), 1))           # reshaping of y matrix and fixing it's no. of columns

##########  Function for spliting boston datasets into training set and testing set  ###############
################# This Function returns x_train, y_train, x_test, y_test ###########################

def split_dataset(x, y, split_perc = 0.8):
     x_train = x[ :int(split_perc*len(x))]
     y_train = y[ :int(split_perc*len(y))]

     x_test = x[int(split_perc*len(x)) : ]
     y_test = y[int(split_perc*len(y)) : ]
     
     return x_train, y_train, x_test, y_test

x_train, y_train, x_test, y_test = split_dataset(x, y, 0.8)   # Calling of split function

################## Printing the shapes of x, y, x_train, x_test, y_train, y_test ###################

print('\n shape of orignal feature matrix :  ', x.shape)
print('\n shape of orignal target matrix :   ', y.shape)
print('\n shape of orignal trainee feature matrix :  ', x_train.shape)
print('\n shape of orignal trainee target matrix :  ', y_train.shape)
print('\n shape of orignal testing feature matrix :  ', x_test.shape)
print('\n shape of orignal testing target matrix :  ', y_test.shape)

############### Function used to scale the features, used to improve performance ####################
# It will convert all our features in the range of [-1,1] 
# Using formula feature_scaled = ((feature - feature_mean) / (data_max - data_min))
# It Returns scaled x_train and y_train 

def feature_scaling(x_train, x_test):
    
    i = len(x_train)
    while i < len(x):           # While loop to scale x_test i = [404, 506) and j = [0,14)
        j = 1
        while j < 14:
            x_test[i-404,j] = ( ((x_test[i-404,j]) - (np.mean(x_train[:,j])))/((max(x_train[:,j])) - (min(x_train[:,j]))) )
            j = j+1
        i = i+1
    
    i = 0
    while i < len(x_train):     # While loops to scale x_train  i = [0, 404) and j = [0,14)
        
        j = 1
        while j < 14:
            mean = np.zeros([14,1]) 
            mean[j] = np.mean(x_train[:,j])
            x_train[i,j] = ( ((x_train[i,j]) - (np.mean(x_train[:,j])))/((max(x_train[:,j])) - (min(x_train[:,j]))) )
            j = j+1
            
        i = i+1
    
    return x_train, x_test

x_train, x_test = feature_scaling( x_train, x_test)   # Calling of feature_scaling Function

######################## Function_to _predict_the_house_pricing ##############################
# Using hypothesis Function for Linear Regression
# X may be x or x_test or x_train or any new set of features
# Returs the coressponding y_pred 

def prediction( X , theta):  
    y_pred = np.dot( X, theta)
    return y_pred

############### Function to Compute Loss or Cost Function or Squared Error Sum #################
# Here too Y may be y, y_test or y_train depending upon y_pred from prediction function
# Returns the value of loss or CostFunction for linear Regeression or Mean Squared Error according to corressponding y_pred
    
def Loss(y_pred, Y):    
    i = 0
    t = 0    
    while i < len(Y):
        t = t + ((y_pred[i][0] - Y[i][0])**2)
        i = i+1
        
    l = t / (2*len(Y))
    return l

########### Function to minimize the theta for finding Global minima of loss function ##############
# CostFunction of Linear Regression by Gradient Descent method
# Returns the theta that minimizes the loss, Current_loss, CostFunction list and Iteration list

def LinearRegression(x_train, y_train, alpha):
    
     theta = np.zeros([len(x_train[0]),1])  # Initalizing theta at origin
     y_pred = np.dot( x_train, theta)       # Calculating y_pred according to updated theta
     prev_loss = Loss(y_pred, y_test)       # Calculating the previous loss i.e loss before current upcoming iteration
     current_loss = 1000                    # initializing current loss i.e loss after the current iteration
     count = 0                              # initializing the count of iteration
     CostFunction = []                      # Initializing list to store current loss after eact iterations
     Iteration = []                         # Initializing list to store number of iterations corresponding to current loss
     
        # Applying Gradient descent through while loop to find theta at which loss minimizes
     # Executing while loog until convergence stops or approximately stops
     # we are enusring this by applying a threshold to the absolute difference between current loss and previous loss
     
     threshold = 0.001
     
     while not (abs( prev_loss - current_loss) <= threshold):
          prev_loss = current_loss                                                                  # Updating prev_loss in each iteration
          D = np.dot((x_train).transpose(), ((np.dot(x_train, theta)) - (y_train)))                 # Calculating D
          theta = theta  - (((alpha)/(len(x_train)))*(D))                                           # Updating theta
          y_pred = np.dot(x_train, theta)                                                           # Calculating y_pred_trainee to calculate curret loss based on updated theta
          current_loss = Loss(y_pred, y_train)                                                      # Updating Current loss
          count = count + 1                                                                         # Updating Count of iteration
          CostFunction.append(current_loss)                                                         # appending/ adding updated current_loss into the list of CostFunction after each iteration
          Iteration.append(count)                                                                   # appending/adding updated count number into the list of iteration after each iteration
     print('\n \n No. of Iterations/steps it took to minimize the loss : ', count)                   # Priting number of iterations/ steps it took to minimize Costfunction
     print('\n Difference between previous loss and current loss : ', abs( prev_loss - current_loss) ) # Printing the final absolute value  Threshold
     
     return theta, current_loss, CostFunction, Iteration
        
theta, current_loss, CostFunction, Iteration = LinearRegression( x_train, y_train, 0.009)

print('\n \n Theta matrix that minimizes the loss :- \n',theta)
print('\n \n Value of loss with training set = ',current_loss)
y_pred = np.dot(x_test, theta)
final_loss = Loss(y_pred, y_test)
print('\n Value of loss with testing set = ',final_loss)

from sklearn.metrics import r2_score

print('\n \n r2_score : ',r2_score(y_test, y_pred))
print('\n Coefficient of determination also called as R2 score is used to evaluate the performance of a linear regression model\n')
print(''' i) The best possible score is 1 which is obtained when the predicted values are the same as the actual values. \n
 ii) R2 score of baseline model is 0. \n
iii) During the worse cases, R2 score can even be negative.''')

#################### Plot the loss with respect to number of iterations ########################

print('\n \n')
print('The plot of the value of loss function of training set with respect to the number of iterations is shown below.')
print('\n')
plt.plot( Iteration , CostFunction )
plt.ylabel(' Current_Loss / CostFunction')
plt.xlabel(' Iteration Numbers')
plt.title(' Ensuring that the loss decreases with each step/ iteration ')
plt.show()