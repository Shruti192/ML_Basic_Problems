# -*- coding: utf-8 -*-
"""
Problem Statement:-

You have to implement Logistic Regression from scratch on the breast cancer dataset in order to classify the type 
of cancer as Malignant (M) or Benign (B).
Read the dataset, carry out the necessary  **feature scaling**  following the same formula as Linear Regression. Define 
the  **sigmoid function**,  the  **cost function**  and implement  **Gradient Descent**  to learn the weights. The function   
  **Logistic Regression**   should return the weights which you are going to use to predict the class.  *Choose M as class label 1 
and B as class label 0 **.
Now, if the logistic regression weights give a probability greater than or equal to 0.5, mark it 
as label 1, otherwise label 0. 
Split the dataset into train and test set and report the accuracy on both training set and test set.

"""
############ Best Output Condition found for Split Ratio = 0.8 ###########################
####  learning rate = 0.8  and  Threshold = 8 x 10^-9   ###################################


############################### Importing required Libraries ##################################
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

################################## Loading breast_cancer.csv file from PC ######################################
df = pd.read_csv (r'C:\Users\shrut\Desktop\breast_cancer_dataset.csv')
df.drop(df.columns[len(df.columns) - 1], axis = 1, inplace = True)
print ('\n Here is our orignal data-table for breast cancer:- \n\n',df)
print('\n\n\n Name of all',len(df.columns),' columns/ features in orignal data-set are as following :- \n\n', df.columns)

x = df.copy()
x.drop(x.columns[1], axis = 1, inplace = True)     #deleting the target coloumn 'diagonosis' from feature matrix

y = np.array( df['diagnosis'])
y = y.reshape((len(y), 1))      # Final Target Matrix

############## Add one dummy feature vector of ones into x for incorporating bias ################
# Generating 
Xo = np.array(x)     
n,m = Xo.shape                     
x0 = np.ones((n,1))                                                     
X = np.hstack((x0, Xo))     # Final feature matrix 

print('\n \n shape of orignal dataset :  ', df.shape)
print('\n shape of orignal feature matrix :  ', X.shape)

##########  Function for spliting boston datasets into training set and testing set  ###############
################# This Function returns x_train, y_train, x_test, y_test ###########################

def split_dataset(X, y, split_perc):
     x_train = X[ :int(split_perc*len(x))]
     y_train = y[ :int(split_perc*len(y))]

     x_test = X[int(split_perc*len(x)) : ]
     y_test = y[int(split_perc*len(y)) : ]
     
     return x_train, y_train, x_test, y_test
 
sr = float( input("Enter split ratio i.e (no. of trainee set / no. of testing set) ratio: "))   # Asking user input for Split Ratio
x_train, y_train, x_test, y_test = split_dataset(X, y, sr)                                      # Calling of split function

################## Printing the shapes of x, y, x_train, x_test, y_train, y_test ###################
print('\n shape of orignal target matrix :   ', y.shape)
print('\n shape of orignal trainee feature matrix :  ', x_train.shape)
print('\n shape of orignal trainee target matrix :  ', y_train.shape)
print('\n shape of orignal testing feature matrix :  ', x_test.shape)
print('\n shape of orignal testing target matrix :  ', y_test.shape)


############### Function used to scale the features, used to improve performance ####################
# It will convert all our features in the range of [-1,1] 
# Using formula feature_scaled = ((feature - feature_mean) / (data_max - data_min))
# It Returns scaled x_train and x_test 

def feature_scaling(x_train, x_test):
    
    i = len(x_train)
    while i < len(X):     #while loops to scale x_test  i =[404, 506) and j = [0,14)
        j = 1
        while j < len(x_train[0]):
            x_test[i-len(x_train),j] = ( ((x_test[i- len(x_train),j]) - (np.mean(x_train[:,j])))/((max(x_train[:,j])) - (min(x_train[:,j]))) )
            j = j+1
        i = i+1
        
    i = 0
    while i < len(x_train):       #while loops to scale x_train  i = [506, 404) and j = [0,14)       
        j = 1
        while j < len(x_train[0]):
            x_train[i,j] = ( ((x_train[i,j]) - (np.mean(x_train[:,j])))/((max(x_train[:,j])) - (min(x_train[:,j]))) )
            j = j+1           
        i = i+1    
        
    return x_train, x_test

x_train, x_test = feature_scaling( x_train, x_test)


############### Function used to lable target matrix as 1 or 0 based on it's value is 'M' or 'B' ####################
# here Y can be y or y_test or y_train
# it returns the Labelled target matrix of corresponding unlabelled target matrix.
  
def labelling(y):
    i = 0
    Y = np.zeros([len(y), 1])
    
    while i < len(y):
        if y[i] == 'M':
            Y[i] = 1
        elif y[i] == 'B':
            Y[i] = 0
        
        i = i+1   
    return Y

YL = labelling(y)
y_trainL = labelling(y_train)
y_testL = labelling(y_test)


######################## Function which gives out predicted Target ##############################
# Using hypothesis Function for Logistic Regression
# X may be X or x_test or x_train or any other suitable feature matrix
# Returs the coressponding y_pred

def hypothesisFun(theta, X):
    i = 0
    t = np.dot(X, theta)
    H = np.zeros([len(X),1])
    while i < len(X):
        H[i] = 1 / (1 + (math.exp(-t[i])))
        i = i+1   
    return H


############### Function to Compute Loss or Cost Function for Logistic Regression problems #################
# Here too Y may be y, y_test or y_train depending upon y_pred from prediction function
# Returns the value of loss or CostFunction or Error for Logistic Regression according to corressponding  Y_pred and Y_actual
def Loss(H, YL):
    t = 0
    C = np.zeros([len(YL),1])
    i = 0
    while i < len(YL):
        C[i] = -(((YL[i])*(math.log(H[i]))) + ((1 - YL[i])*(math.log(1 - H[i]))))
        t = t + C[i]
        i = i+1
    k = float((t / (2*len(YL))))
    return k
        
####### User input for learning Rate ####################
# Learning rate is a tuing parameter 
#which determines the step size at each iteration while moving toward a minimum of a loss function
alpha = float(input('Enter learning rate here: '))

########### Function to minimize the theta for finding Global minima of loss function ##############
# CostFunction of LogisticRegression by Gradient Descent method
# Returns the theta that minimizes the loss, Current_loss, CostFunction list and Iteration list

def LogisticRegression(x_train, y_trainL, alpha):
    
     theta = np.zeros([len(x_train[0]),1])       # Initalizing theta at origin
     y_pred = hypothesisFun(theta, x_train)      # Calculating y_pred according to updated theta  
     prev_loss = Loss(y_pred, y_trainL)          # Calculating the previous loss i.e loss before current upcoming iteration
     current_loss = 1000                         # initializing current loss i.e loss after the current iteration                
     count = 0                                   # initializing the count of iteration
     CostFunction = []                           # Initializing list to store current loss after eact iterations
     Iteration = []                              # Initializing list to store number of iterations corresponding to current loss
     
     # Asking User Input for threshold i.e maximum possible value of abs( prev_loss - current_loss)
     threshold = float(input('Enter the threshold i.e maximum possible value of abs( prev_loss - current_loss) here: '))
     
     # Applying Gradient descent through while loop to find theta at which loss minimizes
     # Executing while loop until convergence stops or approximately stops
     # we are enusring this by applying a threshold to the absolute difference between current loss and previous loss
     
     while not (abs( prev_loss - current_loss) <= threshold):
          prev_loss = current_loss                                                    # Updating prev_loss in each iteration
          D = np.dot((x_train).transpose(), ((np.dot(x_train, theta)) - (y_trainL)))  #Calculating D
          theta = theta  - (((alpha)/(len(x_train)))*(D))                             # Updating theta
          y_pred = hypothesisFun(theta, x_train)                                      # Calculating y_pred_trainee to calculate curret loss based on updated theta
          current_loss = Loss(y_pred, y_trainL)                                       # Updating Current loss
          count = count + 1                                                           # Updating Count of iteration
          CostFunction.append(current_loss)                                           # appending/ adding updated current_loss into the list of CostFunction after each iteration
          Iteration.append(count)                                                     # appending/adding updated count number into the list of iteration after each iteration
     print('\n \n No. of Iterations/ steps it took to minimize the loss :',count)     # Priting number of iterations/ steps it took to minimize Costfunction
     print('\n Difference between previousloss and currentloss : ',abs( prev_loss - current_loss) ) # Printing the final absolute value  Threshold
     
     return theta, current_loss, CostFunction, Iteration
    
theta, current_loss, CostFunction, Iteration = LogisticRegression(x_train, y_trainL, alpha)   # Calling of Logistic Regression


y_pred_train = hypothesisFun(theta, x_train)  # Predicting target value for trainee set according to found theta which minimizes the CostFunction for Linear Regression
y_pred_test = hypothesisFun(theta, x_test)    # Predicting target value for testing set according to found theta which minimizes the CostFunction for Linear Regression

final_loss = Loss(y_pred_test, y_testL)    
print('\n \n Theta matrix that minimizes the loss :- \n',theta)
print('\n \n Value of loss with training set = ',current_loss)
print('\n \n Value of loss with test set = ',current_loss)

############# Function which converts predicted target into matrix with binary output either 1 or 0 ##############

def RoundOff(Y):
    i = 0
    R = np.zeros([len(Y), 1])
    while i < len(Y):
        if Y[i] >= 0.5:
            R[i] = 1
        elif Y[i] < 0.5:
            R[i] = 0
        i = i+1
    return R

    
y_pred_testR = RoundOff(y_pred_test)
y_pred_trainR = RoundOff(y_pred_train)

############# Function which converts predicted and Rounded target into matrix with with predicted Cancer Type ##############
def Cancertype(y_predR):
    i = 0
    T = []
    while i < len(y_predR):
        if y_predR[i] == 1:
            T.append('M')
        elif y_predR[i] == 0:
            T.append('B')
        i = i+1
    return T

y_pred_testCT = Cancertype(y_pred_testR)
print('\n \n Pridected Cancer Type for y_test is :- \n',y_pred_testCT)
            
################ Printing the accuracy Score for testing and trainee set #########################
from sklearn.metrics import accuracy_score

accuracy_test = accuracy_score(y_pred_testR, y_testL)
accuracy_train = accuracy_score(y_pred_trainR, y_trainL)
print('\n \n Accuracy Score for Test sets = ',accuracy_test,'\n Accuracy Score for Trainee sets ', accuracy_train)
print('Accuracy is the proportion of correct predictions over total predictions.')
print('It lies between 0 to 1 and as much as high your accuracy is your prediction was that eually good')

#################### Plot the loss with respect to number of iterations ########################
plt.plot( Iteration , CostFunction )
plt.ylabel(' Current_Loss / CostFunction')
plt.xlabel(' Iteration Numbers')
plt.title(' Checking that loss decreases with each step/ iteration')
plt.show()     