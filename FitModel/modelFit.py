
# coding: utf-8
''' 
	Author : Anurag Kumar
'''
# import required libraries 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression # using sklearn ML library import LinearRegression 


# read data into dataframes
df_train = pd.read_csv('data.csv')
df_test = pd.read_csv('test.csv')


# To work on categorical data , we need to represent the data numerically
# 1. categorical column with two categories can be represented using binary values
mapping = {'black' : 0 , 'white' : 1}
# create a new series called h_color for test and train dataframes
df_train['h_color'] = df_train.h.map(mapping)
df_test['h_color'] = df_test.h.map(mapping)

'''
2. categorical column with unordered categories cannot be represented using normal integers
      since doing that implies ordered categories , so we need to make multiple dummy variables
      to represent them.
'''
# Creating dummy variables using column c ('c_blue','c_green','c_red','c_yellow') separately
# we remove one dummy variable since three of them implicitly defines the fourth one and captures all info in 'c' 
c_dummies_train = pd.get_dummies(df_train.c, prefix='c').iloc[:, 1:] 
c_dummies_test = pd.get_dummies(df_test.c, prefix='c').iloc[:, 1:]

# concatenate the dummy variables to the dataframes to work on them
df_train = pd.concat([df_train, c_dummies_train], axis=1)
df_test = pd.concat([df_test, c_dummies_test], axis=1)



# new features to work on after pre-processing the data 
feature_columns = ['a', 'b','c_green','c_red','c_yellow', 'd', 'e', 'f', 'g','h_color']
target_column = 'y'

# take the training data from df_train and test data from the df_test
X_train = df_train[feature_columns]
Y_train = df_train[target_column]
X_test = df_test[feature_columns]



# linear Regression model is a good fit for the data since all the features and the target variable are continuous
# After some research on the data (such as k-cross evaluation , correlation and visualization) , I came up with this model
# instantiate and fit the linearRegression Model
lm = LinearRegression()
lm.fit(X_train,Y_train)

# predict the test data using the model in the training data
predictions = lm.predict(X_test)


# make a new column in the dataFrame i.e the target variable
df_test['y'] = pd.Series(predictions, index=df_test.index)


# rename the index column already given in the file to 'i'
df_test.columns.values[0] = 'i'

#df2.rename(columns={'Unnamed: 0': 'i'}, inplace=True)
# take the index column and the prediction column and write it to csv file
df_test[['i','y']].to_csv('predicted.csv', encoding='utf-8', index=False)





