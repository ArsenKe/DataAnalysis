# Solution for task 1 (Lasso) of lab assignment - FDA SS23 by [keshishyan_a_lasso.py]

# if necessary, write text as answer in comments or use a Jupyter notebook

# imports
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# load data (change path if necessary)
with open('C:/Users/HP/OneDrive/Desktop/FDAAssignment/fda_lab_ss23/lasso_data.csv','r') as f:
    lines = f.readlines()

header = lines[0].strip().split(',')	
data = []
for line in lines:
        row = line.strip().split(',')
        data.append(row)

df = pd.DataFrame(data[1:], columns=data[0])

# Task 1.1 Is it possible to solve the lasso optimisation problem analytically? Explain. (3 points)
"""
No, it is not possible to solve the Lasso optimization problem analytically. 
The goal of Lasso is to minimize the sum of the squared errors between the predicted values and the actual values,
subject to a constraint on the absolute value of the sum of the regression coefficients.
The Lasso optimization problem involves a penalty term that introduces non-differentiability in the objective function at zero.
This makes it impossible to solve the optimization problem analytically. The Lasso problem is typically solved numerically using algorithms such as coordinate descent.
"""
# Task 1.2 Split the data into a train and a test set with appropriate test size. (2 points)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
#20% of the data will be used for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Task 1.3 Fit a linear regression model for Y using all remaining variables on the training data. (5 points)

# Create a linear regression model 
lin_reg = LinearRegression()

# Fit the model to the training data
lin_reg.fit(X_train, y_train)


# Task 1.4 Make a model prediction on unseen data and assess model performance using a suitable metric. (5 points)
y_pred = lin_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Linear Regression Mean Squared Error:", mse)

# Task 1.5 Perform lasso regression using the same data as in task 1.3 (6 points)

#alpha is a hyperparameter of the Lasso regression model
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)

# Task 1.6 Compare model performance to the original linear model by using the same metric and test set as in 1.4.
# What do you observe? (2 points)
y_pred_lasso = lasso_reg.predict(X_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
print("Lasso Regression Mean Squared Error:", mse_lasso)
#We can observe that the performance of the lasso model is similar to the linear regression model.


# Task 1.7 Print out the model coefficients for both, the linear model and the lasso model. (2 points)
print("Linear Regression Coefficients:", lin_reg.coef_)
print("Lasso Regression Coefficients:", lasso_reg.coef_)

# Task 1.8 What do you observe comparing the estimated model coefficients? Was this result expected? (5 points)
# Hint: Look at the data generating process and lasso explanation to answer this question


"""
         The coefficients of estimated model tells that he Lasso regression model has some coefficients that are exactly equal to zero, 
    while the linear regression model has non-zero coefficients for all variables.It seems that Lasso is a useful method for feature selection as it can identify
    the most important variables and eliminate the irrelevant ones.
        Comparing the result of MSE for both Lasso  and Linear Regression, I can observe that adjusting the hyperparameter alpha to 0.1 ,the result has a lower mean squared error (MSE). 
    This indicates that 0.1 alpha of results may have better performance than the higher number of alpha.

"""

