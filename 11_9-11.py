# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 15:39:06 2021

@author: bhash
"""

import numpy as np
import matplotlib.pyplot as plt
import os

work_path = r'C:\Users\bhash\OneDrive\Documents\CivilLab'
os.chdir(work_path)

#%%

# number of rows describes number of operations
# base is a concept used to define your space
# y = a + bx + cx^2 is the same as 1 x x^2 ...

# y = f(x) = f([1,x,x^2,x^3,...,x^n])

# matrices
# over determinate - just one model - sufficient data
# under determinate - infinite number of models/solutions - insufficient data


#%% read in data from file and perform linear regression

file = r'C:\Users\bhash\OneDrive\Documents\CivilLab\NGES_data.csv'
data = np.genfromtxt(file, dtype=float, delimiter = ",", skip_header=1)

x = data[:,0]
y = data[:,1]

A = np.vstack([np.ones(len(x)), x, x**2, x**3, x**4, x**5, x**6]).T
# first column all ones
# the higher the order, the less the error (order being x**n <-):
    # Beware overfitting - go too high in order, the model will not follow 
    #\ trend lines and become too flexible - experiment with this by adding
    #\ higher and higher orders until it doesn't fit
    # Behavior beyond observed range will be unstable (high error) if high order

coefficients = np.linalg.lstsq(A, y, rcond=None)[0]

import matplotlib.pyplot as plt
plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, A @ coefficients[:,np.newaxis], 'r', label='Fitted line')
# np.newaxis - transforms a row vector into a vertical vector - horizontal matrix to vertical
plt.xlabel('Depth [m]')
plt.ylabel('Qt [-]')
plt.legend()
plt.show()

#%%

def my_product(v1,v2):
    z = v1 * v2
    """
    arguments:
        v1:float:1st input
        v2:float:2nd input
    return:
        z:float np.array:result
    """
    return z
# define - making our own function when there is not a good existing one
# () to say what aspects are attributed to function
# ":" means all the following lines are attached to function
# "z" is a temporary variable only defined inside the function - "z" that is defined later is not attributed to this "z"
# return - output of function
# (""") commenting is necessary to define what is included in the function

x = np.arange(5)
# vector of 0,1,2,3,4 - for arange(5)
y = np.arange(5) + 1

z = my_product(x,y)
# for function - v1 input is "x" and v2 input is "y"

#%% combining previous two segments

def linear_regression(x,y,order):
    """
    arguments:
        x:float np.array:x vector
        y:float np.array: y vector
        order:int:highest order
    return:
        coefficients: float np.array: the coefficients of all of the terms
    """
    # Parameters
    # ----------
    # x : TYPE
    #     DESCRIPTION.
    # y : TYPE
    #     DESCRIPTION.
    # order : TYPE
    #     DESCRIPTION.

    # Returns
    # -------
    # None.
    A = np.ones(len(x))
    for i in np.arange(order) + 1:
        A = np.vstack([A,x**i])
    # "A" vector defined two blocks above:
    # A = np.vstack([np.ones(len(x)), x, x**2, x**3, x**4, x**5]).T
    # "+ 1" because lowest order must be "1" not "0"
    # first row all 1's
    # x to the "i-th" order
    # after each iteration of this for-loop, a new row is added
    A = A.T
    coefficients = np.linalg.lstsq(A, y, rcond=None)[0]
    return coefficients, A
# can use coding for other functions in personally defined function like use of np.arange()

file = r'C:\Users\bhash\OneDrive\Documents\CivilLab\NGES_data.csv'
data = np.genfromtxt(file, dtype=float, delimiter = ",", skip_header=1)

x = data[:,0]
y = data[:,1]

coefficients, A = linear_regression(x,y,order=11)

plt.plot(x, y, 'o', label='Original data', markersize=10)
plt.plot(x, A @ coefficients[:,np.newaxis], 'r', label='Fitted line')
plt.xlabel('Depth [m]')
plt.ylabel('Qt [-]')
plt.legend()
plt.show()
# this overall program makes the linear regression much less error-prone
# order appears to become too high around 10

#%%

import my_module
# my_module - saved .py file
print(my_module.CONST)
# print the CONST variable - all letters must be capital
# the "." after module - tells system to find what follows in the specified file

x = np.arange(5)
y = np.arange(5) + 1

z = my_module.my_product(x,y)
# link created by using "import" and now can bridge my_module and my_product
























