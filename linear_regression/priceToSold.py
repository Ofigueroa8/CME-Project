import pandas as pd
import numpy as np
import pprint
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("linear_regression/datesFfrUnitsPrices.csv")
x = data[["unitsStarted (Thousands)","fedfunds"]]
y = data["lumberPrice"]

pprint.pprint(data)
data["date"] = pd.to_datetime(data["date"])


# Create your training and testing datasets:
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = .2)


# Create the model
model = LinearRegression().fit(xtrain, ytrain)

# Find the coefficient, bias, and r squared values. 
# Each should be a float and rounded to two decimal places. 
coef = np.round(model.coef_, 2)
intercept = round(float(model.intercept_), 2)
r_squared = round(model.score(x, y),2)

# Print out the linear equation and r squared value:
print(f"Model's Linear Equation: y= {coef[0]}x1 + {coef[1]}x2 + {intercept}")
print("R Squared value:", r_squared)

'''
**********TEST THE MODEL**********
'''

# get the predicted y values for the xtest values - returns an array of the results
predict = model.predict(xtest)
# round the value in the np array to 2 decimal places
predict = np.around(predict, 2)
print(predict)

# Test the model by looping through all of the values in the xtest dataset
print("\nTesting Linear Model with Testing Data:")


# print(type(ytest))
# print(ytest)
# print(type(xtest))
# print(xtest)
print(float(coef[0]))


for i in range(len(xtest)):
    actual = ytest.index[i] # gets the actual y value from the ytest dataset
    predicted_y = predict[i] # gets the predicted y value from the predict variable
    x_coord = xtest.loc[xtest.index[i]] # gets the x value from the xtest dataset
    lis = x_coord.tolist() # a new variable making x_coord a list
    print(f"Units Started: {lis[0]} Fedfunds Rate: {lis[1]} Actual: {actual} Predicted: {predicted_y}")

while True:
    print("Type '-1' to stop")
    unit = float(input("How many units will be started? "))
    if unit == -1:
        break
    ffr = float(input("What is the selected Fed Funds Rate? "))
    if ffr == -1:
        break
    total = (coef[0] * unit) + (coef[1] * ffr) + intercept
    print(f"The predicted price of lumber for {unit} housing units and a Fed Funds Rate of {ffr} will be {total}")

# '''
# **********CREATE A VISUAL OF THE RESULTS**********
# '''
# #sets the size of the graph
# plt.figure(figsize=(5,4))

# #creates a scatter plot and labels the axes
# plt.scatter(xtrain,ytrain, c="purple", label="Training Data")
# plt.scatter(xtest, ytest, c="blue", label="Testing Data")

# plt.scatter(xtest, predict, c="red", label="Predictions")

# plt.xlabel("Lumber")
# plt.ylabel("Houses Sold")
# plt.title("Houses Sold Based on Lumber Prices")
# plt.plot(x, coef*x + intercept, c="r", label="Line of Best Fit")

# plt.legend()
# plt.show()
