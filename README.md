# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.
 
## Program:
#### Program to implement the linear regression using gradient descent.
```
# Developed by: THANJIYAPPAN.K
# RegisterNumber: 212222240108 
# Import required package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("ex1.txt",header=None)
data
data.shape
plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City(10,000s)")
plt.ylabel("Profit ($10,000)")
plt.title("Profit Vs Prediction")
def computeCost(X,y,theta):
    m=len(y)
    h=X.dot(theta)
    square_err=(h-y)**2

    return 1/(2*m) * np.sum(square_err)
data_n=data.values
m=data_n[:,0].size
X=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))

computeCost(X,y,theta)
theta.shape
y.shape
X.shape
def gradientDescent(X,y,theta,alpha,num_iters):
  
  m=len(y)
  J_history=[]

  for i in range(num_iters):
    predictions=X.dot(theta)
    error=np.dot(X.transpose(),(predictions - y))
    descent=alpha * 1/m * error
    theta-=descent
    J_history.append(computeCost(X,y,theta))

  return theta, J_history
  
theta,J_history = gradientDescent(X,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")
plt.plot(J_history)
plt.xlabel("Iternations")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")
plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color="r")
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Polpulation of City (10,000s)")
plt.ylabel("Profit (10,000s)")
plt.title("Profit Prediction")
def predict(x,theta):
  predictions= np.dot(theta.transpose(),x)
  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*10000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*10000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))
```

## Output:
### Read the CSV file:
![1](https://github.com/22009011/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343461/8b63d86e-71f3-4e65-8b10-61b05d4448a8)

### Dataset Shape:
![2](https://github.com/22009011/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343461/3e825cbb-3846-4a87-aa8e-54c811a2e60f)

### Profit Vs Prediction Graph:
![3](https://github.com/22009011/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343461/ff869c39-fcaf-4b0b-acf9-f12c27c56072)

### Compute Cost x,y,theta value:
![4](https://github.com/22009011/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343461/da7e6988-b3b9-4491-b6ae-31bfa84fe744)

### x,y,theta Shape:
![5](https://github.com/22009011/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343461/6d7d379b-6133-4aa5-82f0-68252f26f735)

### GradientDescent:
![6](https://github.com/22009011/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343461/d397e5c7-300f-47ba-b2e0-73384b158e75)

### Cost function using Gradient Descent Graph:
![7](https://github.com/22009011/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343461/aa5fd9a2-4202-40b1-a5ed-04b294a17b72)

### Profit Prediction Graph:
![8](https://github.com/22009011/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343461/e0424ef3-b987-459f-982b-c1242bbaf31d)

### Profit Prediction:
![9](https://github.com/22009011/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343461/f573941f-7759-4063-8717-2ae9cd6839c1)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
