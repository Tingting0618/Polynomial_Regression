# Polynomial Regression

A simple way to use a linear model to fit nonlinear data is to add powers of each feature as new features, then train a linear model on this extended set of features.

#### Content Includes:
- Polynomial Regression: Add powers of each feature as new features
- Learning Curves: Plots of the model’s performance on the training set and the validation set as a function of the training set size (or the training iteration).

### Polynomial Regression Procedures:

#### 1. Simulate some random data

```Python
import numpy as np
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X**2 + X + 2 + np.random.randn(m, 1)
```

![download](https://user-images.githubusercontent.com/44503223/127773657-a8c3c48c-8052-4fa3-ada8-c419cdc576a0.png)


#### 2. Perform Polynomial Features transformation using Scikit-Learn

Clearly, a straight line will never fit this data properly. So let’s use Scikit-Learn’s Polynomial Features class to transform our training data, adding the square (seconddegree polynomial) of each feature in the training set as a new feature (in this case there is just one feature)

```Python
from sklearn.preprocessing import PolynomialFeatures
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)
print(X[0])
print(X_poly[0])
```

#### 3. Perform Linear Regression using Scikit-Learn

```Python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
lin_reg.intercept_, lin_reg.coef_
```

#### 4. Visualize the results

![download](https://user-images.githubusercontent.com/44503223/127773696-312ec265-04e3-4fe7-89d5-f6f853c2ea68.png)

### Learning Curves:

**Learning Curves** are used to anwser the question: How can you tell that your model is overfitting or underfitting the data?

#### 1. Linear Regression Learning Curve

- Adding new instances to the training set doesn’t make the average error much better. A straight line cannot do a good job modeling the data.
- The data is under-fitted. 

![download](https://user-images.githubusercontent.com/44503223/127774022-8397b42d-2c4e-4d98-9993-c8663bb947d0.png)

#### 2. Polynomial Regression (degree=2) Learning Curve

- The error on the training data is much lower than with the Linear Regression model, which indicates model fitting improvement. 

![download](https://user-images.githubusercontent.com/44503223/127774071-d1c5af6c-c1bc-4454-a5c0-2da6925aea86.png)

## Learn More

For more information, please check out the [Project Portfolio](https://tingting0618.github.io).

## Reference

This repo is my learning journal following:
- Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition, by Aurélien Géron (O’Reilly). Copyright 2019 Kiwisoft S.A.S., 978-1-492-03264-9.
