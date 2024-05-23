import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data, standardise

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# TODO: calculate closed-form solution
theta_best = [0, 0]

x_train_saved = x_train
x_test_saved = x_test

x_train = np.c_[np.ones((x_train.shape[0], 1)), x_train]
x_test = np.c_[np.ones((x_test.shape[0], 1)), x_test]

theta_best = np.linalg.inv(x_train.T @ x_train) @ x_train.T @ y_train

# TODO: calculate error

y_pred_train = x_train @ theta_best
y_pred_test = x_test @ theta_best

mse_train = np.mean((y_train - y_pred_train)**2)
mse_test = np.mean((y_test - y_pred_test)**2)

x_train = x_train_saved
x_test = x_test_saved

print("mse after closed-form solution: ")
print("train mse: ", mse_train)
print("test mse: ", mse_test)

# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization

u = np.mean(x_train)
sigma = np.std(x_train)
x_train_standardised = (x_train - u)/sigma
x_test_standardised = (x_test - u)/sigma

X_b = np.c_[np.ones((x_train_standardised.shape[0], 1)), x_train_standardised]
X_b_test = np.c_[np.ones((x_test_standardised.shape[0], 1)), x_test_standardised]

# TODO: calculate theta using Batch Gradient Descent
learning_rate = 0.001
n_iterations = 1000
m = X_b.shape[0]

np.random.seed(42)
theta = np.random.randn(2, 1)

for iteration in range(n_iterations):
    gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y_train.reshape(-1, 1))
    theta -= learning_rate * gradients

# TODO: calculate error

y_pred_train_std = X_b.dot(theta)
y_pred_test_std = X_b_test.dot(theta)

mse_train_std = np.mean((y_train - y_pred_train_std.flatten())**2)
mse_test_std = np.mean((y_test - y_pred_test_std.flatten())**2)
print("mse after gradient descent:")
print("train mse: ", mse_train_std)
print("test mse: ", mse_test_std)

# plot the regression line
x = np.linspace(min(x_test_standardised), max(x_test_standardised), 100)
y = float(theta[0]) + float(theta[1]) * x
plt.plot(x, y)
plt.scatter(x_test_standardised, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()