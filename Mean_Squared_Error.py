import numpy as np
import matplotlib.pyplot as plt
from main import data

# Extracting the relevant columns for x (feature) and y (target)
X = data['SIZE'].values
Y = data['PRICE'].values

# Define the Mean Squared Error (MSE) function
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Define the Gradient Descent function
def gradient_descent(X, Y, learning_rate, epochs):
    m, c = np.random.rand(), np.random.rand()  # Initializing slope (m) and y-intercept (c) randomly
    n = len(X)

    for epoch in range(epochs):
        # Make predictions using the current values of m and c
        Y_pred = m * X + c

        # Calculate the MSE for the current epoch
        mse = mean_squared_error(Y, Y_pred)

        # Calculate gradients for m and c
        dm = (-2 / n) * sum(X * (Y - Y_pred))
        dc = (-2 / n) * sum(Y - Y_pred)

        # Update m and c using the gradients
        m -= learning_rate * dm
        c -= learning_rate * dc

        # Print the error at each epoch
        print(f"Epoch {epoch + 1}, MSE: {mse:.4f}")

    return m, c


# Set hyperparameters
learning_rate = 0.001
epochs = 10

# Train the model using Gradient Descent
m, c = gradient_descent(X, Y, learning_rate, epochs)

# Plot the line of best fit
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, m * X + c, color='red', label='Best fit line')
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price')
plt.title('Office Size vs. Price')
plt.legend()
plt.show()

# Predict the office price for 100 sq. ft. using the final model
predicted_price_100 = m * 100 + c
predicted_price_100
