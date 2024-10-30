import numpy as np
from Mean_Squared_Error import mean_squared_error

def gradient_descent(X, Y, learning_rate, epochs):
    m, c = np.random.rand(), np.random.rand()
    n = len(X)

    for epoch in range(epochs):
        Y_pred = m * X + c
        mse = mean_squared_error(Y, Y_pred)

        dm = (-2 / n) * sum(X * (Y - Y_pred))
        dc = (-2 / n) * sum(Y - Y_pred)

        m -= learning_rate * dm
        c -= learning_rate * dc

        print(f"Epoch {epoch + 1}, MSE: {mse:.4f}")

    return m, c