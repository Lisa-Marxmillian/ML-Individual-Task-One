# Set a smaller learning rate
from matplotlib import pyplot as plt
from Mean_Squared_Error import gradient_descent, Y, X, epochs

learning_rate = 0.00001

# Re-train the model using the smaller learning rate
m, c = gradient_descent(X, Y, learning_rate, epochs)

# Plot the line of best fit
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, m * X + c, color='red', label='Best fit line')
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price')
plt.title('Office Size vs. Price')
plt.legend()
plt.show()

# Predict the office price for 100 sq. ft.
predicted_price_100 = m * 100 + c
predicted_price_100
