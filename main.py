import pandas as pd
import matplotlib.pyplot as plt
from Gradient_Descent import gradient_descent

# Load the dataset
data = pd.read_csv('Nairobi Office Price Ex.csv')
X = data['SIZE'].values
Y = data['PRICE'].values

# Train the model
learning_rate = 0.00001
epochs = 10
m, c = gradient_descent(X, Y, learning_rate, epochs)

# Plot the results
plt.scatter(X, Y, color='blue', label='Data points')
plt.plot(X, m * X + c, color='red', label='Best fit line')
plt.xlabel('Office Size (sq. ft.)')
plt.ylabel('Office Price')
plt.title('Office Size vs. Price')
plt.legend()
plt.show()

# Make a prediction for 100 sq. ft.
predicted_price_100 = m * 100 + c
print(f"Predicted price for 100 sq. ft.: {predicted_price_100:.2f}")