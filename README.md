**Linear Regression Model for Office Price Prediction**
This project implements a simple Linear Regression model using Python to predict office prices based on their size. The model uses the Mean Squared Error (MSE) as a performance measure and Gradient Descent as the learning algorithm.

**Project Structure**
The project consists of the following files:

- **`main.py`**: The main script that loads the dataset, trains the Linear Regression model, and makes predictions.
- **`Mean_Squared_Error.py`**: Contains the function to calculate Mean Squared Error (MSE).
- **`Gradient_Descent.py`**: Implements the Gradient Descent algorithm for training the model.

**Dataset**
The dataset used for this project is named **"Nairobi Office Price Ex.csv"**. It contains two columns relevant to the task:
- **`SIZE`**: The size of the office in square feet.
- **`PRICE`**: The price of the office.

**Prerequisites**
Ensure that you have Python installed, along with the necessary libraries:
- `numpy`
- `pandas`
- `matplotlib`

To install the required libraries, run:
```bash
pip install numpy pandas matplotlib
```

**Running the Project**
1. **Clone the repository** or **download the files** into your working directory.
2. **Prepare the dataset**: Make sure the dataset file **"Nairobi Office Price Ex.csv"** is placed in the same directory as your project files.
3. **Run the main script**:
   ```bash
   python main.py
   ```

**Expected Output**
- The model trains over 10 epochs, displaying the Mean Squared Error (MSE) in each epoch.
- A plot showing the line of best fit.
- The predicted price for an office of size **100 sq. ft.**.

**Project Files Description**
- **`Mean_Squared_Error.py`**:
  ```python
  import numpy as np

  def mean_squared_error(y_true, y_pred):
      return np.mean((y_true - y_pred) ** 2)
  ```

- **`Gradient_Descent.py`**:
  ```python
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
  ```

- **`main.py`**:
  ```python
  import pandas as pd
  import matplotlib.pyplot as plt
  from gradient_descent import gradient_descent

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
  ```

**License**
This project is open-source and available for educational purposes.

---

Feel free to modify this **README.md** to better fit your project or if you add more features!
