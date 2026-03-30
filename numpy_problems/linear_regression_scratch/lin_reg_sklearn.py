import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

#From Chatgpt
# Load data
df = pd.read_csv("BostonHousing.csv")
X = df.drop(columns="medv").values
y = df["medv"].values

# Optional: standardize features (like we did manually)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test split
TEST_SIZE = 0.2
total_records = X.shape[0]
test_records_size = int(TEST_SIZE * total_records)
test_indices = np.random.choice(np.arange(total_records), test_records_size, replace=False)
train_indices = np.setdiff1d(np.arange(total_records), test_indices)

X_train, y_train = X_scaled[train_indices], y[train_indices]
X_test, y_test = X_scaled[test_indices], y[test_indices]

# Fit sklearn Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Coefficients and intercept
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Compute MSE
train_loss = mean_squared_error(y_train, y_train_pred)
test_loss = mean_squared_error(y_test, y_test_pred)
print(f"Train MSE: {train_loss:.4f}")
print(f"Test MSE: {test_loss:.4f}")

# Optional: plot predictions vs true values
import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Sklearn Linear Regression Predictions")
plt.grid(True)
plt.savefig("sklearn_lin_reg.jpg",dpi=300)
plt.show()