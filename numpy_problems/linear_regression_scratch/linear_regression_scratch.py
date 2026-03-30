import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Hyperparameters
TEST_SIZE = 0.2
LEARNING_RATE = 1e-3
BATCH_SIZE = 10
EPOCHS = 100

# load and normalize data, normalising important, reduces exploding gradients
df = pd.read_csv("BostonHousing.csv")
df_features = df.drop(columns="medv")
df_target = df["medv"]

X_mean = df_features.mean()
X_std = df_features.std()
df_features_normalized = (df_features - X_mean) / X_std

df = df_features_normalized.copy()
df["medv"] = df_target

# Train/Test split
total_records = df.shape[0]
test_records_size = int(TEST_SIZE * total_records)
test_record_indices = np.random.choice(
    np.arange(total_records), test_records_size, replace=False
)

test_data = df.iloc[test_record_indices]
train_data = df.drop(index=test_record_indices).sample(frac=1)  # shuffle

# Separate features and targets
X_train = train_data.drop(columns="medv").values
y_train = train_data["medv"].values
X_test = test_data.drop(columns="medv").values
y_test = test_data["medv"].values

# Init Random Parameters
coef = np.random.sample(X_train.shape[1])
bias = np.random.sample(1)


# Helper Functions
def predict(X, coef, bias):
    return np.dot(X, coef) + bias


def loss_function(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)


def optimize(coef, bias, pred, X, y, lr=LEARNING_RATE):
    error = pred - y
    grad_coef = (2 / len(pred)) * (X.T @ error)
    grad_bias = (2 / len(pred)) * np.sum(error)
    coef = coef - lr * grad_coef
    bias = bias - lr * grad_bias
    return coef, bias


def data_batch(X, y, batch_size=BATCH_SIZE):
    dataset = []
    for start in range(0, X.shape[0], batch_size):
        end = start + batch_size
        dataset.append((X[start:end], y[start:end]))
    return dataset

#Train start
train_dataset = data_batch(X_train, y_train, BATCH_SIZE)
epoch_loss = []
for epoch in range(EPOCHS):
    loss_log = []
    print(f"\n--- Epoch {epoch+1} ---")
    for batch_idx, (x_batch, y_batch) in enumerate(train_dataset):
        y_pred = predict(x_batch, coef, bias)
        loss = loss_function(y_pred, y_batch)
        loss_log.append(loss)

        print(f"Batch {batch_idx+1} - Loss: {loss:.4f}")
        coef, bias = optimize(coef, bias, y_pred, x_batch, y_batch)
    epoch_loss.append(np.array(loss_log).mean())

#loss plot
plt.figure(figsize=(8, 5))
plt.plot(epoch_loss, marker="o")
plt.title("Training Loss per Epoch")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.savefig("MSE_loss_plot.jpg", dpi=300)
plt.show()

#test Prediction
y_test_pred = predict(X_test, coef, bias)
test_loss = loss_function(y_test_pred, y_test)
print(f"\nTest MSE Loss: {test_loss:.4f}")

plt.figure(figsize=(8,5))
plt.scatter(y_test, y_test_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.title("Sklearn Linear Regression Predictions")
plt.grid(True)
plt.savefig("scratch_lin_reg.jpg",dpi=300)
plt.show()

#Test MSE recorded in my last iteration 20.3541