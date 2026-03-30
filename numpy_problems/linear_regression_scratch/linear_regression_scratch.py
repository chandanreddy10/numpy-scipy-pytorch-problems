import numpy as np 
import pandas as pd 

TEST_SIZE=0.2
LEARNING_RATE=0.01
BATCH_SIZE=1
EPOCHS=1
#Boston Housing Dataset load
df = pd.read_csv("BostonHousing.csv")
total_records = df.shape[0]

test_records_size = int(TEST_SIZE * total_records)
test_record_indices = np.random.choice(np.arange(total_records), test_records_size)
test_data = df.iloc[test_record_indices]
train_data = df.drop(index=test_record_indices)
train_data = train_data.sample(frac=1)
y_test = test_data.iloc[:,-1]
X_test = test_data.drop(columns="medv")
y_train = train_data.iloc[:, -1]
X_train = train_data.drop(columns="medv")

coef = np.random.sample(X_train.shape[1])
bias = np.random.sample(1)

def optimize():
    pass

def gradient():
    pass

def loss_function():
    pass

def data_batch(X_train, y_train, BATCH_SIZE=BATCH_SIZE):
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    dataset = []
    for index in range(0,X_train.shape[0],BATCH_SIZE):
        X_temp = X_train[index:index+BATCH_SIZE].flatten()
        y_temp = y_train[index:index+BATCH_SIZE]
        dataset.append((X_temp, y_temp))
    return dataset

train_dataset = data_batch(X_train, y_train)
print(train_dataset)
for epoch in range(EPOCHS):
    for x, y in train_dataset:
        coef_data_product = coef @ x.T
        result = coef_data_product + bias 
        print(result-y)
        break

