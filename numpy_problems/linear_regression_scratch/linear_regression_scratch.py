import numpy as np 
import pandas as pd 

TEST_SIZE=0.2
#Boston Housing Dataset load
df = pd.read_csv("BostonHousing.csv")
total_records = df.shape[0]

test_records_size = int(TEST_SIZE * total_records)
test_record_indices = np.random.choice(np.arange(total_records), test_records_size)
test_data = df.iloc[test_record_indices]
train_data = df.drop(index=test_record_indices)

def optimize():
    pass

def gradient():
    pass

def loss_function():
    pass

    
