from sklearn.datasets import load_breast_cancer
import numpy as np
import pandas as pd
path_1=r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RETURN_AIR_TEMP_EXPLORATION\data_results\train.csv"
_data_1=pd.read_csv(path_1)

path_2=r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RETURN_AIR_TEMP_EXPLORATION\data_results\validate_x__.csv"
_data_2=pd.read_csv(path_2)

frames = [_data_1, _data_2]

result = pd.concat(frames)

path_final=r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RETURN_AIR_TEMP_EXPLORATION\data_results\final_raw_data.csv"
result.to_csv(path_final)
print(1)