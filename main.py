from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import Predictors
path=r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RETURN_AIR_TEMP_EXPLORATION\data_results\final_raw_data.csv"
initial_data=pd.read_csv(path)
work_or_not=initial_data['Day_Of_Week']
working_days_data= [] #pd.DataFrame()
if len(work_or_not)== len(initial_data):
    for x in range(len(work_or_not)):
        a=initial_data.iloc[x]
        b=work_or_not[x]
        if(work_or_not[x]!=5) and (work_or_not[x]!=6):
              working_days_data.append (a)

working_days_data_np=np.array(working_days_data)
target=pd.Series(working_days_data_np[:,-1]).astype(float)
raw_inputs=working_days_data_np[:,0:-1]
working_days_data_df=pd.DataFrame(working_days_data)
raw_inputs_df = working_days_data_df.iloc[:,0:-1].copy()
########Corr analysis:
# number_of_inputs=raw_inputs.shape[1]
# corr_=[]
# for x in range(number_of_inputs):
#    test_sampe=pd.Series(raw_inputs[:,x]).astype(float)
#    r = target.corr(test_sampe)
#    name=initial_data.columns[x+1]
#    corr_.append([name, r])
# corr_=pd.DataFrame(corr_)
# output_1=r"C:\Users\wan397\OneDrive - CSIRO\Desktop\return air temp\corr.csv"
# corr_.to_csv(output_1)

######selected inputs
slected_inputs_df =raw_inputs_df[['Supply_Fan.Speed_Sensor|AHU.09','Supply_Air_Temperature_Sensor|AHU.09','Outside_Air_Temperature_Sensor|Zone.L3E','Return_Air_Temperature_Sensor|Zone.L3E:-0']]

slected_inputs_df_2 =raw_inputs_df[['Supply_Fan.Speed_Sensor|AHU.09','Supply_Air_Temperature_Sensor|AHU.09','Outside_Air_Temperature_Sensor|Zone.L3E']]

X_train, X_test, y_train, y_test = train_test_split(slected_inputs_df_2, target, random_state=0, train_size = .5)

# training_accuracy, test_accuracy =Predictors.lstm_att(X_train, y_train, X_test, y_test)
# print('lstm+att')
# print(training_accuracy)
# print(test_accuracy)
# training_accuracy, test_accuracy =Predictors.lstm(X_train, y_train, X_test, y_test)
# print('lstm')
# print(training_accuracy)
# print(test_accuracy)
# training_accuracy, test_accuracy =Predictors.rf_(X_train, y_train, X_test, y_test)
# print('rf')
# print(training_accuracy)
# print(test_accuracy)
# training_accuracy, test_accuracy =Predictors.svr_(X_train, y_train, X_test, y_test)
# print('svr')
# print(training_accuracy)
# print(test_accuracy)
# training_accuracy, test_accuracy =Predictors.xgboost_(X_train, y_train, X_test, y_test)
# print('xgboost')
# print(training_accuracy)
# print(test_accuracy)
# training_accuracy, test_accuracy =Predictors.dt_(X_train, y_train, X_test, y_test)
# print('dt')
# print(training_accuracy)
# print(test_accuracy)
training_accuracy, test_accuracy =Predictors.linear(X_train, y_train, X_test, y_test)
print('linear')
print(training_accuracy)
print(test_accuracy)