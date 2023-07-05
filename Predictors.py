import numpy as np
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from numpy import random
from datetime import datetime
from attention import Attention
from keras import Input
import keras.models
from sklearn.ensemble import RandomForestRegressor
from keras.models import load_model
import pickle
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

dynamic_test=0
para=100
# save the iris classification model as a pickle file
model_pkl_file = r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RETURN_AIR_TEMP_EXPLORATION\data_results\xinlin_model_lstm.pkl"

def lstm_att(X_train, y_train, X_test, y_test):

    trainX = np.array(X_train)
    trainY = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    look_forward=1
    loss_ = 'mean_squared_error'

    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    n_samples = trainX.shape[0]
    time_step = trainX.shape[1]

    model_input = Input(batch_input_shape=(None, time_step, 1))
    x = LSTM(time_step*para, return_sequences=True)(model_input)
    x = Attention(units=time_step*para)(x)
    x = Dense(100)(x)
    x = Dense(10)(x)
    x = Dense(look_forward)(x)
    #x=Dropout(0.4)(x)
    model = keras.models.Model(model_input, x)
    model.compile(loss=loss_, optimizer='adam')
    # print(model.summary())
    model.fit(trainX, trainY, epochs=para, batch_size=n_samples, verbose=0)

    pred_trainY = model.predict(trainX)
    r_2_training = r2_score(pred_trainY, trainY)

    pred_testY = model.predict(X_test)
    r_2_testing = r2_score(pred_testY, y_test)
    if(dynamic_test==1):
        path=r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RETURN_AIR_TEMP_EXPLORATION\data_results\lstm_dynamic.csv"
        f_one_step = open(path, "w+")
        for x in range(100):
            # [['Supply_Fan.Speed_Sensor|AHU.09', 'Supply_Air_Temperature_Sensor|AHU.09','Outside_Air_Temperature_Sensor|Zone.L3E', 'Return_Air_Temperature_Sensor|Zone.L3E:-0']]
            Supply_Fan = random.randint(100)
            Supply_Air_Temperature = random.randint(15, 35)
            Outside_Air_Temperature = random.randint(10, 35)
            Return_Air_Temperature = random.randint(10, 35)
            X_test = np.reshape(
                np.array([Supply_Fan, Supply_Air_Temperature, Outside_Air_Temperature, Return_Air_Temperature]), (1, time_step, 1))
            X_test = np.reshape(
                np.array([Supply_Fan, Supply_Air_Temperature, Outside_Air_Temperature]),
                (1, time_step, 1))

            result = model.predict(X_test)
            str_ = str(Supply_Fan) + ',' + str(Supply_Air_Temperature) + ',' + str(Outside_Air_Temperature) + ',' + str(
                Return_Air_Temperature) + ','+str(result[0])+'\n'
            f_one_step = open(path, "a+")
            f_one_step.write(str_.replace('[', '').replace(']', ''))
            f_one_step.close()
        # print(result)


    # filename = 'lstm_model.sav'
    # pickle.dump(model, open(filename, 'wb'))
    #
    # dynamic_test_r_2= dynamic_test(filename)

    return (r_2_training,r_2_testing )

def lstm (X_train, y_train, X_test, y_test):

        trainX = np.array(X_train)
        trainY = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        look_forward = 1
        loss_ = 'mean_squared_error'

        trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        n_samples = trainX.shape[0]
        time_step = trainX.shape[1]

        model = Sequential()
        model.add(LSTM(time_step * para, batch_input_shape=(None, time_step, 1)))
        model.add(Dense(time_step * para))
        model.add(Dense(100))
        model.add(Dense(100))
        model.add(Dense(10))
        model.add(Dense(look_forward))
        model.compile(loss=loss_, optimizer='adam')
        model.fit(trainX, trainY, epochs=para, batch_size=n_samples, verbose=0)

        pred_trainY = model.predict(trainX)
        r_2_training = r2_score(pred_trainY, trainY)

        pred_testY = model.predict(X_test)
        r_2_testing = r2_score(pred_testY, y_test)


        model_file = r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RETURN_AIR_TEMP_EXPLORATION\data_results\xinlin_model_lstm.h5"
        model.save(model_file)


        if (dynamic_test == 1):
            path = r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RETURN_AIR_TEMP_EXPLORATION\data_results\lstm_dynamic.csv"
            f_one_step = open(path, "w+")
            for x in range(100):
                # [['Supply_Fan.Speed_Sensor|AHU.09', 'Supply_Air_Temperature_Sensor|AHU.09','Outside_Air_Temperature_Sensor|Zone.L3E', 'Return_Air_Temperature_Sensor|Zone.L3E:-0']]
                Supply_Fan = random.randint(100)
                Supply_Air_Temperature = random.randint(15, 35)
                Outside_Air_Temperature = random.randint(10, 35)
                # Return_Air_Temperature = random.randint(10, 35)
                # X_test = np.reshape(
                #     np.array([Supply_Fan, Supply_Air_Temperature, Outside_Air_Temperature, Return_Air_Temperature]),
                #     (1, 4, 1))
                X_test = np.reshape(
                    np.array([Supply_Fan, Supply_Air_Temperature, Outside_Air_Temperature]),
                    (1, 3, 1))
                result = model.predict(X_test)
                str_ = str(Supply_Fan) + ',' + str(Supply_Air_Temperature) + ',' + str(Outside_Air_Temperature)  + ',' + str(result[0]) + '\n'
                f_one_step = open(path, "a+")
                f_one_step.write(str_.replace('[', '').replace(']', ''))
                f_one_step.close()
                # print(result)

        # filename = 'lstm_model.sav'
        # pickle.dump(model, open(filename, 'wb'))
        #
        # dynamic_test_r_2= dynamic_test(filename)

        return (r_2_training, r_2_testing)

def rf_ (X_train, y_train, X_test, y_test):
    trainX = np.array(X_train)
    trainY = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    model = RandomForestRegressor(n_estimators=500)
    model.fit(trainX, trainY)

    pred_trainY = model.predict(trainX)
    r_2_training = r2_score(pred_trainY, trainY)

    pred_testY = model.predict(X_test)
    r_2_testing = r2_score(pred_testY, y_test)

    with open(model_pkl_file, 'wb') as file:
        pickle.dump(model, file)
    if (dynamic_test == 1):
        path = r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RETURN_AIR_TEMP_EXPLORATION\data_results\rf_.csv"
        f_one_step = open(path, "w+")
        for x in range(100):
            # [['Supply_Fan.Speed_Sensor|AHU.09', 'Supply_Air_Temperature_Sensor|AHU.09','Outside_Air_Temperature_Sensor|Zone.L3E', 'Return_Air_Temperature_Sensor|Zone.L3E:-0']]
            Supply_Fan = random.randint(100)
            Supply_Air_Temperature = random.randint(15, 35)
            Outside_Air_Temperature = random.randint(10, 35)
            Return_Air_Temperature = random.randint(10, 35)
            X_test = np.reshape(
                np.array([Supply_Fan, Supply_Air_Temperature, Outside_Air_Temperature, Return_Air_Temperature]),
                (1, 4))
            result = model.predict(X_test)
            str_ = str(Supply_Fan) + ',' + str(Supply_Air_Temperature) + ',' + str(Outside_Air_Temperature) + ',' + str(
                Return_Air_Temperature) + ',' + str(result[0]) + '\n'
            f_one_step = open(path, "a+")
            f_one_step.write(str_.replace('[', '').replace(']', ''))
            f_one_step.close()
            # print(result)

    # filename = 'lstm_model.sav'
    # pickle.dump(model, open(filename, 'wb'))
    #
    # dynamic_test_r_2= dynamic_test(filename)

    return (r_2_training, r_2_testing)

def svr_ (X_train, y_train, X_test, y_test):
    from sklearn.svm import SVR
    trainX = np.array(X_train)
    trainY = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    model = SVR (C=1)
    model.fit(trainX, trainY)

    pred_trainY = model.predict(trainX)
    r_2_training = r2_score(pred_trainY, trainY)

    pred_testY = model.predict(X_test)
    r_2_testing = r2_score(pred_testY, y_test)

    if (dynamic_test == 1):
        path = r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RETURN_AIR_TEMP_EXPLORATION\data_results\svr_.csv"
        f_one_step = open(path, "w+")
        for x in range(100):
            # [['Supply_Fan.Speed_Sensor|AHU.09', 'Supply_Air_Temperature_Sensor|AHU.09','Outside_Air_Temperature_Sensor|Zone.L3E', 'Return_Air_Temperature_Sensor|Zone.L3E:-0']]
            Supply_Fan = random.randint(100)
            Supply_Air_Temperature = random.randint(15, 35)
            Outside_Air_Temperature = random.randint(10, 35)
            Return_Air_Temperature = random.randint(10, 35)
            if(x==0):
                f_one_step.write('Supply_Fan, Supply_Air_Temperature, Outside_Air_Temperature, Return_Air_Temperature, Predicted \n')
            X_test = np.reshape(
                np.array([Supply_Fan, Supply_Air_Temperature, Outside_Air_Temperature, Return_Air_Temperature]),
                (1, 4))
            result = model.predict(X_test)
            str_ = str(Supply_Fan) + ',' + str(Supply_Air_Temperature) + ',' + str(Outside_Air_Temperature) + ',' + str(
                Return_Air_Temperature) + ',' + str(result[0]) + '\n'
            f_one_step = open(path, "a+")
            f_one_step.write(str_.replace('[', '').replace(']', ''))
            f_one_step.close()
            # print(result)


    return (r_2_training, r_2_testing)

def xgboost_ (X_train, y_train, X_test, y_test):

    trainX = np.array(X_train)
    trainY = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    model = XGBRegressor(max_depth=100)
    model.fit(trainX, trainY)

    pred_trainY = model.predict(trainX)
    r_2_training = r2_score(pred_trainY, trainY)

    pred_testY = model.predict(X_test)
    r_2_testing = r2_score(pred_testY, y_test)

    if (dynamic_test == 1):
        path = r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RETURN_AIR_TEMP_EXPLORATION\data_results\XGBRegressor_.csv"
        f_one_step = open(path, "w+")
        for x in range(100):
            # [['Supply_Fan.Speed_Sensor|AHU.09', 'Supply_Air_Temperature_Sensor|AHU.09','Outside_Air_Temperature_Sensor|Zone.L3E', 'Return_Air_Temperature_Sensor|Zone.L3E:-0']]
            Supply_Fan = random.randint(100)
            Supply_Air_Temperature = random.randint(15, 35)
            Outside_Air_Temperature = random.randint(10, 35)
            Return_Air_Temperature = random.randint(10, 35)
            if(x==0):
                f_one_step.write('Supply_Fan, Supply_Air_Temperature, Outside_Air_Temperature, Return_Air_Temperature, Predicted \n')
            X_test = np.reshape(
                np.array([Supply_Fan, Supply_Air_Temperature, Outside_Air_Temperature, Return_Air_Temperature]),
                (1, 4))
            result = model.predict(X_test)
            str_ = str(Supply_Fan) + ',' + str(Supply_Air_Temperature) + ',' + str(Outside_Air_Temperature) + ',' + str(
                Return_Air_Temperature) + ',' + str(result[0]) + '\n'
            f_one_step = open(path, "a+")
            f_one_step.write(str_.replace('[', '').replace(']', ''))
            f_one_step.close()
            # print(result)


    return (r_2_training, r_2_testing)

def dt_ (X_train, y_train, X_test, y_test):

    trainX = np.array(X_train)
    trainY = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    model = DecisionTreeRegressor(max_depth=100)
    model.fit(trainX, trainY)

    pred_trainY = model.predict(trainX)
    r_2_training = r2_score(pred_trainY, trainY)

    pred_testY = model.predict(X_test)
    r_2_testing = r2_score(pred_testY, y_test)

    if (dynamic_test == 1):
        path = r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RETURN_AIR_TEMP_EXPLORATION\data_results\DecisionTreeRegressor_.csv"
        f_one_step = open(path, "w+")
        for x in range(100):
            # [['Supply_Fan.Speed_Sensor|AHU.09', 'Supply_Air_Temperature_Sensor|AHU.09','Outside_Air_Temperature_Sensor|Zone.L3E', 'Return_Air_Temperature_Sensor|Zone.L3E:-0']]
            Supply_Fan = random.randint(100)
            Supply_Air_Temperature =  random.randint(15, 35)
            Outside_Air_Temperature = random.randint(10, 35)
            Return_Air_Temperature = random.randint(10, 35)
            if(x==0):
                f_one_step.write('Supply_Fan, Supply_Air_Temperature, Outside_Air_Temperature, Return_Air_Temperature, Predicted \n')
            X_test = np.reshape(
                np.array([Supply_Fan, Supply_Air_Temperature, Outside_Air_Temperature, Return_Air_Temperature]),
                (1, 4))
            result = model.predict(X_test)
            str_ = str(Supply_Fan) + ',' + str(Supply_Air_Temperature) + ',' + str(Outside_Air_Temperature) + ',' + str(
                Return_Air_Temperature) + ',' + str(result[0]) + '\n'
            f_one_step = open(path, "a+")
            f_one_step.write(str_.replace('[', '').replace(']', ''))
            f_one_step.close()
        # print(result)
    return (r_2_training, r_2_testing)

def linear (X_train, y_train, X_test, y_test):
    from sklearn.linear_model import LinearRegression
    trainX = np.array(X_train)
    trainY = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    model = LinearRegression().fit(trainX, trainY)
    pred_trainY = model.predict(trainX)
    r_2_training = r2_score(pred_trainY, trainY)

    pred_testY = model.predict(X_test)
    r_2_testing = r2_score(pred_testY, y_test)

    # filename = 'lstm_model.sav'
    #
    model_pkl_linear= r"C:\Users\wan397\OneDrive - CSIRO\Desktop\RETURN_AIR_TEMP_EXPLORATION\data_results\xinlin_model_linear.pkl"
    pickle.dump(model, open(model_pkl_linear, 'wb'))
    return(r_2_training, r_2_testing)




# def dynamic_test(filename):
#
#     loaded_model = pickle.load(open(filename, 'rb'))
#     for x in range(100):
#         #[['Supply_Fan.Speed_Sensor|AHU.09', 'Supply_Air_Temperature_Sensor|AHU.09','Outside_Air_Temperature_Sensor|Zone.L3E', 'Return_Air_Temperature_Sensor|Zone.L3E:-0']]
#         Supply_Fan=random.randint(100)
#         Supply_Air_Temperature=random.randint(15, 35)
#         Outside_Air_Temperature=random.randint(10, 35)
#         Return_Air_Temperature=random.randint(10, 35)
#         X_test=np.reshape(np.array([Supply_Fan,Supply_Air_Temperature,Outside_Air_Temperature,Return_Air_Temperature]),(1,4,1))
#         result = loaded_model.predict(X_test)
#         str_=str(Supply_Fan)+','+str(Supply_Air_Temperature)+','+str(Outside_Air_Temperature)+','+str(Return_Air_Temperature)+','
#         print(result)
#         pass
