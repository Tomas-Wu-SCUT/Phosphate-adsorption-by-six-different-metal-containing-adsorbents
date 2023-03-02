# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 22:25:38 2023

@author: Administrator
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 20:30:58 2022

@author: Thomas
"""


from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor as XGBR
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np
import math


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False





df = pd.read_excel('Processed 1511 datapoints in scenario I&II.xlsx')
a = df.iloc[:,0:16]
b = pd.concat([a, df.iloc[:,12]], axis = 1)
#c = pd.concat([b, df.iloc[:,19]], axis = 1)
Data = pd.concat([b, df.iloc[:,13]], axis = 1)

for i in range(len(Data)-1, 1, -1):
    print(i)
    for j in range(0, 6):
        if (pd.isna(Data.iloc[i, j])):
            Data.iloc[i, j] = 0.0
        else:
            Data.iloc[i, j] = 1.0
    
    if (Data.iloc[i, 6] > 100):
        Data = Data.drop(i)
        continue
    
    Data.iloc[i, 11] = 1.0
    for j in range(6, Data.columns.size):
        if ((pd.isna(Data.iloc[i, j])) or (not is_number(Data.iloc[i, j]))):
            Data = Data.drop(i)
            break

Input = Data.iloc[2:,0:12]
Output = Data.iloc[2:,12]

nsj1 = len(Input)
nTest = 60

out_c = []
out_e = []
out_x = []
y_real = []
d_c = []
d_e = []
d_x = []
for i in range(0, nTest):
    A1 = Input.iloc[nsj1- nTest +i, :]
    City_block = []
    for j in range(0, nsj1- nTest):
        B1 = Input.iloc[j, :]
        temp = abs((A1 - B1)).sum() 
        City_block.append(temp)
    
    City_block_stId = sorted(range(len(City_block)), key = lambda k : City_block[k], reverse = True)
    
    x_train = []
    nTrain = 20
    Input_Cos = Input.iloc[City_block_stId[0:nTrain], :]
    Output_Cos = Output.iloc[City_block_stId[0:nTrain]]
    x_train = np.array(Input_Cos, dtype=np.float)
    y_train = np.array(Output_Cos, dtype=np.float).reshape(nTrain, 1)
    
    x_test = np.array(A1, dtype=np.float).reshape(1, 12)
    y_test = np.array(Output.iloc[nsj1- nTest +i], dtype=np.float).reshape(1, 1)
    
    clf = DecisionTreeRegressor(random_state=0) 
    rfc = RandomForestRegressor(random_state=0)
    clf = clf.fit(x_train,y_train)          #决策数
    rfc = rfc.fit(x_train,y_train)          #随机森林
    xgb = XGBR().fit(x_train,y_train)       #xgboost
   
    
    score_c = clf.score(x_test,y_test) 
    score_r = rfc.score(x_test,y_test)
    score_x = xgb.score(x_test,y_test) 
    predit_c = clf.predict(x_test)
    predit_e = rfc.predict(x_test)
    predit_x = clf.predict(x_test)
    out_c.append(predit_c)
    out_e.append(predit_e)
    out_x.append(predit_x)    
    y_real.append(y_test)
    
    d_c.append(abs((predit_c - y_test)/min(predit_c, y_test)))
    d_e.append(abs((predit_e - y_test)/min(predit_e, y_test)))
    d_x.append(abs((predit_x - y_test)/min(predit_x, y_test)))
    
    print("Single  Tree:{}".format(score_c)
    ,"Random Forest:{}".format(score_r)
    )




















