# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 11:00:33 2022

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn import linear_model
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
import statsmodels.formula.api as smf

dt = pd.read_csv("D:\\Trabajos U\\354\Proyecto-354\\world.csv",header=0)

#dt.shape
#print(dt.groupby('Year').size())
#añadir una nueva columna para definir en valores mas simples los años
dt['year2']=dt.groupby('Year')['Year'].transform(lambda x:x-1950)

#division de los datos en train y test
X= dt[['year2']]
y=dt['Population']
x_train,x_test,y_train,y_test= train_test_split(
            X.values.reshape(-1,1),
            y.values.reshape(-1,1),
            train_size=0.8,
            random_state=1234,
            shuffle= True)

#crear el modelo
modelo=LinearRegression()
modelo.fit(X=x_train.reshape(-1,1), y=y_train)

#informacion del modelo
print("Intercept:",modelo.intercept_)
print("Coeficiente: ", list(zip(X.columns,modelo.coef_.flatten(),)))
print("Coeficiente de determinacion R cuadrado: ", modelo.score(X, y))

#Error de test del modelo
predicciones= modelo.predict(X=x_test)
print(predicciones[0:3,])

rmse=mean_squared_error(
    y_true=y_test,
    y_pred=predicciones,
    squared=False)
print("----------")
print(f"El error (rmse) de test es: {rmse}")







