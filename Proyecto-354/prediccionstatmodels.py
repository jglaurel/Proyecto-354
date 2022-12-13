# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 11:38:51 2022

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

#crear el modelo utilizando matrices
x_train= sm.add_constant(x_train,prepend=True)
modelo = sm.OLS(endog=y_train, exog=x_train,)
modelo=modelo.fit()
print(modelo.summary())
print("--------------------------------------------")
#intervalos de confianza
intervalos_ci=modelo.conf_int(alpha=0.05)
intervalos_ci.columns = ['2.5%','97.5%']
intervalos_ci

#predicciones con intervalo de confianza del 95%
predicciones=modelo.get_prediction(exog=x_train).summary_frame(alpha=0.1)
print(predicciones.head(4))


