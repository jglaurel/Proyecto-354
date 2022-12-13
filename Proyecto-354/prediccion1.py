# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 22:54:50 2022

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from sklearn import linear_model
from sklearn.metrics import r2_score

dt = pd.read_csv("D:\\Trabajos U\\354\Proyecto-354\\world.csv",header=0)

#dt.shape
#print(dt.groupby('Year').size())
#añadir una nueva columna para definir en valores mas simples los años
dt['year2']=dt.groupby('Year')['Year'].transform(lambda x:x-1950)

#aplicar un modelo de regresion lineal
#seleccionar caracteristicas
cda= dt[['year2','Population']]
cda.head(50)

#graficos caracteristicas
viz = cda[['year2','Population']]
viz.hist()
plt.show()

#grafico año vs poblacion
plt.scatter(cda.year2, cda.Population, color='blue')
plt.xlabel("Año")
plt.ylabel("Poblacion")
plt.show()

#mascara para seleccionar el 80% de datos al azar para el entrenamiento
msk = np.random.rand(len(dt)) < 0.8
entreno= cda[msk]
test=cda[~msk]

#modelamos los datos en una regresion lineal
regr=linear_model.LinearRegression()
entre_x=np.asanyarray(entreno[['year2']])
entre_y=np.asanyarray(entreno[['Population']])
regr.fit(entre_x,entre_y) 

#pendiente y la interseccion
print("Coeficiente: ",regr.coef_) #mostramos theta
print("Interseccion: ",regr.intercept_)

#mostrar la linea de ajuste lineal
plt.scatter(entreno.year2, entreno.Population, color='orange')
plt.plot(entre_x,regr.coef_[0][0]*entre_x + regr.intercept_[0],'-r')
plt.xlabel("Año")
plt.ylabel("Poblacion")
plt.show()

#evaluar modelo utilizando error cuadratico medio
test_x=np.asanyarray(test[['year2']])
test_y=np.asanyarray(test[['Population']])
testeo=regr.predict(test_x)

#print metricas
print("Error cuadratico medio: %.2f" % np.mean(np.absolute(testeo - test_y)))
print("suma cuadratica: %.2f" % np.mean((testeo - test_y) **2))
print("R2: %.2f" % r2_score(test_y,testeo))

#convertir datos categoricos a numericas
#pd.get_dummies(dt,columns=["Country"])





