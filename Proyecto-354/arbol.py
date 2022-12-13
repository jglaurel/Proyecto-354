"""
Created on Tue Dec 13 00:07:08 2022

@author: DELL
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from sklearn.tree import export_graphviz
import graphlib
import matplotlib.pyplot as plt

dt = pd.read_csv("D:\\Trabajos U\\354\Proyecto-354\\world.csv",header=0)
dt['year2']=dt.groupby('Year')['Year'].transform(lambda x:x-1950)

#variables predictoras
predictors=dt[['year2','Median Age','Fertility Rate']]
targets=dt.Population
#print(predictors)

predictors_labels=[['year2','Median Age','Fertility Rate']]
target_labels=[['Population']]

#variables para el entrenamiento del arbol
X_entreno,X_test,y_entreno,y_test=train_test_split(predictors,targets)

#construir el arbol con los datos
arbol=DecisionTreeClassifier()
arbol.fit(X_entreno,y_entreno)

#verifica probabilidad de prediccion del arbol
arbol.score(X_test, y_test)

#se genera la grafica del arbol
export_graphviz(arbol,out_file='arbol.dot',
                class_names=target_labels,
                feature_names=predictors_labels,
                impurity=False,filled=True)
with open('arbol.dot') as f:
    dot_graph=f.read()
graphlib.Source(dot_graph)

#para ver la importancia de las caracteristicas

caract=8
plt.barh(range(caract), arbol.feature_importances_)
plt.yticks(np.arange(caract),predictors_labels)
plt.xlabel('importancia de las caracteristicas')
plt.ylabel('Caracteristicas')
plt.show()

#profundidad del arbol
arbol.tree_.max_depth()






