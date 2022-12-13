# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 21:24:32 2022

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 
%matplotlib inline

dt = pd.read_csv("D:\\Trabajos U\\354\Proyecto-354\\world.csv")

X = dt.iloc[:,0:7]. values
y = dt.iloc[:,7].values

X_std=StandardScaler().fit_transform(X)

print('Matriz de covarianza: \n%s' %np.cov(X_std.T))
matriz_cov=np.cov(X_std.T)

eig_vals,eig_vecs=np.linalg.eig(matriz_cov)

print('Eigen Vectores \n%s' %eig_vecs)
print('Eigen Valores  \n%s' %eig_vecs)

#reducir dimensionalidad del dataset
eig_par=[(np.abs(eig_vals[i]),eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_par.sort(key=lambda x:x[0],reverse=True)

print('Autovalores en orden descendente:')
for i in eig_par:
    print(i[0])

#calcular varianza 
tot=sum(eig_vals)
var_exp=[(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp=np.cumsum(var_exp)

#grafico
"""
with plt.style.context('seaborn-pastel'):
    plt.figure(figsize=(7, 4))

    plt.bar(range(4), var_exp, alpha=0.5, align='center',
            label='Varianza individual explicada', color='g')
    plt.step(range(4), cum_var_exp, where='mid', linestyle='--', label='Varianza explicada acumulada')
    plt.ylabel('Ratio de Varianza Explicada')
    plt.xlabel('Componentes Principales')
    plt.legend(loc='best')
    plt.tight_layout()
"""

#generar la matriz a partir de los pares autovalor-autovector
matriz_w=np.hstack((eig_par[0][1].reshape(7,1),
                    eig_par[1][1].reshape(7,1)))
print('Matriz W:\n', matriz_w)

Y=X_std.dot(matriz_w)

#grafico 
"""
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(7, 4))
    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('magenta', 'cyan', 'limegreen')):
        plt.scatter(Y[y==lab, 0],
                    Y[y==lab, 1],
                    label=lab,
                    c=col)
    plt.xlabel('Componente Principal 1')
    plt.ylabel('Componente Principal 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()  
"""






