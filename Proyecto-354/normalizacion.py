from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from matplotlib import gridspec
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dt = pd.read_csv("D:\\Trabajos U\\354\Proyecto-354\\world.csv",header=0)
dt['year2']=dt.groupby('Year')['Year'].transform(lambda x:x-1950)
pd.get_dummies(dt,columns=["Country"])
#dt=dt.to_csv(r'D:\Trabajos U\354\Proyecto-354\world.csv',index=False)

#dt=dt.replace(np.nan, '0')
df=pd.DataFrame(dt)

print("Media:")
print(df['Population'].mean())

print("Desviacion estandar:")
print(df['Population'].std())

print("Valor maximo:")
print(df['Population'].max())

print("Valor minimo:")
print(df['Population'].min())

fig,(ax1,ax2,ax3)=plt.subplots(ncols=3,figsize=(6,5))
ax1.set_title("Antes de la normalizacion")
sns.kdeplot(df['Population'],ax=ax1)
sns.kdeplot(df['year2'],ax=ax1)
sns.kdeplot(df['Median Age'],ax=ax1)

#scaler=preprocessing.StandardScaler()
scaler=preprocessing.Normalizer(norm='l2',copy=True)

df[['year2','Population']]=scaler.fit_transform(df[['year2','Population']])

print("--------------------------")
print(df['Population'].iloc[0])
print("Media:")
print(df['Population'].mean())

print("Desviacion estandar:")
print(df['Population'].std())

print("Valor maximo:")
print(df['Population'].max())

print("Valor minimo:")
print(df['Population'].min())

df.to_csv('poblacion2',sep='\t')
fig,(ax1,ax2,ax3)=plt.subplots(ncols=3,figsize=(6,5))
ax2.set_title("Despues de la normalizacion")
sns.kdeplot(df['Population'],ax=ax2)
sns.kdeplot(df['year2'],ax=ax2)
sns.kdeplot(df['Median Age'],ax=ax2)

plt.show()

