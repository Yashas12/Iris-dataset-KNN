import numpy as np
import pandas as pd
import matplotlib.pyplot as pp

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

names = ['sepal-length','sepal-width','petal-length','petal-width','Class']

data = pd.read_csv(url,names=names)

#print(data)

x = data.iloc[:,:-1].values
#print(x)

y = data.iloc[:,-1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.20)

print(xtrain)
print("\n")
print(xtest)

from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
scalar.fit(xtrain)

xtrain = scalar.transform(xtrain)
print(xtrain)
print("\n")
xtest = scalar.transform(xtest)
print(xtest)

from sklearn.neighbors import KNeighborsClassifier
clas = KNeighborsClassifier(n_neighbors=12)
clas.fit(xtrain,ytrain)

ypred = clas.predict(xtest)
#print(ypred)

from sklearn.metrics import confusion_matrix
cnf = confusion_matrix(ytest,ypred)

import seaborn as sb
sb.heatmap(cnf,annot=True,fmt="g")
pp.show()
