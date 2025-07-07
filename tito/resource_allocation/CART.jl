
py"""
import numpy as np
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error



def getLeav(regressor,Xnew,X,y):
    Xnew = [i for i in Xnew] #[Xnew]
    #print("prediccion: ",regressor.predict(Xnew))
    idx = regressor.apply(Xnew)[0]
    #print("Nodo en el que term  ina el nuevo feature (idx)",idx)

    children_left = regressor.tree_.children_left
    children_right = regressor.tree_.children_right
    #Basado en https://datascience.stackexchange.com/questions/87122/how-to-obtain-the-final-values-of-a-decisiontreeregressor-in-scikit-learn
    leaf_nodes = []
    for i in range(len(children_left)):
        if children_left[i] == children_right[i]:
            leaf_nodes.append(i)

    all_val = regressor.tree_.value[leaf_nodes,0,0] 
    #print("Obtengo el ID de las hojas",leaf_nodes)
    #print("Obtengo los valores mean de cada hoja",all_val)
    #print("Obtengo los Nsamples de cada hoja",regressor.tree_.n_node_samples[leaf_nodes])
    
    
    on_leaf = regressor.apply(X)
    x_leaves = []
    y_leaves = []
    index = leaf_nodes.index(idx)
    
    for i in np.unique(on_leaf):
        x_leaves.append(X[np.argwhere(on_leaf==i)]) 
        y_leaves.append(y[np.argwhere(on_leaf==i)]) 

    return x_leaves[index], y_leaves[index],regressor.predict(Xnew)


def getRegressor(X_train,y_train,X_test,y_test):
    regressor = DecisionTreeRegressor(criterion="squared_error",min_samples_leaf=25).fit(X_train, y_train)

    y_pred = regressor.predict(X_test)
    print("score ", regressor.score(X_test, y_test))
    print("square ",mean_squared_error(y_test, y_pred))
    #print(cross_val_score(regressor, X_test, y_test, cv=5,scoring="neg_mean_squared_error" ))
    return regressor


"""

#=
J = 30
I_prods = 20
ω = 1
L = 3
σ = 5
p = 2
rep = 100
N_insample = 1000
N_outofsample = 1000*30
#------ generamos matrices para crear data.

ϕ,ζ,Σ = sampleParameters(J,σ,ω)
Y,Yₒₒₛ,X,X_new =  dataGeneration(N_insample,N_outofsample,ϕ,ζ,σ,J,p,L,Σ)


println("..............................")
X = X
y = transpose(Y)

X_train, X_test, y_train, y_test = py"train_test_split"(X, y, test_size=0.2,random_state=42)


tree = py"getRegressor"(X_train,y_train,X_test, y_test) #py"getRegressor"(data_xₜ,transpose(data_yₜ))
getLeav = py"getLeav"(tree,X_new,X_train,y_train) #esto me entrega Y(Nsamples,1)xJ y X(Nsamples,1)xL

#Transformo la formato del AD

X_hoja = reshape(getLeav[1][:,:,:],(length(getLeav[1][:,:,1]),L))
Y_hoja = reshape(getLeav[2][:,:,:],(length(getLeav[2][:,:,1]),J)) 

=#