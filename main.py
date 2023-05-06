import pandas as pd 
import numpy as np
from methods import *

iris_df = pd.read_csv("iris.csv")
#zmenit posledny stlpec na boolean
iris_df.loc[iris_df["species"] != "virginica","species"] = 0
iris_df.loc[iris_df["species"] == "virginica","species"] = 1

#y=bool array
y = np.array(iris_df["species"])
#X=matica vlastnosti kvetov
X_pre = iris_df.iloc[:,:4].to_numpy()
#pridat 1 do X
X = np.column_stack((X_pre,np.ones(X_pre.shape[0])))

#funkcia v maticovom tvare
def fun(beta):
    return ((y-X@beta)@np.ones(150))**2

#predratat c
c = -1*np.array([np.sum(X_pre[:, 0]), np.sum(X_pre[:, 1]), np.sum(X_pre[:, 2]), np.sum(X_pre[:, 3]),150])


#gradient funkcie v bode beta(b1,b2,b3,b4,const)
def dfun(beta, c = c):
    ones = np.ones(X.shape[0])
    first_part = (y-X@beta)@ones
    return 2*first_part*c

#Const
#########
# grad_const_iterations = []
# fill_grad_const = lambda x: grad_const_iterations.append(x)
# print(grad_const(dfun,np.array([0,0,0,0,0]),callback=fill_grad_const,options={"stepsize" : 0.01,"tol" : 1e-5}))
# #############

# #backtrack
# ##########
# print(backtrack(dfun,np.array([0,0,0,0,0]),args=[fun],options={"tol":1e-5}))
# #############

# #cauchy
# ##########
# print(cauchy(dfun,np.array([0,0,0,0,0]),args=[fun],options={"tol":1e-5}))
# print(dfun(np.array([0.08975855, 0.08975855, 0.08975855, 0.08975855, 0.08975855])))
#########
def classifier(x,beta):
    if x@beta >=0:
        return 1
    else:
        return 0

my_beta = cauchy(dfun,np.array([0,0,0,0,0]),args=[fun])
print(my_beta)
classificatedd = []
for row in X:
    xi = row
    c = classifier(xi,my_beta.x)
    classificatedd.append(c)


s = 0
for i, flower in enumerate(classificatedd):
    if flower==y[i]:
        s+=1
print(s/150)
