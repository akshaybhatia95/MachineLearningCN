import numpy as np
data=np.loadtxt("D:\\MachineLearningCN\\New folder\\data.csv",delimiter=",")
x=data[:,0]
y=data[:,1]
from sklearn import model_selection
xtrain,xtest,ytrain,ytest=model_selection.train_test_split(x,y,test_size=0.3)

def fit(x,y):
    num= (x*y).mean() - x.mean() * y.mean()
    den= (x*x).mean() - x.mean() * x.mean()
    m=num/den
    c=y.mean() - m * x.mean()
    return m,c

def predict(x,m,c):
    y=m * x + c
    return y

def score(ypred,ytruth):
    u= ((ytruth - ypred)**2).sum()
    v= ((ytruth-ytruth.mean())**2).sum()
    return 1-u/v

def cost(x,y,m,c):
    cost=((y-(m*x+c))**2).mean()
    return cost
    
m,c=fit(xtrain,ytrain)
ypred=predict(xtest,m,c)
print("Score: ",score(ypred,ytest))
print("M,C: ",m,c)
print("COST: ", cost(xtrain,ytrain,m,c))
print("COST: ", cost(xtrain,ytrain,m+1,c)) # high value for cost by deviating by only one