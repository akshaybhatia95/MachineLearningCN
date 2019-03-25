def cost(data,m,c):
    total_cost=0
    M=len(data)
    for i in range(M):
        x=data[i,0]
        y=data[i,1]
        total_cost+= (1/M)*((y-m*x-c)**2)
    return total_cost    
def stepGradient(data,learning_rate,m,c):
    m_slope=0
    c_slope=0
    N=len(data)
    for i in range(N):
        x=data[i,0]
        y=data[i,1]
        m_slope+=(-2/N) * (y-m*x-c)*x
        c_slope+=(-1/N)* (y-m*x-c)
    new_m= m - learning_rate* m_slope
    new_c= c - learning_rate* c_slope
    return new_m,new_c
def gradientDescent(data,learning_rate,num_iterations):
#   Start with a random value of m & c
    m=0
    c=0
#     x=data[:,0]
#     y=data[:,1]
    
    for i in range(num_iterations):
        m,c=stepGradient(data,learning_rate,m,c)
        print(i," ",cost(data,m,c))
    return m,c  
def run():
    import numpy as np
    from sklearn import model_selection
    data=np.loadtxt("D:\\MachineLearningCN\\New folder\\data.csv",delimiter=",")
#     x=data[:,0]
#     y=data[:,1]
#     xtrain,xtest,ytrain,ytest=model_selection.train_test_split(x,y,test_size=0.3)
    learning_rate=0.0001
    num_iterations=100
    m,c=gradientDescent(data,learning_rate,num_iterations)
    return m,c

m,c=run()
print(m," ",c)