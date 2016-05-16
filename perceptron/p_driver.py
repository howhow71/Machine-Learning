from Perceptron import Perceptron
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
def main():
    print("Percepton Test File")    
    #https://archive.icu.uci.edu/ml/machine-learning-databases/iris/iris.data
    df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',header=None)
    print df.tail()
    y = df.iloc[0:100,4].values
    y = np.where(y == 'Iris-setosa',-1,1)
    X = df.iloc[0:100,[0,2]].values
    print y
    print X
    plt.scatter(X[:50,0],X[:50,1],color='red',marker='o',label="setosa")
    plt.scatter(X[50:100,0],X[50:100,1],color='blue',marker='x',label='versicolor')
    plt.xlabel('petal length')
    plt.ylabel('sepal length')
    plt.legend(loc='upper left')
    #plt.show()
    ppn = Perceptron(eta=0.1,n_iter=10)
    ppn.fit(X,y)
    plt.figure()
    plt.plot(range(1,len(ppn.errors_)+1),ppn.errors_,marker='o')
    #plt.show()
    plot_decision_reg(X,y,classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc='upper left')
    plt.show()



def plot_decision_reg(X,y,classifier,resolution=0.02):
    plt.figure()
    #setup marker gen and color map
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','gray','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    #plot the decision surface
    x1_min,x1_max = X[:,0].min() - 1,X[:,0].max() +1
    x2_min,x2_max = X[:,1].min() - 1,X[:,1].max() + 1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1,xx2,Z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    for idx,c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1,0],y=X[y == c1,1],alpha=0.8,c=cmap(idx),marker=markers[idx],label=c1)
main()
