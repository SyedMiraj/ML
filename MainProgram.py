from Perceptron import Perceptron as pt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

if __name__ == '__main__':
    iris = pd.read_csv("iris_data.csv")
    y = iris.iloc[0:100, 4].values
    
    y = np.where(y == "Iris-setosa", -1, 1)
    X = iris.iloc[0:100, [0,2]].values
    
    # =============================================================================
    # plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker = 'o', label='Setosa')
    # plt.scatter(X[50:100, 0], X[50:100, 1], color = 'green', marker = 'x', label='Versicolor')
    # plt.xlabel('Petal Length')
    # plt.ylabel('Sepal Length')
    # plt.legend(loc = 'upper left')
    # plt.show()
    # =============================================================================
    
    # =============================================================================
    # plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker = 'o')
    # plt.xlabel('Epochs')
    # plt.ylabel('Number of Misclassification')
    # plt.show()
    # =============================================================================
    
    # =============================================================================
    # Color Map Graphics
    # =============================================================================
    
    def plot_decision_region(X, y, classifier, resulution = 0.02):
        markers = ['s', 'x', 'o', '^', 'v']
        colors = ['red', 'blue', 'lightgreen', 'gray', 'cyan']
        cmap = ListedColormap(colors[:len(np.unique(y))])
        
        #plot the decision surface
        x1_min, x1_max = X[:, 0].min() -1, X[:, 0].max()+1
        x2_min, x2_max = X[:, 1].min() -1, X[:, 1].max()+1
        
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resulution), 
                               np.arange(x2_min, x2_max, resulution))
        Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        
        # plot class sample
        for idx, c1 in enumerate(np.unique(y)):
            plt.scatter(x=X[y == c1,0], y=X[y==c1, 1], alpha = 0.8, c=cmap(idx), 
                        marker = markers[idx], label = c1)
            
    
    ppn = pt(eta=.1, n_iter = 10)
    ppn.fit(X, y)   
    plot_decision_region(X, y, classifier=ppn)
    plt.xlabel('sepal length [cm]')
    plt.ylabel('petal length [cm]')
    plt.legend(loc = 'upper left')
    plt.show()    