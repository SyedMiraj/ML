from Perceptron import Perceptron
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris = pd.read_csv("iris_data.csv")
y = iris.iloc[0:100, 4].values

y = np.where(y == "Iris-setosa", -1, 1)
X = iris.iloc[0:100, [0,2]].values
print(X)
