import pandas as pd 
import numpy as np
from sklearn.metrics import r2_score 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import preprocessing
from statistics import mean
import matplotlib.pyplot as plt


df_train = pd.read_csv('../../data/random_linear_train.csv')
df_test = pd.read_csv('../../data/random_linear_test.csv')

x_train = df_train['x']
y_train = df_train['y']
x_test = df_test['x']
y_test = df_test['y']

def best_fit_slope_and_intercept(xs,ys):
    print("Mean: ", xs*ys)

    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    
    b = mean(ys) - m*mean(xs)

    return m, b


def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]

    squared_error_regr = sum((ys_line - ys_orig) * (ys_line - ys_orig))
    squared_error_y_mean = sum((y_mean_line - ys_orig) * (y_mean_line - ys_orig))

    r_squared = 1 - (squared_error_regr/squared_error_y_mean)

    return r_squared

print(x_train)
print()
print(y_train)

m, b = best_fit_slope_and_intercept(x_train,y_train)
print(m, b)
regression_line = [(m*x)+b for x in x_test]
r_squared = coefficient_of_determination(y_test,regression_line)
print("r2 score from built model: ", r_squared)

x_train = np.array(x_train).reshape(-1, 1)
y_train = np.array(y_train).reshape(-1, 1)
x_test = np.array(x_test).reshape(-1, 1)
y_test = np.array(y_test).reshape(-1, 1)

clf = LinearRegression(normalize=True)
clf.fit(x_train,y_train)
y_pred = clf.predict(x_test)
print("r2 score from sklearn: ", r2_score(y_test,y_pred))

plt.scatter(x_test, y_test, color='black')
plt.plot(x_test, regression_line, color='blue', linewidth=3)
plt.xticks(())
plt.yticks(())
plt.savefig("graph.png")