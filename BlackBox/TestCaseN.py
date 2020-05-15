from gplearn.genetic import SymbolicRegressor
import math
from sklearn.utils.random import check_random_state
from sklearn.metrics import mean_absolute_error as mae
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import gplearn.functions as gpn

rng = check_random_state(0)

#Create the base data
x = rng.uniform(0.001, 10, 50)
y = rng.uniform(-10, -0.001, 50)
realSet = []
for i in range (len(x)):
    realSet.append([x[i], y[i]])
x, y = np.meshgrid(x, y)
z = (x/y) + gpn.cos1(2*x) - gpn.sin1(3*y) * gpn.max2(6*x + 10, 2*y -8)
- gpn.min2(y + 5, 3*x) + gpn.inv1(y) - gpn.tan1(x) * gpn.log1(x)
+ gpn.sqrt1(x) * gpn.abs1(-2*x)

'''
ax = plt.figure().gca(projection='3d')
ax.set_xlim(0, 10)
ax.set_ylim(-10, 0)
surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, color='green', alpha=0.5)
#plt.show()
'''

#Create Training sample
x_train = rng.uniform(0.001, 10, 50)
y_train = rng.uniform(-10, -0.001, 50)
trainSet = []
for i in range (len(x_train)):
    trainSet.append([x_train[i], y_train[i]])
z_train = (x_train/y_train) + gpn.cos1(2*x_train) - gpn.sin1(3*y_train) * gpn.max2(6*x_train + 10, 2*y_train -8)
- gpn.min2(y_train + 5, 3*x_train) + gpn.inv1(y_train) - gpn.tan1(x_train) * gpn.log1(x_train)
+ gpn.sqrt1(x_train) * gpn.abs1(-2*x_train)

#Create Testing sample
x_test = rng.uniform(0.001, 10, 50)
y_test = rng.uniform(-10, -0.001, 50)
testSet = []
for i in range (len(x_test)):
    testSet.append([x_test[i], y_test[i]])
z_test = (x_test/y_test) + gpn.cos1(2*x_test) - gpn.sin1(3*y_test) * gpn.max2(6*x_test + 10, 2*y_test -8)
- gpn.min2(y_test + 5, 3*x_test) + gpn.inv1(y_test) - gpn.tan1(x_test) * gpn.log1(x_test)
+ gpn.sqrt1(x_test) * gpn.abs1(-2*x_test)

#print(trainSet)

est_gp = SymbolicRegressor(population_size=5000,
                           generations=100, stopping_criteria=0.01,
                           p_crossover=0.7, p_subtree_mutation=0.1,
                           p_hoist_mutation=0.05, p_point_mutation=0.1,
                           max_samples=0.9, verbose=1,
                           parsimony_coefficient=0.01, random_state=0,
                           function_set=('add', 'sub', 'mul', 'div', 'sqrt', 'log'
                                         , 'abs', 'neg', 'inv', 'max', 'min', 'sin', 'cos', 'tan'))
est_gp.fit(trainSet, z_train)
print(est_gp._program)


score_gp = est_gp.score(testSet, z_test)
                   
#score_gp = est_gp.mean_absolute_error(testSet, z_test)

print(score_gp)
#19 generations required
#min(add(add(log(add(inv(div(sin(X0), mul(X0, 0.952))),neg(abs(X0)))), neg(inv(div(sin(X1), mul(X0, 0.952))))), min(add(log(add(min(mul(-0.020, X1), cos(X1)), neg(add(log(add(min(inv(div(sin(X1), mul(X0, 0.952))), cos(X1)), div(sin(X1), add(log(add(min(mul(-0.020, X1), cos(X1)), neg(add(log(cos(X1)), neg(0.952))))), add(tan(tan(sin(X1))), neg(inv(div(sin(X1), neg(div(X1, 0.794)))))))))), neg(0.952))))), add(tan(tan(sin(X1))), neg(inv(div(sin(X1), neg(div(X1, 0.794))))))), div(neg(cos(tan(X1))), inv(mul(add(X1, X1), log(X1)))))), div(neg(cos(inv(mul(add(X1, X1), log(X1))))), inv(mul(add(X1, X1), log(X1)))))

z_gp = est_gp.predict(np.c_[x.ravel(), y.ravel()])
#print(z_gp)
ax = plt.figure().gca(projection = '3d')
ax.set_xlim(0,10)
ax.set_ylim(-10, 0)
surf = ax.plot_trisurf(x_test, y_test, z_gp, color='green')
points = ax.scatter(x_train, y_train, z_train)
#score = ax.text(-.7, 1, .2, "$R^2 =\/ %.6f$" % score_gp, 'x', fontsize=14)
title = "Symbolic Regressor"
plt.title(title)
plt.show()

surf = ax.plot_trisurf(x, y, z, color='green')
points = ax.scatter(x_train, y_train, z_train)
title = "Ground Truth"
plt.title(title)
plt.show()
