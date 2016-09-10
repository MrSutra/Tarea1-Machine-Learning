import pandas as pd
import numpy as np
import sklearn.linear_model as lm
from scipy.sparse import csr_matrix
from scipy.io import mmread
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# loading data
# training set
X_train = csr_matrix(mmread('train.x.mm'))
y_train = np.loadtxt('train.y.dat')
# test set
X_test = csr_matrix(mmread('test.x.mm'))
y_test = np.loadtxt('test.y.dat')
# CV
X_cv = csr_matrix(mmread('dev.x.mm'))
y_cv = np.loadtxt('dev.y.dat')


# Regresion lineal
model = lm.LinearRegression(fit_intercept = False)
model.fit(X_train,y_train)
print "Regresion lineal\nCV: ",model.score(X_cv,y_cv),"\nTest: ",model.score(X_test, y_test)

# ElasticNet
print "\nElasticNet"
alpha = [0.1, 0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1.0]
cr2 = []
tr2 = []

for a in alpha:
	# modelo con elasticnet
	# se calcula el valor de R2 o coeficiente de determinacion
	enet = lm.ElasticNet(fit_intercept = False)
	enet.set_params(alpha= a, l1_ratio=0.9, max_iter = 1000)
	enet.fit(X_train, y_train)
	
	cvR2 = enet.score(X_cv,y_cv)
	testR2 = enet.score(X_test, y_test)
	cr2.append(cvR2)
	tr2.append(testR2)
	print "alpha: ",a,"\nCV: ",cvR2,"\nTest: ",testR2
	
plt.plot(alpha,cr2,'ro')
plt.xlabel("alpha")
plt.ylabel("R2")
plt.show()