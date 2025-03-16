# Ridge-Regression-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes

data=load_diabetes()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.2,random_state=42)

from sklearn.linear_model import LinearRegression
L = LinearRegression()

L.fit(x_train,y_train)

y_pred = L.predict(x_test)

from sklearn.metrics import r2_score,mean_squared_error
print("R2 score",r2_score(y_test,y_pred))
print("MSE is:",mean_squared_error(y_test,y_pred))

from sklearn.linear_model import Ridge
R = Ridge(alpha = 0.0001)

R.fit(x_train,y_train)

y_pred1 = R.predict(x_test)

print("R2 score",r2_score(y_test,y_pred1))#find the r2 score
print("MSE is:",mean_squared_error(y_test,y_pred1))

m=100
x1=5 * np.random.rand(m, 1) - 2
x2 = 0.7 * x1 ** 2 - 2 * x1 + 1 + np.random.randn(m, 1)
plt.scatter(x1, x2)
plt.show()

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

def get_preds_ridge(x1, x2, alpha):
    model = Pipeline([
        ('poly_feats', PolynomialFeatures(degree=16)),
        ('ridge', Ridge(alpha=alpha))
    ])
    model.fit(x1, x2)
    return model.predict(x1)

alphas = [0, 20, 200]
cs = ['r', 'g', 'b']

plt.figure(figsize=(10, 6))
plt.plot(x1, x2, 'b+', label='Datapoints')

for alpha, c in zip(alphas, cs):
    preds = get_preds_ridge(x1, x2, alpha)
    # Plot
    plt.plot(sorted(x1[:, 0]), preds[np.argsort(x1[:, 0])], c, label='Alpha: {}'.format(alpha))

plt.legend()
plt.show()

from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test, y_pred)
print("MAE is:", mae)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE is:", rmse)

mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
print("MAPE is:", mape, "%")

adjusted_r2 = 1 - (1 - r2_score(y_test, y_pred)) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1)
print("Adjusted R2 score:", adjusted_r2)

from sklearn.model_selection import cross_val_score

cross_val_scores = cross_val_score(L, data.data, data.target, cv=5, scoring='r2')
print("Cross-Validation R2 scores:", cross_val_scores)
print("Mean CV R2:", np.mean(cross_val_scores))

