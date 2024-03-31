import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("datasets/student-mat.csv", sep=";")
"""with pd.option_context('display.max_rows', 1,
                       'display.max_columns', 33,
                       'display.precision', 3,
                       ):
    print(data)"""

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
predict = "G3"
x = np.array(data.drop([predict],axis=1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)
"""
best = 0
for i in range(30):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1)

    reg = linear_model.LinearRegression()
    reg.fit(x_train, y_train)
    acc = reg.score(x_test, y_test)
    print(f"accuracy : {acc}")
    #print(f"coefficient of determination: {reg.coef_}")
    #print(f"Intercept: {reg.intercept_}")
    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(reg, f)
"""
pickle_in = open("studentmodel.pickle", "rb")
reg = pickle.load(pickle_in)
y_pred = reg.predict(x_test)

for x in range(len(y_pred)):
    print(y_pred[x], x_test[x], y_test[x])

p = "G1"
style.use("ggplot")
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()