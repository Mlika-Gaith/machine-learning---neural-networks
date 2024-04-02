from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

cancer = datasets.load_breast_cancer()

"""print(cancer.feature_names)
print(cancer.target_names)
"""
x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

"""print(x_train)
print(y_train)"""
classes = ['malignant' 'benign']

classifier = svm.SVC(kernel='linear', C=5)
classifier.fit(x_train, y_train)

predictions = classifier.predict(x_test)
accuracy = metrics.accuracy_score(y_test, predictions)

print(f"accuracy : {accuracy}")