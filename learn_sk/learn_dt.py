from sklearn import tree, svm
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split
import pandas as pd

x = [[0, 0], [1, 1], [2, 4], [3, 9], [4, 16], [5, 10], [3, 6], [4, 8]]
y = [1, 1, 1, 1, 1, 0, 0, 0]
# clf = tree.DecisionTreeClassifier()
clf = linear_model.LogisticRegression()
clf = clf.fit(x, y)
z = [[1.5, 2.25], [5, 25], [1.3, 1.7]]
print(clf.predict(z))

# ts = pd.read_excel('/Users/liufengxu/Documents/guard/time_spend1.xlsx')

iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
     iris.data, iris.target, test_size=0.4, random_state=0)
# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
clf = tree.DecisionTreeClassifier().fit(X_train, y_train)
s = clf.score(X_test, y_test)
print(s)

