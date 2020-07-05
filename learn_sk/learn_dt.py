from sklearn import tree

x = [[0, 0], [1, 1], [2, 4], [3, 9], [4, 16], [5, 10], [3, 6], [4, 8]]
y = [1, 1, 1, 1, 1, 0, 0, 0]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x, y)
z = [[1.5, 2.25], [5, 25], [1.3, 1.7]]
print(clf.predict(z))

