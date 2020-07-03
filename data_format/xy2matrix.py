# -*- coding: utf-8 -*-

import sys

data_dict = {}
y_set = set()

with open(sys.argv[1]) as fp:
    for line in fp:
        x, y, v = line[:-1].split(',')
        y_set.add(y)
        if x not in data_dict:
            data_dict[x] = {}
        data_dict[x][y] = v

print('-', end=',')
for yi in y_set:
    print(yi, end=',')
print()
for i in data_dict:
    print(i, end=',')
    for j in y_set:
        if j not in data_dict[i]:
            data_dict[i][j] = 0
        print(data_dict[i][j], end=',')
    print()
