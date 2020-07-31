import json
x = '"{}"'
y = '{}'
jx = json.loads(x)
jy = json.loads(y)
print(jx)
print(jy)
print(type(jx), type(jy))

# RESULT:
# {}
# {}
# <class 'str'> <class 'dict'>
