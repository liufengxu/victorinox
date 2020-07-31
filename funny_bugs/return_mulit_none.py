def test_multi1():
    return 

def test_multi2():
    return None 

def test_multi3():
    return None, None 

try:
    x, y = test_multi1()
    print(x, y)
except Exception as e:
    print(e)

try:
    x, y = test_multi2()
    print(x, y)
except Exception as e:
    print(e)

try:
    x, y = test_multi3()
    print(x, y)
except Exception as e:
    print(e)

# RESULT
# 'NoneType' object is not iterable
# 'NoneType' object is not iterable
# None None
