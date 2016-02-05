
import numpy as np
import theano
import theano.tensor as T
from io import BytesIO

# declare two symbolic floating-point scalars
a = T.matrix('a')
b = T.matrix('b')

# create a simple expression
#c = T.exp(T.dot(a, b))
c = T.dot(a, b).sum()

# convert the expression into a callable object that takes (a,b)
# values as input and computes a value for c
f = theano.function([a, b], c)

# bind 1.5 to 'a', 2.5 to 'b', and evaluate 'c'
print(f([[2, 2], [3, 3]], [[1, 0], [1, 1]]))
print(np.exp(np.matrix([[4, 2], [6, 3]])))
assert np.exp(np.matrix([[4, 2], [6, 3]])) == f([[2, 2], [3, 3]], [[1, 1], [1, 1]])

data = "1, 2, 3\n4, 5, 6"
d = np.genfromtxt(BytesIO(data.encode()), delimiter=",")
print(d[0][0])

s = BytesIO("1,1.3,abcde\n1,1.2,asdfs".encode())
data = np.genfromtxt(s, dtype=[('myint', 'i8'), ('myfloat', 'f8'), ('mystring', 'S5')], delimiter=",")
print(data[0:2][0])

z1 = np.ones((3, 4))
z2 = np.ones((3, 4))
print(z1)
