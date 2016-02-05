import numpy as np
import theano
import theano.tensor as T
import pylab as pl


def prepare(num, features):
    data = rng.rand(num, features+1)
    for i in range(num):
        t = data[i]
        if 1.5 > t[0] + t[1] > 0.5 > t[1] - t[0] > -0.5:
            data[i][2] = 1
        else:
            data[i][2] = 0
    return data


def show(data):
    red = np.array(list(filter(lambda t: t[2] == 1, data)))
    blue = np.array(list(filter(lambda t: t[2] == 0, data)))

    pl.plot(red[:, 0], red[:, 1], 'or')
    pl.plot(blue[:, 0], blue[:, 1], 'ob')
    pl.show()

rng = np.random
# Define parameters
N = 3000
feats = 2
M = 2
training_steps = 200

# Prepare data
D = prepare(N, feats)

# Show data
# show(D)

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
m = M
w = theano.shared(rng.randn(feats, m), name="w")
u = theano.shared(rng.randn(feats, m), name="u")
b = theano.shared(np.zeros(m), name="b")
a1 = 0.0
a2 = 0.01
print("Initial model:")
print(w.get_value())
print(u.get_value())
print(b.get_value())

# Construct Theano expression graph
p_1 = ((T.log(1 + T.exp(T.dot(x, u))) / T.reshape(T.log(1 + T.exp(T.dot(x, u))).sum(axis=1), [N, 1])) / (1 + T.exp(-T.dot(x, w) - b))).sum(axis=1)  # Probability that target = 1
p_2 = (1 + T.exp(-T.dot(x, w) - b)).sum(axis=1)
p_3 = (T.exp(T.dot(x, u)) / T.reshape(T.exp(T.dot(x, u)).sum(axis=1), [N, 1]))
prediction = p_1 > 0.5  # The prediction thresholded
# print(theano.printing.pprint(prediction))
# theano.printing.pydotprint(prediction, outfile="pics/mlr_prediction.png", var_with_name_simple=True)
xent = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)  # Cross-entropy loss function
l2 = T.sqrt((w ** 2 + u ** 2).sum(axis=0)).sum()
l1 = abs(w).sum() + abs(u).sum()
cost = xent.mean() + a2 * l2 + a1 * l1  # The cost to minimize
gw, gu, gb = T.grad(cost, [w, u, b])  # Compute the gradient of the cost
# (we shall return to this in a
# following section of this tutorial)
print(theano.printing.pprint(gw))
# Compile
train = theano.function(
        inputs=[x, y],
        outputs=[prediction, xent],
        updates=((w, w - 0.1 * gw), (u, u - 0.1 * gu), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Function test
# test = theano.function(inputs=[x], outputs=p_3)
# result = test(D[:, 0:2])
#

# Train
for i in range(training_steps):
    pred, err = train(D[:, 0:2], D[:, 2])

# Predict
D[:, 2] = predict(D[:, 0:2])

# Print model
print("Final model:")
print(w.get_value())
print(u.get_value())
print(b.get_value())

# Show result
show(D)

