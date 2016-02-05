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
N = 2000
feats = 2
M = 2
training_steps = 10000

# Prepare data
D = prepare(N, feats)

# Show data
show(D)

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
m = M
w = theano.shared(rng.randn(feats), name="w")
u = theano.shared(rng.randn(feats), name="u")
b = theano.shared(0., name="b")
a1 = 0.0
a2 = 0.01
print("Initial model:")
print(w.get_value())
print(b.get_value())

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))  # Probability that target = 1
prediction = p_1 > 0.5  # The prediction thresholded
xent = -y * T.log(p_1) - (1 - y) * T.log(1 - p_1)  # Cross-entropy loss function
l2 = (w ** 2).sum()
l1 = (abs(w)).sum()
cost = xent.mean() + a2 * l2 + a1 * l1  # The cost to minimize
gw, gb = T.grad(cost, [w, b])  # Compute the gradient of the cost
# (we shall return to this in a
# following section of this tutorial)

# Compile
train = theano.function(
        inputs=[x, y],
        outputs=[prediction, xent],
        updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[:, 0:2], D[:, 2])

# Predict
D[:, 2] = predict(D[:, 0:2])

# Print model
print("Final model:")
print(w.get_value())
print(b.get_value())

# Show result
show(D)

